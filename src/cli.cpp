#include "inferedge_runtime/cli.hpp"

#include "inferedge_runtime/engine.hpp"
#include "inferedge_runtime/version.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferedge_runtime {
namespace {

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for option: " + option);
    }

    std::string value = argv[++index];
    if (value.empty() || value.rfind("--", 0) == 0) {
        throw std::invalid_argument("missing value for option: " + option);
    }

    return value;
}

bool is_supported_engine(const std::string& engine) {
    return engine == "onnxruntime" || engine == "ort";
}

bool is_supported_device(const std::string& device) {
    return device == "cpu";
}

void validate_engine(const std::string& engine) {
    if (!is_supported_engine(engine)) {
        throw std::invalid_argument("unsupported engine: " + engine + " (supported: onnxruntime, ort)");
    }
}

void validate_device(const std::string& device) {
    if (!is_supported_device(device)) {
        throw std::invalid_argument("unsupported device: " + device + " (supported: cpu)");
    }
}

int parse_int_with_minimum(const std::string& value, const std::string& option, int min_value) {
    try {
        std::size_t parsed_chars = 0;
        const int parsed_value = std::stoi(value, &parsed_chars);
        if (parsed_chars != value.size() || parsed_value < min_value) {
            throw std::invalid_argument("invalid integer value for option: " + option);
        }
        return parsed_value;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "invalid integer value for option: " + option + " (" + value + ", minimum: " +
            std::to_string(min_value) + ")");
    }
}

std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

void print_tensor_metadata_list(const std::vector<TensorMetadata>& tensors) {
    if (tensors.empty()) {
        std::cout << " []\n";
        return;
    }

    std::cout << '\n';
    for (const TensorMetadata& tensor : tensors) {
        std::cout
            << "    - name: " << tensor.name << '\n'
            << "      type: " << tensor.element_type << '\n'
            << "      shape: " << format_shape(tensor.shape) << '\n';
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream stream;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                stream << ch;
                break;
        }
    }
    return stream.str();
}

std::string json_string(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time{};
#if defined(_WIN32)
    gmtime_s(&utc_time, &time);
#else
    gmtime_r(&time, &utc_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string system_os_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#else
    return "unknown";
#endif
}

std::string compiler_name() {
#if defined(__apple_build_version__)
    return "AppleClang";
#elif defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

void write_shape_json(std::ostream& output, const std::vector<int64_t>& shape) {
    output << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << shape[i];
    }
    output << ']';
}

void write_double_vector_json(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << std::fixed << std::setprecision(6) << values[i];
    }
    output << ']';
}

void write_tensor_metadata_json(std::ostream& output, const std::vector<TensorMetadata>& tensors, int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const std::string item_indent(static_cast<std::size_t>(indent_spaces + 2), ' ');

    output << "[";
    if (!tensors.empty()) {
        output << '\n';
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const TensorMetadata& tensor = tensors[i];
        output
            << item_indent << "{\n"
            << item_indent << "  \"name\": " << json_string(tensor.name) << ",\n"
            << item_indent << "  \"element_type\": " << json_string(tensor.element_type) << ",\n"
            << item_indent << "  \"shape\": ";
        write_shape_json(output, tensor.shape);
        output << '\n' << item_indent << "}";
        if (i + 1 < tensors.size()) {
            output << ',';
        }
        output << '\n';
    }

    if (!tensors.empty()) {
        output << indent;
    }
    output << "]";
}

void write_result_json(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const ModelMetadata& model_metadata,
    const BenchmarkResult& benchmark_result) {
    const std::filesystem::path output_path(config.output_path);
    const std::filesystem::path parent_path = output_path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open output JSON file: " + config.output_path);
    }

    output << std::fixed << std::setprecision(6);
    const std::string model_name = std::filesystem::path(config.model_path).filename().string();
    output
        << "{\n"
        << "  \"schema_version\": \"inferedge-runtime-result-v1\",\n"
        << "  \"model_name\": " << json_string(model_name) << ",\n"
        << "  \"model_path\": " << json_string(config.model_path) << ",\n"
        << "  \"engine_name\": " << json_string(engine_metadata.name) << ",\n"
        << "  \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << "  \"device_name\": " << json_string(config.device) << ",\n"
        << "  \"batch\": " << config.batch << ",\n"
        << "  \"height\": " << config.height << ",\n"
        << "  \"width\": " << config.width << ",\n"
        << "  \"warmup\": " << config.warmup << ",\n"
        << "  \"runs\": " << config.runs << ",\n"
        << "  \"mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << "  \"p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << "  \"fps_value\": " << benchmark_result.fps << ",\n"
        << "  \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << "  \"status\": " << json_string(benchmark_result.status) << ",\n"
        << "  \"model\": {\n"
        << "    \"path\": " << json_string(config.model_path) << ",\n"
        << "    \"name\": " << json_string(model_name) << "\n"
        << "  },\n"
        << "  \"engine\": {\n"
        << "    \"name\": " << json_string(engine_metadata.name) << ",\n"
        << "    \"backend\": " << json_string(engine_metadata.backend) << ",\n"
        << "    \"available\": " << (engine_metadata.available ? "true" : "false") << ",\n"
        << "    \"status_message\": " << json_string(engine_metadata.status_message) << "\n"
        << "  },\n"
        << "  \"device\": {\n"
        << "    \"name\": " << json_string(config.device) << "\n"
        << "  },\n"
        << "  \"precision\": \"fp32\",\n"
        << "  \"run_config\": {\n"
        << "    \"batch\": " << config.batch << ",\n"
        << "    \"height\": " << config.height << ",\n"
        << "    \"width\": " << config.width << ",\n"
        << "    \"warmup\": " << config.warmup << ",\n"
        << "    \"runs\": " << config.runs << "\n"
        << "  },\n"
        << "  \"latency_ms\": {\n"
        << "    \"mean\": " << benchmark_result.mean_ms << ",\n"
        << "    \"min\": " << benchmark_result.min_ms << ",\n"
        << "    \"max\": " << benchmark_result.max_ms << ",\n"
        << "    \"std\": " << benchmark_result.std_ms << ",\n"
        << "    \"p50\": " << benchmark_result.p50_ms << ",\n"
        << "    \"p90\": " << benchmark_result.p90_ms << ",\n"
        << "    \"p99\": " << benchmark_result.p99_ms << ",\n"
        << "    \"samples\": ";
    write_double_vector_json(output, benchmark_result.samples_ms);
    output
        << "\n"
        << "  },\n"
        << "  \"fps\": " << benchmark_result.fps << ",\n"
        << "  \"benchmark\": {\n"
        << "    \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << "    \"status\": " << json_string(benchmark_result.status) << ",\n"
        << "    \"message\": " << json_string(benchmark_result.message) << "\n"
        << "  },\n"
        << "  \"timestamp\": " << json_string(current_utc_timestamp()) << ",\n"
        << "  \"system\": {\n"
        << "    \"os\": " << json_string(system_os_name()) << ",\n"
        << "    \"compiler\": " << json_string(compiler_name()) << ",\n"
        << "    \"cpp_standard\": \"17\"\n"
        << "  },\n"
        << "  \"model_metadata\": {\n"
        << "    \"inputs\": ";
    write_tensor_metadata_json(output, model_metadata.inputs, 4);
    output << ",\n";
    output << "    \"outputs\": ";
    write_tensor_metadata_json(output, model_metadata.outputs, 4);
    output
        << "\n"
        << "  },\n"
        << "  \"extra\": {\n"
        << "    \"runtime\": \"inferedge-runtime\",\n"
        << "    \"json_export\": \"enabled\"\n"
        << "  }\n"
        << "}\n";
}

}  // namespace

void print_help() {
    std::cout
        << "InferEdgeRuntime CLI\n"
        << "\n"
        << "Usage:\n"
        << "  inferedge-runtime [options]\n"
        << "\n"
        << "Options:\n"
        << "  -h, --help             Show this help message\n"
        << "  --version              Show version information\n"
        << "  --model <path>         Path to an input model file\n"
        << "  --engine <name>        Runtime engine name (supported: onnxruntime, ort; default: onnxruntime)\n"
        << "  --device <name>        Target device name (supported: cpu; default: cpu)\n"
        << "  --batch <n>            Dummy input batch size, n >= 1 (default: 1)\n"
        << "  --height <n>           Dummy input height, n >= 1 (default: 224)\n"
        << "  --width <n>            Dummy input width, n >= 1 (default: 224)\n"
        << "  --warmup <n>           Number of warmup runs, n >= 0 (default: 5)\n"
        << "  --runs <n>             Number of benchmark runs, n >= 1 (default: 50)\n"
        << "  --output <path>        Output result path (default: results/runtime_result.json)\n";
}

void print_version() {
    std::cout << "inferedge-runtime " << INFEREDGE_RUNTIME_VERSION << '\n';
}

RuntimeConfig parse_args(int argc, char** argv) {
    RuntimeConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "-h" || option == "--help") {
            config.show_help = true;
        } else if (option == "--version") {
            config.show_version = true;
        } else if (option == "--model") {
            config.model_path = require_value(argc, argv, i, option);
        } else if (option == "--engine") {
            config.engine = require_value(argc, argv, i, option);
            validate_engine(config.engine);
        } else if (option == "--device") {
            config.device = require_value(argc, argv, i, option);
            validate_device(config.device);
        } else if (option == "--batch") {
            config.batch = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--height") {
            config.height = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--width") {
            config.width = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--warmup") {
            config.warmup = parse_int_with_minimum(require_value(argc, argv, i, option), option, 0);
        } else if (option == "--runs") {
            config.runs = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--output") {
            config.output_path = require_value(argc, argv, i, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    validate_engine(config.engine);
    validate_device(config.device);

    return config;
}

int run_cli(const RuntimeConfig& config) {
    if (config.model_path.empty()) {
        std::cerr << "model path is required for benchmark execution\n";
        return 1;
    }

    const std::unique_ptr<IInferenceEngine> engine = create_engine(config);
    engine->load_model(config.model_path);
    const EngineMetadata metadata = engine->metadata();
    const ModelMetadata model_metadata = engine->model_metadata();

    std::cout
        << "InferEdgeRuntime benchmark configuration\n"
        << "  model:  " << config.model_path << '\n'
        << "  engine: " << config.engine << '\n'
        << "  device: " << config.device << '\n'
        << "  batch:  " << config.batch << '\n'
        << "  height: " << config.height << '\n'
        << "  width:  " << config.width << '\n'
        << "  warmup: " << config.warmup << '\n'
        << "  runs:   " << config.runs << '\n'
        << "  output: " << config.output_path << '\n'
        << "\n"
        << "Engine metadata\n"
        << "  name:      " << metadata.name << '\n'
        << "  backend:   " << metadata.backend << '\n'
        << "  device:    " << metadata.device << '\n'
        << "  available: " << (metadata.available ? "true" : "false") << '\n'
        << "  status:    " << metadata.status_message << '\n'
        << "\n"
        << "Model metadata\n"
        << "  inputs:";
    print_tensor_metadata_list(model_metadata.inputs);
    std::cout << "  outputs:";
    print_tensor_metadata_list(model_metadata.outputs);

    std::cout << "\nBenchmark\n";
    BenchmarkResult result;
    if (metadata.available) {
        result = engine->benchmark(config.warmup, config.runs);
        std::cout
            << "  status: success\n"
            << "  warmup: " << result.warmup_runs << '\n'
            << "  runs: " << result.timed_runs << '\n'
            << std::fixed << std::setprecision(3)
            << "  latency_ms:\n"
            << "    mean: " << result.mean_ms << '\n'
            << "    min: " << result.min_ms << '\n'
            << "    max: " << result.max_ms << '\n'
            << "    std: " << result.std_ms << '\n'
            << "    p50: " << result.p50_ms << '\n'
            << "    p90: " << result.p90_ms << '\n'
            << "    p99: " << result.p99_ms << '\n'
            << "  fps: " << result.fps << '\n';
    } else {
        result.success = false;
        result.status = "skipped";
        result.message = "backend is not available in this build";
        result.warmup_runs = config.warmup;
        result.timed_runs = config.runs;
        std::cout
            << "  status: " << result.status << '\n'
            << "  reason: " << result.message << '\n';
    }

    write_result_json(config, metadata, model_metadata, result);

    std::cout
        << "\nResult JSON\n"
        << "  path: " << config.output_path << '\n'
        << "  status: written\n";

    return 0;
}

}  // namespace inferedge_runtime
