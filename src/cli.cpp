#include "inferedge_runtime/cli.hpp"

#include "inferedge_runtime/engine.hpp"
#include "inferedge_runtime/manifest.hpp"
#include "inferedge_runtime/result_writer.hpp"
#include "inferedge_runtime/version.hpp"

#include <filesystem>
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
    return engine == "onnxruntime" || engine == "ort" || engine == "tensorrt" || engine == "trt";
}

bool is_supported_device(const std::string& device) {
    return device == "cpu" || device == "jetson" || device == "cuda";
}

void validate_engine(const std::string& engine) {
    if (!is_supported_engine(engine)) {
        throw std::invalid_argument(
            "unsupported engine: " + engine + " (supported: onnxruntime, ort, tensorrt, trt)");
    }
}

void validate_device(const std::string& device) {
    if (!is_supported_device(device)) {
        throw std::invalid_argument("unsupported device: " + device + " (supported: cpu, jetson, cuda)");
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
        << "  --manifest <path>      Optional Forge/build manifest path used for default runtime config\n"
        << "  --model <path>         Path to an input model file\n"
        << "  --input <image_path>   Optional real image input path (requires OpenCV-enabled build)\n"
        << "  --engine <name>        Runtime engine name (supported: onnxruntime, ort, tensorrt, trt; default: onnxruntime)\n"
        << "  --device <name>        Target device name (supported: cpu, jetson, cuda; default: cpu)\n"
        << "  --batch <n>            Dummy input batch size, n >= 1 (default: 1)\n"
        << "  --height <n>           Dummy input height, n >= 1 (default: 224)\n"
        << "  --width <n>            Dummy input width, n >= 1 (default: 224)\n"
        << "  --warmup <n>           Number of warmup runs, n >= 0 (default: 5)\n"
        << "  --runs <n>             Number of benchmark runs, n >= 1 (default: 50)\n"
        << "  --run-once             Run one inference without benchmark timing\n"
        << "  --output <path|auto>   Output result path or auto-generated results filename (default: results/runtime_result.json)\n";
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
        } else if (option == "--manifest") {
            config.manifest_path = require_value(argc, argv, i, option);
        } else if (option == "--model") {
            config.model_path = require_value(argc, argv, i, option);
            config.model_path_overridden = true;
        } else if (option == "--input") {
            config.input_path = require_value(argc, argv, i, option);
        } else if (option == "--engine") {
            config.engine = require_value(argc, argv, i, option);
            config.engine_overridden = true;
            validate_engine(config.engine);
        } else if (option == "--device") {
            config.device = require_value(argc, argv, i, option);
            config.device_overridden = true;
            validate_device(config.device);
        } else if (option == "--batch") {
            config.batch = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.batch_overridden = true;
        } else if (option == "--height") {
            config.height = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.height_overridden = true;
        } else if (option == "--width") {
            config.width = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.width_overridden = true;
        } else if (option == "--warmup") {
            config.warmup = parse_int_with_minimum(require_value(argc, argv, i, option), option, 0);
        } else if (option == "--runs") {
            config.runs = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--run-once") {
            config.run_once = true;
        } else if (option == "--output") {
            config.output_path = require_value(argc, argv, i, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    if (!config.manifest_path.empty()) {
        const ManifestConfig manifest = load_manifest_config(config.manifest_path);
        apply_manifest_defaults(config, manifest);
        config.manifest_precision = manifest.precision;
        config.manifest_format = manifest.format;
        config.manifest_applied = true;
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
        << "  manifest: " << (config.manifest_path.empty() ? "none" : config.manifest_path) << '\n'
        << "  manifest_applied: " << (config.manifest_applied ? "true" : "false") << '\n'
        << "  model:  " << config.model_path << '\n'
        << "  input:  " << (config.input_path.empty() ? "dummy" : config.input_path) << '\n'
        << "  engine: " << config.engine << '\n'
        << "  device: " << config.device << '\n'
        << "  batch:  " << config.batch << '\n'
        << "  height: " << config.height << '\n'
        << "  width:  " << config.width << '\n'
        << "  warmup: " << config.warmup << '\n'
        << "  runs:   " << config.runs << '\n'
        << "  run_once: " << (config.run_once ? "true" : "false") << '\n'
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

    BenchmarkResult result;
    if (config.run_once) {
        std::cout << "\nInference\n";
        if (metadata.available) {
            try {
                engine->run_once();
                result.success = true;
                result.status = "success";
                result.message = "one-shot inference completed";
                result.warmup_runs = 0;
                result.timed_runs = 1;
                std::cout
                    << "  status: " << result.status << '\n'
                    << "  runs: 1\n";
            } catch (const std::runtime_error& error) {
                result.success = false;
                result.status = "skipped";
                result.message = error.what();
                result.warmup_runs = 0;
                result.timed_runs = 1;
                std::cout
                    << "  status: " << result.status << '\n'
                    << "  reason: " << result.message << '\n';
            }
        } else {
            result.success = false;
            result.status = "skipped";
            result.message = "backend is not available in this build";
            result.warmup_runs = 0;
            result.timed_runs = 1;
            std::cout
                << "  status: " << result.status << '\n'
                << "  reason: " << result.message << '\n';
        }
    } else if (metadata.available) {
        std::cout << "\nBenchmark\n";
        try {
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
        } catch (const std::runtime_error& error) {
            result.success = false;
            result.status = "skipped";
            result.message = error.what();
            result.warmup_runs = config.warmup;
            result.timed_runs = config.runs;
            std::cout
                << "  status: " << result.status << '\n'
                << "  reason: " << result.message << '\n';
        }
    } else {
        std::cout << "\nBenchmark\n";
        result.success = false;
        result.status = "skipped";
        result.message = "backend is not available in this build";
        result.warmup_runs = config.warmup;
        result.timed_runs = config.runs;
        std::cout
            << "  status: " << result.status << '\n'
            << "  reason: " << result.message << '\n';
    }

    const std::filesystem::path output_path = write_result_json(config, metadata, model_metadata, result);

    std::cout
        << "\nResult JSON\n"
        << "  path: " << output_path.string() << '\n'
        << "  latest: results/latest.json\n"
        << "  status: written\n";

    return 0;
}

}  // namespace inferedge_runtime
