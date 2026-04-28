#include "inferedge_runtime/result_writer.hpp"

#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferedge_runtime {
namespace {

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

std::string timestamp_for_filename(const std::string& timestamp) {
    std::string value;
    value.reserve(timestamp.size());
    for (const char ch : timestamp) {
        if (ch != '-' && ch != ':') {
            value.push_back(ch);
        }
    }
    return value;
}

std::string sanitize_filename_component(const std::string& value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized.empty() ? "unknown" : sanitized;
}

std::filesystem::path resolve_output_path(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const std::string& timestamp) {
    if (config.output_path != "auto") {
        return std::filesystem::path(config.output_path);
    }

    const std::string model_stem = sanitize_filename_component(
        std::filesystem::path(config.model_path).stem().string());
    const std::string engine_name = sanitize_filename_component(engine_metadata.name);
    const std::string device_name = sanitize_filename_component(engine_metadata.device);
    const std::string filename =
        model_stem + "__" + engine_name + "__" + device_name + "__fp32__b" +
        std::to_string(config.batch) + "__h" + std::to_string(config.height) + "w" +
        std::to_string(config.width) + "__" + timestamp_for_filename(timestamp) + ".json";

    return std::filesystem::path("results") / filename;
}

std::string stem_from_path_like_value(const std::string& value) {
    return std::filesystem::path(value).stem().string();
}

std::string compare_model_name(const RuntimeConfig& config) {
    if (!config.manifest_model_name.empty()) {
        return sanitize_filename_component(stem_from_path_like_value(config.manifest_model_name));
    }

    return sanitize_filename_component(std::filesystem::path(config.model_path).stem().string());
}

std::string compare_model_source(const RuntimeConfig& config) {
    return config.manifest_model_name.empty() ? "model_path" : "manifest_source_model";
}

std::string make_compare_key(const RuntimeConfig& config) {
    const std::string model_stem = compare_model_name(config);
    const std::string precision = config.manifest_precision.empty() ? "fp32" : config.manifest_precision;
    return model_stem + "__b" + std::to_string(config.batch) + "__h" +
           std::to_string(config.height) + "w" + std::to_string(config.width) + "__" +
           sanitize_filename_component(precision);
}

std::string make_backend_key(const EngineMetadata& engine_metadata, const RuntimeConfig& config) {
    return sanitize_filename_component(engine_metadata.backend) + "__" +
           sanitize_filename_component(config.device);
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

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    const std::filesystem::path parent_path = path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output JSON file: " + path.string());
    }

    output << content;
}

}  // namespace

std::filesystem::path write_result_json(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const ModelMetadata& model_metadata,
    const BenchmarkResult& benchmark_result) {
    const std::string timestamp = current_utc_timestamp();
    const std::filesystem::path output_path = resolve_output_path(config, engine_metadata, timestamp);
    const std::filesystem::path latest_path("results/latest.json");
    const std::string output_mode = config.output_path == "auto" ? "auto" : "explicit";
    const std::string compare_key = make_compare_key(config);
    const std::string backend_key = make_backend_key(engine_metadata, config);
    const std::string compare_name = compare_model_name(config);
    const std::string compare_source = compare_model_source(config);
    const std::string precision = config.manifest_precision.empty() ? "fp32" : config.manifest_precision;

    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    const std::string model_name = std::filesystem::path(config.model_path).filename().string();
    output
        << "{\n"
        << "  \"schema_version\": \"inferedge-runtime-result-v1\",\n"
        << "  \"compare_key\": " << json_string(compare_key) << ",\n"
        << "  \"backend_key\": " << json_string(backend_key) << ",\n"
        << "  \"runtime_role\": \"runtime-result\",\n"
        << "  \"manifest_path\": " << json_string(config.manifest_path) << ",\n"
        << "  \"manifest_applied\": " << (config.manifest_applied ? "true" : "false") << ",\n"
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
        << "  \"p50_ms\": " << benchmark_result.p50_ms << ",\n"
        << "  \"p95_ms\": " << benchmark_result.p95_ms << ",\n"
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
        << "  \"precision\": " << json_string(precision) << ",\n"
        << "  \"run_config\": {\n"
        << "    \"batch\": " << config.batch << ",\n"
        << "    \"height\": " << config.height << ",\n"
        << "    \"width\": " << config.width << ",\n"
        << "    \"warmup\": " << config.warmup << ",\n"
        << "    \"runs\": " << config.runs << ",\n"
        << "    \"manifest_path\": " << json_string(config.manifest_path) << ",\n"
        << "    \"manifest_applied\": " << (config.manifest_applied ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"latency_ms\": {\n"
        << "    \"mean\": " << benchmark_result.mean_ms << ",\n"
        << "    \"min\": " << benchmark_result.min_ms << ",\n"
        << "    \"max\": " << benchmark_result.max_ms << ",\n"
        << "    \"std\": " << benchmark_result.std_ms << ",\n"
        << "    \"p50\": " << benchmark_result.p50_ms << ",\n"
        << "    \"p90\": " << benchmark_result.p90_ms << ",\n"
        << "    \"p95\": " << benchmark_result.p95_ms << ",\n"
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
        << "  \"timestamp\": " << json_string(timestamp) << ",\n"
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
        << "    \"json_export\": \"enabled\",\n"
        << "    \"output_mode\": " << json_string(output_mode) << ",\n"
        << "    \"latest_path\": \"results/latest.json\",\n"
        << "    \"manifest_recorded\": " << (config.manifest_path.empty() ? "false" : "true") << ",\n"
        << "    \"manifest_precision\": " << json_string(config.manifest_precision) << ",\n"
        << "    \"manifest_format\": " << json_string(config.manifest_format) << ",\n"
        << "    \"manifest_preset_name\": " << json_string(config.manifest_preset_name) << ",\n"
        << "    \"manifest_build_id\": " << json_string(config.manifest_build_id) << ",\n"
        << "    \"source_model_path\": " << json_string(config.manifest_source_model_path) << ",\n"
        << "    \"source_model_sha256\": " << json_string(config.manifest_source_sha256) << ",\n"
        << "    \"runtime_artifact_sha256\": " << json_string(config.manifest_artifact_sha256) << ",\n"
        << "    \"runtime_artifact_path\": " << json_string(config.model_path) << ",\n"
        << "    \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << "    \"input_path\": " << json_string(config.input_path) << ",\n"
        << "    \"input_preprocess\": " << json_string(config.input_preprocess()) << ",\n"
        << "    \"compare_ready\": true,\n"
        << "    \"compare_key\": " << json_string(compare_key) << ",\n"
        << "    \"backend_key\": " << json_string(backend_key) << ",\n"
        << "    \"compare_model_source\": " << json_string(compare_source) << ",\n"
        << "    \"compare_model_name\": " << json_string(compare_name) << "\n"
        << "  }\n"
        << "}\n";

    const std::string json = output.str();
    write_text_file(output_path, json);
    if (output_path.lexically_normal() != latest_path.lexically_normal()) {
        write_text_file(latest_path, json);
    }

    return output_path;
}

}  // namespace inferedge_runtime
