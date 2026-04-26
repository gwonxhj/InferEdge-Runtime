#include "inferedge_runtime/cli.hpp"

#include "inferedge_runtime/engine.hpp"
#include "inferedge_runtime/version.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

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

    std::cout
        << "InferEdgeRuntime benchmark configuration\n"
        << "  model:  " << config.model_path << '\n'
        << "  engine: " << config.engine << '\n'
        << "  device: " << config.device << '\n'
        << "  warmup: " << config.warmup << '\n'
        << "  runs:   " << config.runs << '\n'
        << "  output: " << config.output_path << '\n';

    const std::unique_ptr<IInferenceEngine> engine = create_engine(config);
    engine->load_model(config.model_path);
    const EngineMetadata metadata = engine->metadata();

    std::cout
        << "\n"
        << "Engine metadata\n"
        << "  name:      " << metadata.name << '\n'
        << "  backend:   " << metadata.backend << '\n'
        << "  device:    " << metadata.device << '\n'
        << "  available: " << (metadata.available ? "true" : "false") << '\n'
        << "  status:    " << metadata.status_message << '\n'
        << "\n"
        << "Inference execution is not implemented yet.\n";

    return 0;
}

}  // namespace inferedge_runtime
