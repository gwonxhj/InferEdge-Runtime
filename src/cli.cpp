#include "inferedge_runtime/cli.hpp"

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
    if (value.empty() || value.rfind("-", 0) == 0) {
        throw std::invalid_argument("missing value for option: " + option);
    }

    return value;
}

int parse_positive_int(const std::string& value, const std::string& option) {
    try {
        std::size_t parsed_chars = 0;
        const int parsed_value = std::stoi(value, &parsed_chars);
        if (parsed_chars != value.size() || parsed_value < 0) {
            throw std::invalid_argument("invalid integer value for option: " + option);
        }
        return parsed_value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid integer value for option: " + option + " (" + value + ")");
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
        << "  --engine <name>        Runtime engine name (default: onnxruntime)\n"
        << "  --device <name>        Target device name (default: cpu)\n"
        << "  --warmup <n>           Number of warmup runs (default: 5)\n"
        << "  --runs <n>             Number of benchmark runs (default: 50)\n"
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
        } else if (option == "--device") {
            config.device = require_value(argc, argv, i, option);
        } else if (option == "--warmup") {
            config.warmup = parse_positive_int(require_value(argc, argv, i, option), option);
        } else if (option == "--runs") {
            config.runs = parse_positive_int(require_value(argc, argv, i, option), option);
        } else if (option == "--output") {
            config.output_path = require_value(argc, argv, i, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

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
        << "  output: " << config.output_path << '\n'
        << "\n"
        << "Inference execution is not implemented yet.\n";

    return 0;
}

}  // namespace inferedge_runtime
