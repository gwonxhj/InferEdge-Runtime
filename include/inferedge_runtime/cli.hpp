#pragma once

#include <string>

namespace inferedge_runtime {

struct RuntimeConfig {
    std::string model_path;
    std::string engine = "onnxruntime";
    std::string device = "cpu";
    int warmup = 5;
    int runs = 50;
    std::string output_path = "results/runtime_result.json";
    bool show_help = false;
    bool show_version = false;
};

void print_help();
void print_version();
RuntimeConfig parse_args(int argc, char** argv);
int run_cli(const RuntimeConfig& config);

}  // namespace inferedge_runtime
