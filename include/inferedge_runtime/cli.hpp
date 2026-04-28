#pragma once

#include <string>

namespace inferedge_runtime {

struct RuntimeConfig {
    std::string manifest_path;
    std::string forge_metadata_path;
    std::string forge_manifest_path;
    std::string lab_worker_request_path;
    std::string manifest_model_name;
    std::string manifest_precision;
    std::string manifest_format;
    std::string manifest_artifact_sha256;
    std::string manifest_source_sha256;
    std::string manifest_preset_name;
    std::string manifest_build_id;
    std::string model_path;
    std::string input_path;
    std::string engine = "onnxruntime";
    std::string device = "cpu";
    int batch = 1;
    int height = 224;
    int width = 224;
    int warmup = 5;
    int runs = 50;
    std::string output_path = "results/runtime_result.json";
    bool run_once = false;
    bool manifest_applied = false;
    bool model_path_overridden = false;
    bool engine_overridden = false;
    bool device_overridden = false;
    bool batch_overridden = false;
    bool height_overridden = false;
    bool width_overridden = false;
    bool show_help = false;
    bool show_version = false;
    bool validate_forge_handoff = false;
    bool validate_lab_worker_request = false;

    bool has_real_input() const {
        return !input_path.empty();
    }

    std::string input_mode() const {
        return has_real_input() ? "image" : "dummy";
    }

    std::string input_preprocess() const {
        return has_real_input() ? "opencv_bgr_to_rgb_resize_float32_nchw" : "dummy_zero_float32";
    }
};

void print_help();
void print_version();
RuntimeConfig parse_args(int argc, char** argv);
int run_cli(const RuntimeConfig& config);

}  // namespace inferedge_runtime
