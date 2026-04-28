#pragma once

#include <filesystem>
#include <string>

namespace inferedge_runtime {

struct LabWorkerRequestConfig {
    std::string job_id;
    std::string requested_at;
    std::string workflow;
    std::string model_path;
    std::string artifact_path;
    std::string metadata_path;
    std::string manifest_path;
    std::string engine;
    std::string device;
    std::string precision;
    int batch = 0;
    int height = 0;
    int width = 0;
    int warmup = 0;
    int runs = 0;
};

LabWorkerRequestConfig load_lab_worker_request_config(const std::string& path);
void validate_lab_worker_request_config(const LabWorkerRequestConfig& request, const std::string& path);
std::filesystem::path write_worker_response_dry_run(
    const LabWorkerRequestConfig& request,
    const std::string& status,
    const std::string& output_path,
    const std::string& error_message);

}  // namespace inferedge_runtime
