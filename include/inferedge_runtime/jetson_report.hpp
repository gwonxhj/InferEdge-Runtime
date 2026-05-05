#pragma once

#include <filesystem>
#include <string>

namespace inferedge_runtime {

std::filesystem::path write_jetson_evidence_markdown_report(
    const std::string& result_json_path,
    const std::string& tegrastats_log_path,
    const std::string& output_path);

std::filesystem::path write_power_mode_comparison_markdown_report(
    const std::string& base_result_json_path,
    const std::string& candidate_result_json_path,
    const std::string& output_path);

}  // namespace inferedge_runtime
