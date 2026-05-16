#pragma once

#include "inferedge_runtime/cli.hpp"

#include <string>

namespace inferedge_runtime {

struct AgentManifestConfig {
    std::string schema_version;
    std::string agent_id;
    std::string agent_type;
    std::string input_type;
    std::string output_type;
    std::string required_backend;
    std::string device_target;
    std::string precision;
    std::string runtime_artifact_path;
    std::string fallback_policy_mode;
    std::string telemetry_contract_version;
    int priority = 0;
    int latency_budget_ms = 0;
    int deadline_ms = 0;
};

AgentManifestConfig load_agent_manifest_config(const std::string& path);
void validate_agent_manifest_config(const AgentManifestConfig& manifest, const std::string& path);
void apply_agent_manifest_defaults(RuntimeConfig& config, const AgentManifestConfig& manifest);

}  // namespace inferedge_runtime
