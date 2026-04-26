#pragma once

#include "inferedge_runtime/cli.hpp"

#include <memory>
#include <string>

namespace inferedge_runtime {

struct EngineMetadata {
    std::string name;
    std::string backend;
    std::string device;
    bool available = false;
    std::string status_message;
};

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    virtual EngineMetadata metadata() const = 0;
    virtual void load_model(const std::string& model_path) = 0;
    virtual void run_once() = 0;
};

std::unique_ptr<IInferenceEngine> create_engine(const RuntimeConfig& config);

}  // namespace inferedge_runtime
