#pragma once

#include "inferedge_runtime/engine.hpp"

#include <string>

namespace inferedge_runtime {

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
    explicit OnnxRuntimeEngine(RuntimeConfig config);

    EngineMetadata metadata() const override;
    void load_model(const std::string& model_path) override;
    void run_once() override;

private:
    RuntimeConfig config_;
    std::string loaded_model_path_;
};

}  // namespace inferedge_runtime
