#pragma once

#include "inferedge_runtime/engine.hpp"

#include <memory>
#include <string>

namespace inferedge_runtime {

struct OnnxRuntimeEngineImpl;

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
    explicit OnnxRuntimeEngine(RuntimeConfig config);
    ~OnnxRuntimeEngine() override;

    EngineMetadata metadata() const override;
    ModelMetadata model_metadata() const override;
    void load_model(const std::string& model_path) override;
    void run_once() override;

private:
    std::unique_ptr<OnnxRuntimeEngineImpl> impl_;
};

}  // namespace inferedge_runtime
