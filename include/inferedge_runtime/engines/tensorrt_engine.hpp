#pragma once

#include "inferedge_runtime/engine.hpp"

#include <string>

namespace inferedge_runtime {

class TensorRTEngine final : public IInferenceEngine {
public:
    explicit TensorRTEngine(RuntimeConfig config);
    ~TensorRTEngine() override;

    EngineMetadata metadata() const override;
    ModelMetadata model_metadata() const override;
    void load_model(const std::string& model_path) override;
    void run_once() override;
    BenchmarkResult benchmark(int warmup, int runs) override;

private:
    RuntimeConfig config_;
    std::string loaded_model_path_;
};

}  // namespace inferedge_runtime
