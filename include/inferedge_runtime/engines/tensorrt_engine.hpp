#pragma once

#include "inferedge_runtime/engine.hpp"

#include <memory>
#include <string>

namespace inferedge_runtime {

struct TensorRTEngineImpl;

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
    std::unique_ptr<TensorRTEngineImpl> impl_;
};

}  // namespace inferedge_runtime
