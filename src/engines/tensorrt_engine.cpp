#include "inferedge_runtime/engines/tensorrt_engine.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef INFEREDGE_TENSORRT_LINKED
#include <NvInfer.h>
#include <NvInferRuntime.h>
#endif

namespace inferedge_runtime {
namespace {

#ifdef INFEREDGE_TENSORRT_LINKED
class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

template <typename T>
struct TensorRTDeleter {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter<nvinfer1::IRuntime>>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter<nvinfer1::ICudaEngine>>;

std::vector<char> read_binary_file(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("TensorRT engine file not found: " + path);
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine file is empty: " + path);
    }

    std::vector<char> buffer(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine file: " + path);
    }

    return buffer;
}

std::string tensor_rt_dtype_to_string(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kINT64:
            return "int64";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        default:
            return "unknown";
    }
}

std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        shape.push_back(static_cast<int64_t>(dims.d[index]));
    }
    return shape;
}

ModelMetadata extract_model_metadata(const nvinfer1::ICudaEngine& engine) {
    ModelMetadata metadata;
    const int tensor_count = engine.getNbIOTensors();
    if (tensor_count <= 0) {
        throw std::runtime_error("TensorRT engine metadata extraction failed: no I/O tensors");
    }

    for (int index = 0; index < tensor_count; ++index) {
        const char* tensor_name = engine.getIOTensorName(index);
        if (tensor_name == nullptr) {
            throw std::runtime_error("TensorRT engine metadata extraction failed: null tensor name");
        }

        TensorMetadata tensor;
        tensor.name = tensor_name;
        tensor.element_type = tensor_rt_dtype_to_string(engine.getTensorDataType(tensor_name));
        tensor.shape = dims_to_shape(engine.getTensorShape(tensor_name));

        const nvinfer1::TensorIOMode mode = engine.getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            metadata.inputs.push_back(std::move(tensor));
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            metadata.outputs.push_back(std::move(tensor));
        }
    }

    if (metadata.inputs.empty() || metadata.outputs.empty()) {
        throw std::runtime_error("TensorRT engine metadata extraction failed: no inputs or outputs");
    }

    return metadata;
}
#endif

}  // namespace

struct TensorRTEngineImpl {
    explicit TensorRTEngineImpl(RuntimeConfig runtime_config) : config(std::move(runtime_config)) {}

    RuntimeConfig config;
    std::string loaded_model_path;
    ModelMetadata model_metadata;

#ifdef INFEREDGE_TENSORRT_LINKED
    TensorRTLogger logger;
    RuntimePtr runtime;
    EnginePtr engine;
#endif
};

TensorRTEngine::TensorRTEngine(RuntimeConfig config)
    : impl_(std::make_unique<TensorRTEngineImpl>(std::move(config))) {}

TensorRTEngine::~TensorRTEngine() = default;

EngineMetadata TensorRTEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "tensorrt";
    metadata.backend = "tensorrt";
    metadata.device = impl_->config.device;
#ifdef INFEREDGE_TENSORRT_LINKED
    metadata.available = true;
    metadata.status_message = "TensorRT backend is linked. Engine metadata loading is available.";
#else
    metadata.available = false;
    metadata.status_message = "TensorRT backend is not implemented in this build. This stub prepares Jetson integration.";
#endif
    return metadata;
}

ModelMetadata TensorRTEngine::model_metadata() const {
    return impl_->model_metadata;
}

void TensorRTEngine::load_model(const std::string& model_path) {
    impl_->loaded_model_path = model_path;
    impl_->model_metadata = {};

#ifdef INFEREDGE_TENSORRT_LINKED
    const std::vector<char> engine_buffer = read_binary_file(model_path);

    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()));
    if (!impl_->engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine: " + model_path);
    }

    impl_->model_metadata = extract_model_metadata(*impl_->engine);
#endif
}

void TensorRTEngine::run_once() {
#ifdef INFEREDGE_TENSORRT_LINKED
    throw std::runtime_error("TensorRT backend is linked, but inference execution is not implemented yet");
#else
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

BenchmarkResult TensorRTEngine::benchmark(int warmup, int runs) {
    (void)warmup;
    (void)runs;
#ifdef INFEREDGE_TENSORRT_LINKED
    throw std::runtime_error("TensorRT backend is linked, but benchmark execution is not implemented yet");
#else
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

}  // namespace inferedge_runtime
