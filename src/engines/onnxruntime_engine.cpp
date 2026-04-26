#include "inferedge_runtime/engines/onnxruntime_engine.hpp"

#ifdef INFEREDGE_ORT_LINKED
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferedge_runtime {
namespace {

#ifdef INFEREDGE_ORT_LINKED
double percentile_nearest_rank(std::vector<double> sorted_samples, double percentile) {
    if (sorted_samples.empty()) {
        return 0.0;
    }

    std::sort(sorted_samples.begin(), sorted_samples.end());
    const double rank = percentile / 100.0 * static_cast<double>(sorted_samples.size());
    std::size_t index = static_cast<std::size_t>(std::ceil(rank));
    if (index == 0) {
        index = 1;
    }
    return sorted_samples[index - 1];
}

std::string element_type_to_string(ONNXTensorElementDataType element_type) {
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}

std::string copy_allocated_name(Ort::AllocatedStringPtr name) {
    return name ? std::string{name.get()} : std::string{};
}

std::vector<int64_t> resolve_input_shape(const std::vector<int64_t>& shape, const RuntimeConfig& config) {
    std::vector<int64_t> resolved_shape;
    resolved_shape.reserve(shape.size());

    for (std::size_t i = 0; i < shape.size(); ++i) {
        const int64_t dim = shape[i];
        if (dim > 0) {
            resolved_shape.push_back(dim);
            continue;
        }

        if (i == 0) {
            resolved_shape.push_back(config.batch);
        } else if (i == 1) {
            resolved_shape.push_back(3);
        } else if (i == 2) {
            resolved_shape.push_back(config.height);
        } else if (i == 3) {
            resolved_shape.push_back(config.width);
        } else {
            resolved_shape.push_back(1);
        }
    }

    return resolved_shape;
}

std::size_t tensor_element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("resolved input shape contains non-positive dimension");
        }

        const std::size_t size_dim = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / size_dim) {
            throw std::runtime_error("resolved input shape is too large");
        }
        count *= size_dim;
    }
    return count;
}
#endif

}  // namespace

struct OnnxRuntimeEngineImpl {
    RuntimeConfig config;
    std::string loaded_model_path;
    ModelMetadata model_metadata;

#ifdef INFEREDGE_ORT_LINKED
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "inferedge-runtime"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    std::vector<std::vector<int64_t>> resolved_input_shapes;
    std::vector<std::vector<float>> input_buffers;
#endif
};

OnnxRuntimeEngine::OnnxRuntimeEngine(RuntimeConfig config) : impl_(std::make_unique<OnnxRuntimeEngineImpl>()) {
    impl_->config = std::move(config);
}

OnnxRuntimeEngine::~OnnxRuntimeEngine() = default;

EngineMetadata OnnxRuntimeEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "onnxruntime";
    metadata.backend = "onnxruntime";
    metadata.device = impl_->config.device;
#ifdef INFEREDGE_ORT_LINKED
    metadata.available = true;
    metadata.status_message = "ONNX Runtime backend is linked. Benchmark execution is available.";
#else
    metadata.available = false;
    metadata.status_message =
        "ONNX Runtime backend is disabled. Reconfigure with -DINFEREDGE_ENABLE_ORT=ON after installing ONNX Runtime C++.";
#endif
    return metadata;
}

ModelMetadata OnnxRuntimeEngine::model_metadata() const {
    return impl_->model_metadata;
}

void OnnxRuntimeEngine::load_model(const std::string& model_path) {
    impl_->loaded_model_path = model_path;
    impl_->model_metadata = {};

#ifdef INFEREDGE_ORT_LINKED
    impl_->input_names.clear();
    impl_->output_names.clear();
    impl_->input_name_ptrs.clear();
    impl_->output_name_ptrs.clear();
    impl_->resolved_input_shapes.clear();
    impl_->input_buffers.clear();

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("model file not found: " + model_path);
    }

    impl_->session_options = Ort::SessionOptions{};
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t input_count = impl_->session->GetInputCount();
    impl_->model_metadata.inputs.reserve(input_count);
    impl_->input_names.reserve(input_count);
    impl_->input_name_ptrs.reserve(input_count);
    impl_->resolved_input_shapes.reserve(input_count);

    for (std::size_t i = 0; i < input_count; ++i) {
        TensorMetadata metadata;
        metadata.name = copy_allocated_name(impl_->session->GetInputNameAllocated(i, allocator));

        const Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(i);
        const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
        metadata.element_type = element_type_to_string(tensor_info.GetElementType());
        metadata.shape = tensor_info.GetShape();
        for (int64_t& dim : metadata.shape) {
            if (dim < 0) {
                dim = -1;
            }
        }

        impl_->resolved_input_shapes.push_back(resolve_input_shape(metadata.shape, impl_->config));
        impl_->input_names.push_back(metadata.name);
        impl_->model_metadata.inputs.push_back(std::move(metadata));
    }

    for (const std::string& name : impl_->input_names) {
        impl_->input_name_ptrs.push_back(name.c_str());
    }

    const std::size_t output_count = impl_->session->GetOutputCount();
    impl_->model_metadata.outputs.reserve(output_count);
    impl_->output_names.reserve(output_count);
    impl_->output_name_ptrs.reserve(output_count);

    for (std::size_t i = 0; i < output_count; ++i) {
        TensorMetadata metadata;
        metadata.name = copy_allocated_name(impl_->session->GetOutputNameAllocated(i, allocator));

        const Ort::TypeInfo type_info = impl_->session->GetOutputTypeInfo(i);
        const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
        metadata.element_type = element_type_to_string(tensor_info.GetElementType());
        metadata.shape = tensor_info.GetShape();
        for (int64_t& dim : metadata.shape) {
            if (dim < 0) {
                dim = -1;
            }
        }

        impl_->output_names.push_back(metadata.name);
        impl_->model_metadata.outputs.push_back(std::move(metadata));
    }

    for (const std::string& name : impl_->output_names) {
        impl_->output_name_ptrs.push_back(name.c_str());
    }
#endif
}

void OnnxRuntimeEngine::run_once() {
#ifdef INFEREDGE_ORT_LINKED
    if (!impl_->session) {
        throw std::runtime_error("model is not loaded");
    }

    if (impl_->model_metadata.inputs.empty()) {
        throw std::runtime_error("model has no inputs");
    }

    impl_->input_buffers.clear();
    impl_->input_buffers.reserve(impl_->model_metadata.inputs.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(impl_->model_metadata.inputs.size());
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (std::size_t i = 0; i < impl_->model_metadata.inputs.size(); ++i) {
        const TensorMetadata& input = impl_->model_metadata.inputs[i];
        if (input.element_type != "float32") {
            throw std::runtime_error("dummy inference currently supports float32 inputs only");
        }

        const std::vector<int64_t>& shape = impl_->resolved_input_shapes[i];
        const std::size_t element_count = tensor_element_count(shape);
        impl_->input_buffers.emplace_back(element_count, 0.0F);
        std::vector<float>& buffer = impl_->input_buffers.back();

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            buffer.data(),
            buffer.size(),
            shape.data(),
            shape.size()));
    }

    (void)impl_->session->Run(
        Ort::RunOptions{nullptr},
        impl_->input_name_ptrs.data(),
        input_tensors.data(),
        input_tensors.size(),
        impl_->output_name_ptrs.data(),
        impl_->output_name_ptrs.size());
#else
    throw std::runtime_error("ONNX Runtime backend is not available in this build");
#endif
}

BenchmarkResult OnnxRuntimeEngine::benchmark(int warmup, int runs) {
#ifdef INFEREDGE_ORT_LINKED
    if (!impl_->session) {
        throw std::runtime_error("model is not loaded");
    }

    if (warmup < 0) {
        throw std::runtime_error("warmup must be greater than or equal to 0");
    }

    if (runs < 1) {
        throw std::runtime_error("runs must be greater than or equal to 1");
    }

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    BenchmarkResult result;
    result.warmup_runs = warmup;
    result.timed_runs = runs;
    result.samples_ms.reserve(static_cast<std::size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        run_once();
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        result.samples_ms.push_back(elapsed.count());
    }

    const auto minmax = std::minmax_element(result.samples_ms.begin(), result.samples_ms.end());
    result.min_ms = *minmax.first;
    result.max_ms = *minmax.second;
    result.mean_ms =
        std::accumulate(result.samples_ms.begin(), result.samples_ms.end(), 0.0) /
        static_cast<double>(result.samples_ms.size());

    double variance = 0.0;
    for (const double sample : result.samples_ms) {
        const double diff = sample - result.mean_ms;
        variance += diff * diff;
    }
    variance /= static_cast<double>(result.samples_ms.size());
    result.std_ms = std::sqrt(variance);

    std::vector<double> sorted_samples = result.samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    result.p50_ms = percentile_nearest_rank(sorted_samples, 50.0);
    result.p90_ms = percentile_nearest_rank(sorted_samples, 90.0);
    result.p99_ms = percentile_nearest_rank(sorted_samples, 99.0);
    result.fps = result.mean_ms > 0.0 ? 1000.0 / result.mean_ms : 0.0;

    return result;
#else
    (void)warmup;
    (void)runs;
    throw std::runtime_error("ONNX Runtime backend is not available in this build");
#endif
}

}  // namespace inferedge_runtime
