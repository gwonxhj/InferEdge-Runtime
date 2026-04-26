#include "inferedge_runtime/engines/tensorrt_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef INFEREDGE_TENSORRT_LINKED
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

#ifdef INFEREDGE_OPENCV_LINKED
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter<nvinfer1::IExecutionContext>>;

struct TensorRTBuffer {
    std::string name;
    bool is_input = false;
    std::string element_type;
    std::vector<int64_t> shape;
    std::size_t element_count = 0;
    std::size_t byte_size = 0;
    std::vector<float> host_float_buffer;
    void* device_buffer = nullptr;
};

struct CudaStreamGuard {
    CudaStreamGuard() {
        check(cudaStreamCreate(&stream), "Failed to create CUDA stream");
    }

    ~CudaStreamGuard() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    CudaStreamGuard(const CudaStreamGuard&) = delete;
    CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;

    static void check(cudaError_t status, const std::string& message) {
        if (status != cudaSuccess) {
            throw std::runtime_error(message + ": " + cudaGetErrorString(status));
        }
    }

    cudaStream_t stream = nullptr;
};

void check_cuda(cudaError_t status, const std::string& message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(status));
    }
}

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

std::size_t tensor_element_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("TensorRT tensor shape is empty");
    }

    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("TensorRT one-shot inference currently requires static tensor shapes");
        }

        const auto dimension = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("TensorRT tensor element count overflow");
        }
        count *= dimension;
    }

    return count;
}

std::size_t tensor_byte_size(const TensorMetadata& tensor, std::size_t element_count) {
    if (tensor.element_type != "float32") {
        throw std::runtime_error("TensorRT one-shot inference currently supports float32 tensors only: " + tensor.name);
    }

    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("TensorRT tensor byte size overflow: " + tensor.name);
    }

    return element_count * sizeof(float);
}

std::vector<float> load_image_tensor_nchw(
    const RuntimeConfig& config,
    const TensorMetadata& tensor,
    std::size_t element_count) {
    if (tensor.shape.size() != 4) {
        throw std::runtime_error("real image input currently requires NCHW 4D input tensor: " + tensor.name);
    }

    const int64_t batch = tensor.shape[0];
    const int64_t channels = tensor.shape[1];
    const int64_t height = tensor.shape[2];
    const int64_t width = tensor.shape[3];
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error("real image input requires positive tensor dimensions: " + tensor.name);
    }
    if (channels != 3) {
        throw std::runtime_error("real image input currently supports 3-channel NCHW tensors only: " + tensor.name);
    }

#ifndef INFEREDGE_OPENCV_LINKED
    (void)config;
    throw std::runtime_error("real image input requires OpenCV-enabled build: reconfigure with -DINFEREDGE_ENABLE_OPENCV=ON");
#else
    if (!std::filesystem::exists(config.input_path)) {
        throw std::runtime_error("input image file not found: " + config.input_path);
    }

    cv::Mat image = cv::imread(config.input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("failed to read input image: " + config.input_path);
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(static_cast<int>(width), static_cast<int>(height)));

    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<float> buffer(element_count, 0.0F);
    const std::size_t image_stride = static_cast<std::size_t>(channels * height * width);
    for (int64_t b = 0; b < batch; ++b) {
        const std::size_t batch_offset = static_cast<std::size_t>(b) * image_stride;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const cv::Vec3f pixel = float_image.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x));
                for (int64_t c = 0; c < channels; ++c) {
                    const std::size_t offset =
                        batch_offset +
                        static_cast<std::size_t>(c * height * width + y * width + x);
                    buffer[offset] = pixel[static_cast<int>(c)];
                }
            }
        }
    }
    return buffer;
#endif
}

void release_buffers(std::vector<TensorRTBuffer>& buffers) {
    for (TensorRTBuffer& buffer : buffers) {
        if (buffer.device_buffer != nullptr) {
            cudaFree(buffer.device_buffer);
            buffer.device_buffer = nullptr;
        }
    }
    buffers.clear();
}

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
#endif

}  // namespace

struct TensorRTEngineImpl {
    explicit TensorRTEngineImpl(RuntimeConfig runtime_config) : config(std::move(runtime_config)) {}

    RuntimeConfig config;
    std::string loaded_model_path;
    ModelMetadata model_metadata;

#ifdef INFEREDGE_TENSORRT_LINKED
    ~TensorRTEngineImpl() {
        release_buffers(buffers);
    }

    TensorRTLogger logger;
    RuntimePtr runtime;
    EnginePtr engine;
    ContextPtr context;
    std::vector<TensorRTBuffer> buffers;
#endif
};

#ifdef INFEREDGE_TENSORRT_LINKED
void prepare_buffers(TensorRTEngineImpl& impl) {
    release_buffers(impl.buffers);

    if (!impl.engine) {
        throw std::runtime_error("TensorRT engine is not loaded");
    }

    const auto add_buffer = [&impl](const TensorMetadata& tensor, bool is_input) {
        TensorRTBuffer buffer;
        buffer.name = tensor.name;
        buffer.is_input = is_input;
        buffer.element_type = tensor.element_type;
        buffer.shape = tensor.shape;
        buffer.element_count = tensor_element_count(buffer.shape);
        buffer.byte_size = tensor_byte_size(tensor, buffer.element_count);
        if (is_input && !impl.config.input_path.empty()) {
            buffer.host_float_buffer = load_image_tensor_nchw(impl.config, tensor, buffer.element_count);
        } else {
            buffer.host_float_buffer.assign(buffer.element_count, 0.0F);
        }

        check_cuda(cudaMalloc(&buffer.device_buffer, buffer.byte_size), "Failed to allocate CUDA buffer: " + buffer.name);
        check_cuda(cudaMemset(buffer.device_buffer, 0, buffer.byte_size), "Failed to initialize CUDA buffer: " + buffer.name);

        impl.buffers.push_back(std::move(buffer));
    };

    for (const TensorMetadata& tensor : impl.model_metadata.inputs) {
        add_buffer(tensor, true);
    }
    for (const TensorMetadata& tensor : impl.model_metadata.outputs) {
        add_buffer(tensor, false);
    }

    if (impl.buffers.empty()) {
        throw std::runtime_error("TensorRT buffer preparation failed: no tensors");
    }
}

void execute_once(TensorRTEngineImpl& impl) {
    if (!impl.context) {
        throw std::runtime_error("TensorRT model is not loaded");
    }

    if (impl.buffers.empty()) {
        prepare_buffers(impl);
    }

    CudaStreamGuard stream;

    for (TensorRTBuffer& buffer : impl.buffers) {
        if (buffer.is_input) {
            check_cuda(
                cudaMemcpyAsync(
                    buffer.device_buffer,
                    buffer.host_float_buffer.data(),
                    buffer.byte_size,
                    cudaMemcpyHostToDevice,
                    stream.stream),
                "Failed to copy TensorRT input to device: " + buffer.name);
        }

        if (!impl.context->setTensorAddress(buffer.name.c_str(), buffer.device_buffer)) {
            throw std::runtime_error("Failed to set TensorRT tensor address: " + buffer.name);
        }
    }

    if (!impl.context->enqueueV3(stream.stream)) {
        throw std::runtime_error("Failed to enqueue TensorRT inference");
    }

    for (TensorRTBuffer& buffer : impl.buffers) {
        if (!buffer.is_input) {
            check_cuda(
                cudaMemcpyAsync(
                    buffer.host_float_buffer.data(),
                    buffer.device_buffer,
                    buffer.byte_size,
                    cudaMemcpyDeviceToHost,
                    stream.stream),
                "Failed to copy TensorRT output to host: " + buffer.name);
        }
    }

    check_cuda(cudaStreamSynchronize(stream.stream), "Failed to synchronize TensorRT inference stream");
}
#endif

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
    metadata.status_message =
        "TensorRT backend is linked. Engine metadata, one-shot inference, and benchmark execution are available.";
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

    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    prepare_buffers(*impl_);
#endif
}

void TensorRTEngine::run_once() {
#ifdef INFEREDGE_TENSORRT_LINKED
    execute_once(*impl_);
#else
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

BenchmarkResult TensorRTEngine::benchmark(int warmup, int runs) {
#ifdef INFEREDGE_TENSORRT_LINKED
    if (!impl_->context) {
        throw std::runtime_error("TensorRT model is not loaded");
    }

    if (warmup < 0) {
        throw std::runtime_error("warmup must be greater than or equal to 0");
    }

    if (runs < 1) {
        throw std::runtime_error("runs must be greater than or equal to 1");
    }

    for (int i = 0; i < warmup; ++i) {
        execute_once(*impl_);
    }

    BenchmarkResult result;
    result.success = true;
    result.status = "success";
    result.message = "benchmark completed";
    result.warmup_runs = warmup;
    result.timed_runs = runs;
    result.samples_ms.reserve(static_cast<std::size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        execute_once(*impl_);
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
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

}  // namespace inferedge_runtime
