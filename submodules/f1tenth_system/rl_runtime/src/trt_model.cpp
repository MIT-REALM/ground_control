#include "trt_model.h"

#include <fmt/format.h>

#include <fstream>
#include <vector>

#include "cuda_common.h"
#include "cuda_stream.h"

#include "logger_trt.h"

namespace rl_runtime {

Logger g_logger{};

Logger &get_logger() { return g_logger; }

std::vector<char> load_file(const fs::path &filepath) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);

    if (!ifs) {
        throw std::runtime_error(fmt::format("{}: {}", filepath.string(), std::strerror(errno)));
    }

    auto end = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    auto size = std::size_t(end - ifs.tellg());

    if (size == 0)  // avoid undefined behavior
        return {};

    std::vector<char> buffer(size);

    if (!ifs.read(buffer.data(), static_cast<std::streamsize>(buffer.size()))) {
        throw std::runtime_error(fmt::format("{}: {}", filepath.string(), std::strerror(errno)));
    }

    return buffer;
}

size_t get_total_bytes(nvinfer1::Dims dims) {
    size_t n_bytes = 1;
    for (int ii = 0; ii < dims.nbDims; ++ii) {
        n_bytes *= dims.d[ii];
    }
    return n_bytes;
}

TRTModel::TRTModel(const fs::path &trt_model_path) : stream_{} {
    Logger &logger = get_logger();

    // Read serialized TensorRT model.
    const auto trt_model_bytes = load_file(trt_model_path);

    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(trt_model_bytes.data(), trt_model_bytes.size());
    context_ = engine_->createExecutionContext();

    // Allocate buffers.
    const auto *name_input = engine_->getIOTensorName(0);
    const auto *name_output = engine_->getIOTensorName(1);
    if (nvinfer1::TensorIOMode::kINPUT != engine_->getTensorIOMode(name_input)) {
        throw std::runtime_error(fmt::format("Expected {} to be an input tensor."));
    }
    if (nvinfer1::TensorIOMode::kOUTPUT != engine_->getTensorIOMode(name_output)) {
        throw std::runtime_error(fmt::format("Expected {} to be an output tensor."));
    }
    auto dims_input = engine_->getTensorShape(name_input);
    auto dims_output = engine_->getTensorShape(name_output);
    hbuf_input_ = std::vector<float>(get_total_bytes(dims_input), 0);
    hbuf_output_ = std::vector<float>(get_total_bytes(dims_output), 0);

    constexpr auto dtype_size = sizeof(float);
    buf_input_bytes_ = dtype_size * hbuf_input_.size();
    buf_output_bytes_ = dtype_size * hbuf_output_.size();

    CHECK_CUDA_OK(cudaMalloc(&dbuf_input_, buf_input_bytes_));
    CHECK_CUDA_OK(cudaMalloc(&dbuf_output_, buf_output_bytes_));

    context_->setTensorAddress(name_input, dbuf_input_);
    context_->setTensorAddress(name_output, dbuf_output_);

    // Try inference.
    stream_ = make_cuda_stream();
}

const std::vector<float> &TRTModel::run() {
    CHECK_CUDA_OK(cudaMemcpyAsync(dbuf_input_, hbuf_input_.data(), buf_input_bytes_,
                                  cudaMemcpyHostToDevice, *stream_));
    context_->enqueueV3(*stream_);
    CHECK_CUDA_OK(cudaMemcpyAsync(hbuf_output_.data(), dbuf_output_, buf_output_bytes_,
                                  cudaMemcpyDeviceToHost, *stream_));
    CHECK_CUDA_OK(cudaStreamSynchronize(*stream_));
    return hbuf_output_;
}

}  // namespace rl_runtime