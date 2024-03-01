#pragma once

#include <filesystem>
#include <vector>

#include "NvInfer.h"
#include "cuda_stream.h"

namespace rl_runtime {
namespace fs = std::filesystem;
class TRTModel {
   public:
    explicit TRTModel(const fs::path& trt_model_path);

    [[nodiscard]] size_t n_obs() const { return hbuf_input_.size(); }

    std::vector<float>& input() { return hbuf_input_; };
    [[nodiscard]] const std::vector<float>& output() const { return hbuf_output_; };

    const std::vector<float>& run();

   private:
    std::vector<float> hbuf_input_;
    std::vector<float> hbuf_output_;
    void* dbuf_input_;
    void* dbuf_output_;

    size_t buf_input_bytes_;
    size_t buf_output_bytes_;

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    CudaStreamPtr stream_;
};
}  // namespace rl_runtime