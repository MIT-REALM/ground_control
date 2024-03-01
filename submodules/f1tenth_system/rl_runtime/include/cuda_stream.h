#pragma once

#include <memory>

#include "cuda_runtime_api.h"

namespace rl_runtime {

struct StreamDeleter {
    void operator()(cudaStream_t *stream);
};

using CudaStreamPtr = std::unique_ptr<cudaStream_t, StreamDeleter>;

CudaStreamPtr make_cuda_stream();

}  // namespace rl_runtime
