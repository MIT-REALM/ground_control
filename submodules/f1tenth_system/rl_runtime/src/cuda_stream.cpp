#include "cuda_stream.h"

namespace rl_runtime {

void StreamDeleter::operator()(cudaStream_t *stream) {
    if (stream) {
        cudaStreamDestroy(*stream);
        delete stream;
    }
}

CudaStreamPtr make_cuda_stream() {
    CudaStreamPtr stream(new cudaStream_t, StreamDeleter{});
    if (cudaStreamCreate(stream.get()) != cudaSuccess) {
        stream.reset(nullptr);
    }

    return stream;
}
}  // namespace rl_runtime
