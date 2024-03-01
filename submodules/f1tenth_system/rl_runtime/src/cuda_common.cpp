#include "cuda_common.h"

#include <fmt/format.h>

namespace detail {

void check_cuda_impl(cudaError_t error, const char *file, uint32_t line, const char *expr,
                     const char *error_msg) {
    const auto msg = fmt::format(
        "CUDA fn call error at {}:{}\n"
        "  Error: {}\n"
        "  Failing expression: {}",
        file, line, error_msg, expr);

    throw std::runtime_error(msg);
}
}  // namespace detail