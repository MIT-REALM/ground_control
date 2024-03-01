#pragma once

#pragma once

#include <cstdint>

#include "cuda_runtime.h"

namespace detail {
/**
 * @brief Function which does the actual checking.
 * @param error
 * @param file
 * @param line
 * @param expr
 * @param error_msg
 */
void check_cuda_impl(cudaError_t error, const char *file, uint32_t line, const char *expr,
                     const char *error_msg);
}  // namespace detail

#define CHECK_CUDA_OK(expr)                                             \
    do {                                                                \
        const cudaError_t error = expr;                                 \
        if (error != cudaSuccess) {                                     \
            ::detail::check_cuda_impl(error, __FILE__, __LINE__, #expr, \
                                      cudaGetErrorString(error));       \
        }                                                               \
    } while (false)
