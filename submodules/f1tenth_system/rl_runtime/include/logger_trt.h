#pragma once

#include "NvInfer.h"

namespace rl_runtime {
class Logger final : public ::nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};
}  // namespace rl_runtime
