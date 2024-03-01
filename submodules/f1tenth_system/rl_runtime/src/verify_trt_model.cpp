#include <fmt/format.h>

#include <filesystem>

#include "infer_v0.h"
#include "trt_model.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 2) {
        fmt::print("Usage: verify_trt_model <PATH_TO_MODEL>\n");
        return 1;
    }

    const fs::path trt_model_path{argv[1]};
    fmt::print("Reading model from {}!", trt_model_path.string());

    rl_runtime::TRTModel model{trt_model_path};
    const auto n_obs = model.n_obs();

    model.input() = std::vector<float>(n_obs, 0.f);
    fmt::print("0: {}\n", fmt::join(model.run(), ", "));

    model.input() = std::vector<float>(n_obs, 1.f);
    fmt::print("1: {}\n", fmt::join(model.run(), ", "));

    return 0;
}