// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/backends/xpu/compiler_xpu.h"

#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>

#ifdef CINN_WITH_XPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#endif

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {
namespace xpurtc {

#ifdef CINN_WITH_XPU
#define NVRTC_CHECK(expr)                                                     \
  {                                                                           \
    nvrtcResult status = (expr);                                              \
    if (status != NVRTC_SUCCESS) {                                            \
      PADDLE_THROW(::common::errors::Fatal(                                   \
          "NVRTC Error in XPU CINN: %s", nvrtcGetErrorString(status)));       \
    }                                                                         \
  }

#define CUDA_CHECK(expr)                                                      \
  {                                                                           \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      PADDLE_THROW(::common::errors::Fatal(                                   \
          "CUDA Error in XPU CINN: %s", cudaGetErrorString(status)));         \
    }                                                                         \
  }
#endif  // CINN_WITH_XPU

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  return CompileWithNvrtc(code, include_headers);
}

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path = std::string(cuda_path_env) + delimiter + "include";
    return {cuda_include_path};
  }

#if defined(__linux__)
  struct stat st;
  cuda_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return {cuda_include_path};
  }
#endif
  PADDLE_THROW(::common::errors::Fatal(
      "Cannot find CUDA include path. CUDA_PATH is not set or CUDA is not "
      "installed in the default path. Set CUDA_PATH to your CUDA installation "
      "directory."));
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileWithNvrtc(const std::string& code,
                                       bool include_headers) {
#ifndef CINN_WITH_XPU
  PADDLE_THROW(::common::errors::Unimplemented(
      "CompileWithNvrtc requires CINN_WITH_XPU to be enabled."));
  return "";
#else
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};

  compile_options.push_back(std::string("-arch=") + GetDeviceArch());
  compile_options.push_back("-std=c++17");
  compile_options.push_back("--use_fast_math");

  if (include_headers) {
    std::vector<std::string> cuda_headers = FindCUDAIncludePaths();
    std::vector<std::string> cinn_headers = FindCINNRuntimeIncludePaths();
    for (const auto& header : cuda_headers) {
      compile_options.push_back("--include-path=" + header);
    }
    for (const auto& header : cinn_headers) {
      compile_options.push_back("--include-path=" + header);
    }
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(5) << "xpu (nvrtc) compile options: "
          << utils::Join(compile_options, " ");

  nvrtcProgram prog;
  NVRTC_CHECK(
      nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  {
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    PADDLE_ENFORCE_EQ(
        compile_res,
        NVRTC_SUCCESS,
        ::common::errors::External("NVRTC compilation error (XPU): %s", log));
  }

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
#endif  // CINN_WITH_XPU
}

std::string Compiler::GetDeviceArch() {
#ifndef CINN_WITH_XPU
  return "sm_80";
#else
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &minor, cudaDevAttrComputeCapabilityMinor, 0));
  return "sm_" + std::to_string(major) + std::to_string(minor);
#endif
}

}  // namespace xpurtc
}  // namespace backends
}  // namespace cinn
