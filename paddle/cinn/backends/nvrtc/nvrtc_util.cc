// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/nvrtc/header_generator.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/utils/string.h"

DECLARE_bool(nvrtc_compile_to_cubin);

namespace cinn {
namespace backends {
namespace nvrtc {

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  return CompileCudaSource(code, include_headers);
}

Compiler::Compiler() {
  if (FLAGS_nvrtc_compile_to_cubin) {
#if CUDA_VERSION >= 11010
    compile_to_cubin_ = true;
#endif
  }
  VLOG(4) << "FLAGS_nvrtc_compile_to_cubin: " << FLAGS_nvrtc_compile_to_cubin
          << ", compile_to_cubin_: " << compile_to_cubin_;
}

bool Compiler::compile_to_cubin() { return compile_to_cubin_; }

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    return {cuda_include_path};
  }

#if defined(__linux__)
  struct stat st;
  cuda_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return {cuda_include_path};
  }
#endif
  LOG(FATAL) << "Cannot find cuda include path."
             << "CUDA_PATH is not set or CUDA is not installed in the default "
                "installation path."
             << "In other than linux, it is necessary to set CUDA_PATH.";
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileCudaSource(const std::string& code,
                                        bool include_headers) {
  const auto& header_gen = JitSafeHeaderGenerator::GetInstance();
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }
  if (compile_to_cubin_) {
    compile_options.push_back("-arch=sm_" + cc);
  } else {
    compile_options.push_back("-arch=compute_" + cc);
  }
  compile_options.push_back("-std=c++14");
  compile_options.push_back("-default-device");

  if (include_headers) {  // prepare include headers
    auto cuda_headers = FindCUDAIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    compile_options.insert(
        std::end(compile_options), include_paths.begin(), include_paths.end());
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(3) << "compile options: " << utils::Join(compile_options, " ");
  NVRTC_CALL(nvrtcCreateProgram(&prog,
                                code.c_str(),
                                nullptr,
                                header_gen.size(),
                                header_gen.headers().data(),
                                header_gen.include_names().data()));
  nvrtcResult compile_res =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  {  // get log
    size_t log_size;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
    CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;
  }

  size_t size;
  std::string data;
  if (compile_to_cubin_) {
    NVRTC_CALL(nvrtcGetCUBINSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetCUBIN(prog, &data[0]));
  } else {
    NVRTC_CALL(nvrtcGetPTXSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetPTX(prog, &data[0]));
  }

  NVRTC_CALL(nvrtcDestroyProgram(&prog));
  return data;
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
