/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_code.h"
#include <algorithm>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
inline bool is_error(nvrtcResult stat) { return stat != NVRTC_SUCCESS; }

inline void throw_on_error(nvrtcResult stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(dynload::nvrtcGetErrorString(stat) + msg);
#else
  LOG(FATAL) << dynload::nvrtcGetErrorString(stat) << msg;
#endif
}

CUDADeviceCode::CUDADeviceCode(const Place& place, const std::string& name,
                               const std::string& kernel) {
  if (!is_gpu_place(place)) {
    PADDLE_THROW("CUDADeviceCode can only launch on GPU place.");
  }

  place_ = place;
  name_ = name;
  kernel_ = kernel;
}

void CUDADeviceCode::Compile() {
  nvrtcProgram program;
  PADDLE_ENFORCE_EQ(dynload::nvrtcCreateProgram(&program,
                                                kernel_.c_str(),  // buffer
                                                name_.c_str(),    // name
                                                0,                // numHeaders
                                                nullptr,          // headers
                                                nullptr),  // includeNames
                    NVRTC_SUCCESS,
                    "nvrtcCreateProgram failed.");

  // Compile the program for specified compute_capability
  auto* dev_ctx = reinterpret_cast<CUDADeviceContext*>(
      DeviceContextPool::Instance().Get(place_));
  int compute_capability = dev_ctx->GetComputeCapability();
  std::string compute_flag =
      "--gpu-architecture=compute_" + std::to_string(compute_capability);
  const std::vector<const char*> options = {"--std=c++11",
                                            compute_flag.c_str()};
  nvrtcResult compile_result =
      dynload::nvrtcCompileProgram(program,          // program
                                   options.size(),   // numOptions
                                   options.data());  // options
  if (compile_result == NVRTC_ERROR_COMPILATION) {
    // Obtain compilation log from the program
    size_t log_size;
    PADDLE_ENFORCE_EQ(dynload::nvrtcGetProgramLogSize(program, &log_size),
                      NVRTC_SUCCESS, "nvrtcGetProgramLogSize failed.");
    std::vector<char> log;
    log.resize(log_size + 1);
    PADDLE_ENFORCE_EQ(dynload::nvrtcGetProgramLog(program, log.data()),
                      NVRTC_SUCCESS, "nvrtcGetProgramLog failed.");
    LOG(FATAL) << "JIT compiling of CUDA code failed:\n" << log.data();
  }

  // Obtain PTX from the program
  size_t ptx_size;
  PADDLE_ENFORCE_EQ(dynload::nvrtcGetPTXSize(program, &ptx_size), NVRTC_SUCCESS,
                    "nvrtcGetPTXSize failed.");
  ptx_.resize(ptx_size + 1);
  PADDLE_ENFORCE_EQ(dynload::nvrtcGetPTX(program, ptx_.data()), NVRTC_SUCCESS,
                    "nvrtcGetPTX failed.");

  PADDLE_ENFORCE_EQ(dynload::nvrtcDestroyProgram(&program), NVRTC_SUCCESS,
                    "nvrtcDestroyProgram failed.");

  PADDLE_ENFORCE_EQ(
      dynload::cuModuleLoadData(&module_, ptx_.data()), CUDA_SUCCESS,
      "Fail to load PTX of %s (in cuModuleLoadData.)", name_.c_str());
  PADDLE_ENFORCE_EQ(
      dynload::cuModuleGetFunction(&function_, module_, name_.c_str()),
      CUDA_SUCCESS, "Fail to get function of %s (in cuModuleGetFunction.)",
      name_.c_str());

  max_threads_ = dev_ctx->GetMaxPhysicalThreadCount();
}

void CUDADeviceCode::Launch(const size_t n, std::vector<void*>* args) const {
  int max_blocks = std::max(max_threads_ / num_threads_, 1);
  int workload_per_block = workload_per_thread_ * num_threads_;
  int num_blocks =
      std::min(max_blocks, (static_cast<int>(n) + workload_per_block - 1) /
                               workload_per_block);

  auto* dev_ctx = reinterpret_cast<CUDADeviceContext*>(
      DeviceContextPool::Instance().Get(place_));
  PADDLE_ENFORCE_EQ(
      dynload::cuLaunchKernel(function_, num_blocks, 1, 1,  // grid dim
                              num_threads_, 1, 1,           // block dim
                              0,                            // shared memory
                              dev_ctx->stream(),            // stream
                              args->data(),                 // arguments
                              nullptr),
      CUDA_SUCCESS, "Fail to launch kernel %s (in cuLaunchKernel.)",
      name_.c_str());
}
#endif

}  // namespace platform
}  // namespace paddle
