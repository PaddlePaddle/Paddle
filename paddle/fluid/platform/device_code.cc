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
#include <set>
#include <utility>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

DeviceCodePool* DeviceCodePool::pool = nullptr;

void DeviceCodePool::Set(std::unique_ptr<DeviceCode>&& code) {
  Place place = code->GetPlace();
  std::string name = code->GetName();

  auto iter = device_codes_.find(place);
  if (iter == device_codes_.end()) {
    PADDLE_THROW("Place %s is not supported.", place);
  }

  auto& codes_map = iter->second;
  codes_map.emplace(name, std::move(code));
}

platform::DeviceCode* DeviceCodePool::Get(const platform::Place& place,
                                          const std::string& name) {
  auto iter = device_codes_.find(place);
  if (iter == device_codes_.end()) {
    PADDLE_THROW("Place %s is not supported.", place);
  }

  auto& codes_map = iter->second;
  auto code_iter = codes_map.find(name);
  if (code_iter == codes_map.end()) {
    PADDLE_THROW("There is not a device code named %s for place %s.",
                 name.c_str(), place);
  }

  return code_iter->second.get();
}

DeviceCodePool::DeviceCodePool(const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(), 0);
  // Remove the duplicated places
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }
  for (auto& p : set) {
    if (is_gpu_place(p)) {
#ifdef PADDLE_WITH_CUDA
      device_codes_.emplace(p, DeviceCodeMap());
#else
      PADDLE_THROW(
          "'CUDAPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    }
  }
}

#ifdef PADDLE_WITH_CUDA
CUDADeviceCode::CUDADeviceCode(const Place& place, const std::string& name,
                               const std::string& kernel) {
  if (!is_gpu_place(place)) {
    PADDLE_THROW("CUDADeviceCode can only launch on GPU place.");
  }

  place_ = place;
  name_ = name;
  kernel_ = kernel;
}

bool CUDADeviceCode::Compile() {
  is_compiled_ = false;
  if (!dynload::HasNVRTC() || !dynload::HasCUDADriver()) {
    LOG(WARNING)
        << "NVRTC and CUDA driver are need for JIT compiling of CUDA code.";
    return false;
  }

  nvrtcProgram program;
  if (!CheckNVRTCResult(dynload::nvrtcCreateProgram(&program,
                                                    kernel_.c_str(),  // buffer
                                                    name_.c_str(),    // name
                                                    0,         // numHeaders
                                                    nullptr,   // headers
                                                    nullptr),  // includeNames
                        "nvrtcCreateProgram")) {
    return false;
  }

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
    if (!CheckNVRTCResult(dynload::nvrtcGetProgramLogSize(program, &log_size),
                          "nvrtcGetProgramLogSize")) {
      return false;
    }
    std::vector<char> log;
    log.resize(log_size + 1);
    if (!CheckNVRTCResult(dynload::nvrtcGetProgramLog(program, log.data()),
                          "nvrtcGetProgramLog")) {
      return false;
    }
    LOG(WARNING) << "JIT compiling of CUDA code failed:"
                 << "\n  Kernel name: " << name_ << "\n  Kernel body:\n"
                 << kernel_ << "\n  Compiling log: " << log.data();

    return false;
  }

  // Obtain PTX from the program
  size_t ptx_size;
  if (!CheckNVRTCResult(dynload::nvrtcGetPTXSize(program, &ptx_size),
                        "nvrtcGetPTXSize")) {
    return false;
  }
  ptx_.resize(ptx_size + 1);
  if (!CheckNVRTCResult(dynload::nvrtcGetPTX(program, ptx_.data()),
                        "nvrtcGetPTX")) {
    return false;
  }

  if (!CheckNVRTCResult(dynload::nvrtcDestroyProgram(&program),
                        "nvrtcDestroyProgram")) {
    return false;
  }

  if (!CheckCUDADriverResult(dynload::cuModuleLoadData(&module_, ptx_.data()),
                             "cuModuleLoadData")) {
    return false;
  }

  if (!CheckCUDADriverResult(
          dynload::cuModuleGetFunction(&function_, module_, name_.c_str()),
          "cuModuleGetFunction")) {
    return false;
  }

  max_threads_ = dev_ctx->GetMaxPhysicalThreadCount();
  is_compiled_ = true;
  return true;
}

void CUDADeviceCode::Launch(const size_t n, std::vector<void*>* args) const {
  PADDLE_ENFORCE_EQ(is_compiled_, true, "Please compile the code first.");

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

bool CUDADeviceCode::CheckNVRTCResult(nvrtcResult result,
                                      std::string function) {
  if (result != NVRTC_SUCCESS) {
    LOG(WARNING) << "Call " << function
                 << " failed: " << dynload::nvrtcGetErrorString(result);
    return false;
  }
  return true;
}

bool CUDADeviceCode::CheckCUDADriverResult(CUresult result,
                                           std::string function) {
  if (result != CUDA_SUCCESS) {
    const char* error = nullptr;
    LOG(WARNING) << "Call " << function
                 << " failed: " << dynload::cuGetErrorString(result, &error);
    return false;
  }
  return true;
}
#endif

}  // namespace platform
}  // namespace paddle
