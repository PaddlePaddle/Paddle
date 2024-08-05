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

#include "paddle/phi/backends/device_code.h"

#include <glog/logging.h>
#include <sys/stat.h>

#include <algorithm>
#include <set>
#include <utility>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

PHI_DECLARE_string(cuda_dir);

namespace phi {

DeviceCodePool* DeviceCodePool::pool = nullptr;

void DeviceCodePool::Set(std::unique_ptr<DeviceCode>&& code) {
  Place place = code->GetPlace();
  std::string name = code->GetName();

  auto iter = device_codes_.find(place);
  if (iter == device_codes_.end()) {
    PADDLE_THROW(common::errors::NotFound(
        "Place %s is not supported for runtime compiling.", place));
  }

  auto& codes_map = iter->second;
  codes_map.emplace(name, std::move(code));
}

DeviceCode* DeviceCodePool::Get(const phi::Place& place,
                                const std::string& name) {
  auto iter = device_codes_.find(place);
  if (iter == device_codes_.end()) {
    PADDLE_THROW(common::errors::NotFound(
        "Place %s is not supported for runtime compiling.", place));
  }

  auto& codes_map = iter->second;
  auto code_iter = codes_map.find(name);
  if (code_iter == codes_map.end()) {
    PADDLE_THROW(common::errors::NotFound(
        "Device code named %s for place %s does not exist.",
        name.c_str(),
        place));
  }

  return code_iter->second.get();
}

DeviceCodePool::DeviceCodePool(const std::vector<phi::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(),
                    0,
                    errors::InvalidArgument(
                        "Expected the number of places >= 1. But received %d.",
                        places.size()));
  // Remove the duplicated places
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }
  for (auto& p : set) {
    if (p.GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      device_codes_.emplace(p, DeviceCodeMap());
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "CUDAPlace or HIPPlace is not supported, please re-compile with "
          "WITH_GPU=ON or WITH_ROCM=ON."));
#endif
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  GPUDeviceCode::CheckAvailableStatus();
#endif
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_HIP
static bool CheckCUDADriverResult(hipError_t result,
                                  std::string caller,
                                  std::string kernel_name = "") {
  if (result != hipSuccess) {
    const char* error = nullptr;
    error = dynload::hipGetErrorString(result);
#else
static bool CheckCUDADriverResult(CUresult result,
                                  std::string caller,
                                  std::string kernel_name = "") {
  if (result != CUDA_SUCCESS) {
    const char* error = nullptr;
    dynload::cuGetErrorString(result, &error);
#endif
    LOG_FIRST_N(WARNING, 1) << "Call " << caller << " for < " << kernel_name
                            << " > failed: " << error << " (" << result << ")";
    return false;
  }
  return true;
}

bool GPUDeviceCode::available_ = false;
void GPUDeviceCode::CheckAvailableStatus() {
  available_ = false;
  if (!dynload::HasNVRTC() || !dynload::HasCUDADriver()) {
    LOG_FIRST_N(WARNING, 1)
        << "NVRTC and CUDA driver are need for JIT compiling of CUDA code.";
    return;
  }

  int nvrtc_major = 0;
  int nvrtc_minor = 0;
#ifdef PADDLE_WITH_HIP
  hiprtcResult nvrtc_result =
      dynload::hiprtcVersion(&nvrtc_major, &nvrtc_minor);
#else
  nvrtcResult nvrtc_result = dynload::nvrtcVersion(&nvrtc_major, &nvrtc_minor);
#endif

  int driver_version = 0;
  int driver_major = 0;
  int driver_minor = 0;
#ifdef PADDLE_WITH_HIP
  hipError_t driver_result = dynload::hipDriverGetVersion(&driver_version);
  if (driver_result == hipSuccess) {
#else
  CUresult driver_result = dynload::cuDriverGetVersion(&driver_version);
  if (driver_result == CUDA_SUCCESS) {
#endif
    driver_major = driver_version / 1000;
    driver_minor = (driver_version % 1000) / 10;
  }

  LOG_FIRST_N(INFO, 1) << "CUDA Driver Version: " << driver_major << "."
                       << driver_minor << "; NVRTC Version: " << nvrtc_major
                       << "." << nvrtc_minor;
#ifdef PADDLE_WITH_HIP
  if (nvrtc_result != HIPRTC_SUCCESS || driver_result != hipSuccess) {
#else
  if (nvrtc_result != NVRTC_SUCCESS || driver_result != CUDA_SUCCESS) {
#endif
    return;
  }

  int count = 0;
#ifdef PADDLE_WITH_HIP
  if (CheckCUDADriverResult(dynload::hipGetDeviceCount(&count),
                            "hipGetDeviceCount")) {
#else
  if (CheckCUDADriverResult(dynload::cuDeviceGetCount(&count),
                            "cuDeviceGetCount")) {
#endif
    available_ = true;
  }
}

static std::string FindCUDAIncludePath() {
  auto EndWith = [](std::string str, std::string substr) -> bool {
    size_t pos = str.rfind(substr);
    return pos != std::string::npos && pos == (str.length() - substr.length());
  };

  struct stat st = {};
  std::string cuda_include_path;
  if (!FLAGS_cuda_dir.empty()) {
    cuda_include_path = FLAGS_cuda_dir;
    if (EndWith(cuda_include_path, "/")) {
      cuda_include_path.erase(cuda_include_path.end() - 1);
    }
    for (std::string suffix : {"/lib", "/lib64"}) {
      if (EndWith(FLAGS_cuda_dir, suffix)) {
        cuda_include_path.erase(cuda_include_path.end() -
                                suffix.length());  // NOLINT
        break;
      }
    }

    if (!EndWith(cuda_include_path, "include")) {
      cuda_include_path += "/include";
    }
    // Whether the cuda_include_path exists on the file system.
    if (stat(cuda_include_path.c_str(), &st) == 0) {
      return cuda_include_path;
    }
  }

#ifdef PADDLE_WITH_HIP
  cuda_include_path = "/opt/rocm/include";
#else
  cuda_include_path = "/usr/local/cuda/include";
#endif

  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return cuda_include_path;
  }
  LOG(WARNING)
      << "Cannot find CUDA or ROCM include path."
      << "Please check whether CUDA or ROCM is installed in the default "
         "installation path, or specify it by export "
         "FLAGS_cuda_dir=xxx.";
  return "";
}

GPUDeviceCode::GPUDeviceCode(const Place& place,
                             const std::string& name,
                             const std::string& kernel)
    : module_(nullptr), function_(nullptr) {
  if (place.GetType() != phi::AllocationType::GPU) {
    PADDLE_THROW(common::errors::PermissionDenied(
        "GPUDeviceCode can only launch on GPU place."));
  }

  place_ = place;
  name_ = name;
#ifdef PADDLE_WITH_HIP
  kernel_ = "#include <hip/hip_runtime.h>\n" + kernel;
#else
  kernel_ = kernel;
#endif
}

bool GPUDeviceCode::Compile(bool include_path) {
  is_compiled_ = false;
  if (!dynload::HasNVRTC() || !dynload::HasCUDADriver()) {
    LOG_FIRST_N(WARNING, 1)
        << "NVRTC and CUDA driver are need for JIT compiling of CUDA code.";
    return false;
  }
#ifdef PADDLE_WITH_HIP
  hiprtcProgram program;
  if (!CheckNVRTCResult(dynload::hiprtcCreateProgram(&program,
                                                     kernel_.c_str(),  // buffer
                                                     name_.c_str(),    // name
                                                     0,         // numHeaders
                                                     nullptr,   // headers
                                                     nullptr),  // includeNames
                        "hiprtcCreateProgram")) {
    return false;
  }

  // Compile the program for specified compute_capability
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      DeviceContextPool::Instance().Get(place_));
  int compute_capability = dev_ctx->GetComputeCapability();
  std::vector<const char*> options = {"-std=c++11"};
  std::string include_option;
  if (include_path) {
    std::string cuda_include_path = FindCUDAIncludePath();
    if (!cuda_include_path.empty()) {
      include_option = "-I" + cuda_include_path;
      options.push_back(include_option.c_str());
    }
  }
  hiprtcResult compile_result =
      dynload::hiprtcCompileProgram(program,          // program
                                    options.size(),   // numOptions
                                    options.data());  // options
  if (compile_result == HIPRTC_ERROR_COMPILATION) {
    // Obtain compilation log from the program
    size_t log_size;
    if (!CheckNVRTCResult(dynload::hiprtcGetProgramLogSize(program, &log_size),
                          "hiprtcGetProgramLogSize")) {
      return false;
    }
    std::vector<char> log;
    log.resize(log_size + 1);
    if (!CheckNVRTCResult(dynload::hiprtcGetProgramLog(program, log.data()),
                          "hiprtcGetProgramLog")) {
      return false;
    }
    LOG(WARNING) << "JIT compiling of ROCM GPU code failed:"
                 << "\n  Kernel name: " << name_ << "\n  Kernel body:\n"
                 << kernel_ << "\n  Compiling log: " << log.data();

    return false;
  }

  // Obtain PTX from the program for cuda
  // Obtain Code from the program for hip
  size_t ptx_size;
  if (!CheckNVRTCResult(dynload::hiprtcGetCodeSize(program, &ptx_size),
                        "hiprtcGetCodeSize")) {
    return false;
  }
  ptx_.resize(ptx_size + 1);
  if (!CheckNVRTCResult(dynload::hiprtcGetCode(program, ptx_.data()),
                        "hiprtcGetCode")) {
    return false;
  }

  if (!CheckNVRTCResult(dynload::hiprtcDestroyProgram(&program),
                        "hiprtcDestroyProgram")) {
    return false;
  }

  if (!CheckCUDADriverResult(dynload::hipModuleLoadData(&module_, ptx_.data()),
                             "hipModuleLoadData")) {
    return false;
  }

  if (!CheckCUDADriverResult(
          dynload::hipModuleGetFunction(&function_, module_, name_.c_str()),
          "hipModuleGetFunction")) {
    return false;
  }
#else
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
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      DeviceContextPool::Instance().Get(place_));
  int compute_capability = dev_ctx->GetComputeCapability();
  std::string compute_flag =
      "--gpu-architecture=compute_" + std::to_string(compute_capability);
  std::vector<const char*> options = {"--std=c++11", compute_flag.c_str()};
  std::string include_option;
  if (include_path) {
    std::string cuda_include_path = FindCUDAIncludePath();
    if (!cuda_include_path.empty()) {
      include_option = "--include-path=" + cuda_include_path;
      options.push_back(include_option.c_str());
    }
  }
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
                             "cuModuleLoadData",
                             name_)) {
    return false;
  }

  if (!CheckCUDADriverResult(
          dynload::cuModuleGetFunction(&function_, module_, name_.c_str()),
          "cuModuleGetFunction",
          name_)) {
    return false;
  }
#endif

  max_threads_ = dev_ctx->GetMaxPhysicalThreadCount();
  is_compiled_ = true;
  return true;
}

void GPUDeviceCode::Launch(const size_t n, std::vector<void*>* args) const {
  PADDLE_ENFORCE_EQ(
      is_compiled_,
      true,
      errors::PreconditionNotMet(
          "Please compile the code before launching the kernel."));

  int max_blocks = std::max(max_threads_ / num_threads_, 1);
  int workload_per_block = workload_per_thread_ * num_threads_;
  int num_blocks = std::min(
      max_blocks,
      (static_cast<int>(n) + workload_per_block - 1) / workload_per_block);

  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      DeviceContextPool::Instance().Get(place_));
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_EQ(
      dynload::hipModuleLaunchKernel(function_,
                                     num_blocks,
                                     1,
                                     1,  // grid dim
                                     num_threads_,
                                     1,
                                     1,                  // block dim
                                     0,                  // shared memory
                                     dev_ctx->stream(),  // stream
                                     args->data(),       // arguments
                                     nullptr),
      hipSuccess,
      errors::External("Fail to launch kernel %s (in hipModuleLaunchKernel.)",
                       name_.c_str()));
#else
  PADDLE_ENFORCE_EQ(
      dynload::cuLaunchKernel(function_,
                              num_blocks,
                              1,
                              1,  // grid dim
                              num_threads_,
                              1,
                              1,                  // block dim
                              0,                  // shared memory
                              dev_ctx->stream(),  // stream
                              args->data(),       // arguments
                              nullptr),
      CUDA_SUCCESS,
      errors::External("Fail to launch kernel %s (in cuLaunchKernel.)",
                       name_.c_str()));
#endif
}

#ifdef PADDLE_WITH_HIP
bool GPUDeviceCode::CheckNVRTCResult(hiprtcResult result,
                                     std::string function) {
  if (result != HIPRTC_SUCCESS) {
    LOG_FIRST_N(WARNING, 1)
        << "Call " << function << " for < " << name_
        << " > failed: " << dynload::hiprtcGetErrorString(result);
    return false;
  }
#else
bool GPUDeviceCode::CheckNVRTCResult(nvrtcResult result, std::string function) {
  if (result != NVRTC_SUCCESS) {
    LOG_FIRST_N(WARNING, 1)
        << "Call " << function << " for < " << name_
        << " > failed: " << dynload::nvrtcGetErrorString(result);
    return false;
  }
#endif
  return true;
}
#endif

}  // namespace phi
