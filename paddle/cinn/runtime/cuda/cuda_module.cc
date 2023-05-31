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

#include "paddle/cinn/runtime/cuda/cuda_module.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace cuda {

CUDAModule::CUDAModule(const std::string& data, Kind kind)
    : data_(data), kind_(kind) {
  CHECK(!data.empty());

  cudaGetDeviceCount(&num_devices_);
  CHECK_GT(num_devices_, 0) << "No available devices";

  // TODO(Superjomn) Determine whether to initialize all the devices.
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaSetDevice(current_device_id);
  cuDeviceGet(&device_, current_device_id);
  cuCtxGetCurrent(&context_);
  cuDevicePrimaryCtxRetain(&context_, device_);
}

void CUDAModule::LaunchKernel(int device_id,
                              const std::string& func_name,
                              dim3 gridDim,
                              dim3 blockDim,
                              void** args,
                              size_t share_memory_size,
                              CUstream stream) {
  VLOG(3) << "cuLaunchKernel with func_name : " << func_name
          << ", gridDim.x:" << gridDim.x << ", gridDim.y:" << gridDim.y
          << ", gridDim.z:" << gridDim.z << ", blockDim.x:" << blockDim.x
          << ", blockDim.y:" << blockDim.y << ", blockDim.z:" << blockDim.z
          << ", share_memory_size:" << share_memory_size;
  auto function = GetFunction(device_id, func_name);
  CHECK(function);
  cinn::utils::RecordEvent record_run("cuLaunchKernel",
                                      cinn::utils::EventType::kInstruction);
  CUDA_DRIVER_CALL(cuLaunchKernel(function,
                                  gridDim.x,
                                  gridDim.y,
                                  gridDim.z,
                                  blockDim.x,
                                  blockDim.y,
                                  blockDim.z,
                                  share_memory_size,
                                  stream,
                                  args,
                                  nullptr));
}

CUfunction CUDAModule::GetFunction(int device_id,
                                   const std::string& func_name) {
  VLOG(5) << "GetFuncion : " << func_name << " with device_id : " << device_id;
  cinn::utils::RecordEvent record_run("cuLaunchKernel",
                                      cinn::utils::EventType::kOrdinary);
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Compilation with parameters
    const size_t jit_num_options = 5;
    std::vector<CUjit_option> jit_options(jit_num_options);
    std::vector<void*> jit_opt_vals(jit_num_options);

    // set up size of compilation log buffer
    jit_options[0] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    size_t log_buffer_size = 1024;
    jit_opt_vals[0] = reinterpret_cast<void*>(log_buffer_size);

    // set up pointer to the compilation log buffer
    jit_options[1] = CU_JIT_ERROR_LOG_BUFFER;
    std::vector<char> log_buffer(log_buffer_size, '\0');
    jit_opt_vals[1] = log_buffer.data();

    int value = 1;
    // Specifies whether to create debug information in output (-g)
    jit_options[2] = CU_JIT_GENERATE_DEBUG_INFO;
    jit_opt_vals[2] = reinterpret_cast<void*>(value);

    // Generate verbose log messages
    jit_options[3] = CU_JIT_LOG_VERBOSE;
    jit_opt_vals[3] = reinterpret_cast<void*>(value);

    // Generate line number information (-lineinfo)
    jit_options[4] = CU_JIT_GENERATE_LINE_INFO;
    jit_opt_vals[4] = reinterpret_cast<void*>(value);

    CUresult status = cuModuleLoadDataEx(&module_per_card_[device_id],
                                         data_.c_str(),
                                         jit_num_options,
                                         jit_options.data(),
                                         jit_opt_vals.data());

    if (CUDA_SUCCESS != status) {
      RAW_LOG(ERROR, "PTX JIT ERROR LOG: %s\n.", log_buffer.data());
      const char* name;
      cuGetErrorName(status, &name);
      const char* msg;
      cuGetErrorString(status, &msg);
      RAW_LOG(FATAL,
              "The error `%s` occurs while compiling the ptx! And its message "
              "is `%s`.",
              name,
              msg);
    }
  }

  CUfunction func;
  CUDA_DRIVER_CALL(cuModuleGetFunction(
      &func, module_per_card_[device_id], func_name.c_str()));
  return func;
}

CUdeviceptr CUDAModule::GetGlobal(int device_id,
                                  const std::string& name,
                                  size_t nbytes) {
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    CUDA_DRIVER_CALL(
        cuModuleLoadData(&module_per_card_[device_id], data_.c_str()));
  }

  CUdeviceptr global;
  size_t _nbytes;
  CUDA_DRIVER_CALL(cuModuleGetGlobal(
      &global, &_nbytes, module_per_card_[device_id], name.c_str()));
  return global;
}

CUDAModule::~CUDAModule() {
  for (int i = 0; i < module_per_card_.size(); i++) {
    auto* module = module_per_card_[i];
    if (module) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_DRIVER_CALL(cuModuleUnload(module));
    }
  }
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
