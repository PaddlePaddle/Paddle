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

#pragma once
#ifdef CINN_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"

namespace cinn {
namespace runtime {
namespace cuda {

/**
 * The CUDA module, helps to compile CUDA codes and fetch symbols.
 * Currently, it is a wrapper of NVRTC.
 */
class CUDAModule {
 public:
  enum class Kind {
    PTX = 0,
    CUBIN = 1,
  };

  CUDAModule(const std::string& data, Kind kind);

  void LaunchKernel(int device_id,
                    const std::string& func_name,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t share_memory_size = 0,
                    CUstream stream = nullptr);

  //! Get a function.
  CUfunction GetFunction(int device_id, const std::string& func_name);

  //! Get a function by CudaGetDevice
  CUfunction GetFunction(const std::string& func_name);

  //! Get a global variable.
  CUdeviceptr GetGlobal(int device_id, const std::string& name, size_t nbytes);

  ~CUDAModule();

 private:
  //! The input data.
  std::string data_;
  //! Kind of the input.
  Kind kind_;
  //! To make parallel, we prepare one module for each card.
  std::vector<CUmodule> module_per_card_{kCUDAMaxCards, nullptr};
  std::string cuda_source_;
  std::mutex mutex_;

  CUdevice device_;
  CUcontext context_;
  int num_devices_{0};
};

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn

#endif  // CINN_WITH_CUDA
