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

#pragma once

#ifdef CINN_WITH_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <mutex>
#include <string>
#include <vector>

#include "paddle/cinn/runtime/xpu/xpu_util.h"

namespace cinn {
namespace runtime {
namespace xpu {

const int kXpuMaxCards{8};

/**
 * XpuModule wraps a compiled PTX/CUBIN blob and manages per-card CUmodule
 * handles.  The CUDA driver API is used to load kernels, mirroring the
 * CUDAModule design.
 */
class XpuModule {
 public:
  enum class Kind {
    PTX = 0,
    CUBIN = 1,
  };

  XpuModule(const std::string& data, Kind kind);

  //! Get a kernel function handle for the given device and function name.
  CUfunction GetFunction(int device_id, const std::string& func_name);

  //! Convenience overload using the currently active CUDA device.
  CUfunction GetFunction(const std::string& func_name);

  ~XpuModule();

 private:
  std::string data_;
  Kind kind_;
  std::vector<CUmodule> module_per_card_{kXpuMaxCards, nullptr};
  std::mutex mutex_;

  CUdevice device_;
  CUcontext context_;
  int num_devices_{0};
};

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
