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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/dynload/nvrtc.h"
#endif

namespace paddle {
namespace platform {

class DeviceCode {
 public:
  virtual ~DeviceCode() {}
  virtual void Compile() = 0;
  virtual void Launch(const size_t n, std::vector<void*>* args) const = 0;

 protected:
  Place place_;
  std::string name_;
  std::string kernel_;
};

#ifdef PADDLE_WITH_CUDA
class CUDADeviceCode : public DeviceCode {
 public:
  explicit CUDADeviceCode(const Place& place, const std::string& name,
                          const std::string& kernel);
  void Compile() override;
  void Launch(const size_t n, std::vector<void*>* args) const override;

  void SetNumThreads(int num_threads) { num_threads_ = num_threads; }
  void SetWorkloadPerThread(int workload_per_thread) {
    workload_per_thread_ = workload_per_thread;
  }

 private:
  int max_threads_{0};
  int num_threads_{1024};
  int workload_per_thread_{1};
  std::vector<char> ptx_;
  CUmodule module_;
  CUfunction function_;
};
#endif

}  // namespace platform
}  // namespace paddle
