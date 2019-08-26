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
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/dynload/nvrtc.h"
#endif

namespace paddle {
namespace platform {

enum DataType { INT = 0, FLOAT = 1, FLOAT_PTR = 2 };

class DeviceCode {
 public:
  virtual ~DeviceCode() {}
  virtual void Compile() = 0;
  virtual void Launch(Place place, const size_t n,
                      std::vector<void*>* args) const = 0;

 protected:
  std::string name_;
  std::vector<DataType> formals_;
  std::string kernel_;
};

#ifdef PADDLE_WITH_CUDA
class CUDADeviceCode : public DeviceCode {
 public:
  explicit CUDADeviceCode(const std::string& name, const std::string& kernel,
                          int compute_capability);
  void Compile() override;
  void Launch(Place place, const size_t n,
              std::vector<void*>* args) const override;

 private:
  int compute_capability_;
  std::vector<char> ptx_;
  CUmodule module_;
  CUfunction function_;
};
#endif

}  // namespace platform
}  // namespace paddle
