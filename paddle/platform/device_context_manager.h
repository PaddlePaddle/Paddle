/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

template <typename T>
struct Converter;

template <>
struct Converter<CPUPlace> {
  using DeviceContextType = CPUDeviceContext;
};

#ifndef PADDLE_ONLY_CPU
template <>
struct Converter<GPUPlace> {
  using DeviceContextType = CUDADeviceContext;
};
#endif

class DeviceContextManager {
 public:
  DeviceContextManager();
  // ~DeviceContextManager();

  template <typename PlaceType, typename DeviceType = typename Converter<
                                    PlaceType>::DeviceContextType>
  DeviceType* GetDeviceContext(const PlaceType& place);

  // DeviceContext* GetDeviceContext(const Place& place);

  static DeviceContextManager* Get() {
    static DeviceContextManager inst;
    return &inst;
  }

 private:
  std::unique_ptr<CPUDeviceContext> cpu_context_;
#ifndef PADDLE_ONLY_CPU
  int device_count_;
  std::vector<std::unique_ptr<CUDADeviceContext>> cuda_contexts_;
#endif
};
}  // namespace platform
}  // namespace paddle
