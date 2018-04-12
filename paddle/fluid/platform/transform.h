/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <type_traits>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/fluid/platform/place.h"

#ifdef __NVCC__
#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/transform.h"
#endif

namespace paddle {
namespace platform {

// Transform on host or device. It provides the same API in std library.
template <typename DeviceContext>
struct Transform {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const DeviceContext& context, InputIter first, InputIter last,
                  OutputIter result, UnaryOperation op);

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const DeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op);
};

template <>
struct Transform<platform::CPUDeviceContext> {
  template <typename T, typename UnaryOperation>
  void operator()(const platform::CPUDeviceContext& context, const T* first,
                  const T* last, T* result, UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename T, typename BinaryOperation>
  void operator()(const platform::CPUDeviceContext& context, const T* first1,
                  const T* last1, const T* first2, T* result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

#ifdef __NVCC__
template <>
struct Transform<platform::CUDADeviceContext> {
  template <typename T, typename UnaryOperation>
  void operator()(const platform::CUDADeviceContext& context, const T* first,
                  const T* last, T* result, UnaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place), "It must use GPU place.");
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      thrust::device_pointer_cast(first),
                      thrust::device_pointer_cast(last),
                      thrust::device_pointer_cast(result), op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const platform::CUDADeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place), "It must use GPU place.");
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      thrust::device_pointer_cast(first1),
                      thrust::device_pointer_cast(last1),
                      thrust::device_pointer_cast(first2),
                      thrust::device_pointer_cast(result), op);
  }
};
#endif

}  // namespace platform
}  // namespace paddle
