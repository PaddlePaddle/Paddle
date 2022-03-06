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
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace platform {
class CPUDeviceContext;
class CUDADeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

template <typename DeviceContext, typename T>
class LayerNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class LayerNormDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream, const T* input,
                  std::vector<int> input_shape, const T* bias, const T* scale,
                  T* output, T* mean, T* variance, int begin_norm_axis,
                  float eps);
};
#endif

}  // namespace operators
}  // namespace paddle
