/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"
// See Note [ Why still include the fluid headers? ]
#include "paddle/pten/common/float16.h"

namespace pten {

template <typename InT>
struct ScaleFunctor {
  InT bias;
  InT scale;
  bool bias_after_scale;

  ScaleFunctor(InT scale_data, InT bias_data, bool is_bias_after_sacle) {
    scale = scale_data;
    bias = bias_data;
    bias_after_scale = is_bias_after_sacle;
  }

  __device__ __forceinline__ InT operator()(const InT x) const {
    if (bias_after_scale) {
      return scale * x + bias;
    } else {
      return scale * (x + bias);
    }
  }
};

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  outputs.emplace_back(out);
  out->mutable_data<T>();
  pten::funcs::LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary,
                                                   T,
                                                   T>(
      dev_ctx,
      inputs,
      &outputs,
      ScaleFunctor<T>(scale.to<T>(), static_cast<T>(bias), bias_after_scale));
}

}  // namespace pten

PT_REGISTER_KERNEL(scale,
                   GPU,
                   ALL_LAYOUT,
                   pten::ScaleKernel,
                   float,
                   double,
                   paddle::platform::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
