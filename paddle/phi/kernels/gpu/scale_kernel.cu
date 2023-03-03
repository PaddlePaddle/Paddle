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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename InT, typename AuxT>
struct ScaleFunctor {
  AuxT bias;
  AuxT scale;
  bool bias_after_scale;

  ScaleFunctor(AuxT scale_data, AuxT bias_data, bool is_bias_after_sacle)
      : bias(bias_data),
        scale(scale_data),
        bias_after_scale(is_bias_after_sacle) {}

  __device__ __forceinline__ InT operator()(const InT x) const {
    if (bias_after_scale) {
      return static_cast<InT>(scale * static_cast<AuxT>(x) + bias);
    } else {
      return static_cast<InT>(scale * (static_cast<AuxT>(x) + bias));
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
  dev_ctx.template Alloc<T>(out);
  if (x.numel() <= 0 || (!x.IsInitialized())) {
    return;
  }
  using AuxT =
      typename std::conditional<std::is_integral<T>::value, float, T>::type;
  phi::funcs::ElementwiseKernel<T>(
      dev_ctx,
      inputs,
      &outputs,
      ScaleFunctor<T, AuxT>(
          scale.to<AuxT>(), static_cast<AuxT>(bias), bias_after_scale));
}

}  // namespace phi

PD_REGISTER_KERNEL(scale,
                   GPU,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
