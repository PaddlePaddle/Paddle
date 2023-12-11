// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/skip_layernorm_functor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SkipLayerNormKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor &y,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         const float epsilon,
                         const int begin_norm_axis,
                         DenseTensor *out) {
  auto *X_d = x.data<T>();
  auto *Y_d = y.data<T>();
  auto *scale_d = scale.data<T>();
  auto *bias_d = bias.data<T>();

  out->Resize(x.dims());
  auto *output_d = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

  size_t num = 1;
  for (size_t i = 0; i < x.dims().size(); i++) {
    num *= x.dims()[i];
  }
  int hidden = x.dims()[2];
  phi::funcs::SkipLayerNormFunctor<T> skip_layer_norm_func;

  if (std::is_same<T, phi::dtype::float16>::value) {
    const half *X_new = reinterpret_cast<const half *>(X_d);
    const half *Y_new = reinterpret_cast<const half *>(Y_d);
    const half *scale_new = reinterpret_cast<const half *>(scale_d);
    const half *bias_new = reinterpret_cast<const half *>(bias_d);
    half *output_new = reinterpret_cast<half *>(output_d);
    phi::funcs::SkipLayerNormFunctor<half> skip_layer_norm_func;
    skip_layer_norm_func(num,
                         hidden,
                         X_new,
                         Y_new,
                         scale_new,
                         bias_new,
                         output_new,
                         epsilon,
                         dev_ctx.stream());
  } else {
    phi::funcs::SkipLayerNormFunctor<T> skip_layer_norm_func;
    skip_layer_norm_func(num,
                         hidden,
                         X_d,
                         Y_d,
                         scale_d,
                         bias_d,
                         output_d,
                         epsilon,
                         dev_ctx.stream());
  }
}

}  // namespace fusion
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
PD_REGISTER_KERNEL(skip_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::SkipLayerNormKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(
    skip_layernorm, GPU, ALL_LAYOUT, phi::fusion::SkipLayerNormKernel, float) {}
#endif
