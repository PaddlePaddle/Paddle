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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include "ln.h"  // NOLINT
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void RMSLnBwdKernel(const Context &dev_ctx,
                    const DenseTensor &x,
                    const DenseTensor &scale,
                    const DenseTensor &invvar,
                    const DenseTensor &y_grad,
                    float epsilon,
                    DenseTensor *x_grad,
                    DenseTensor *scale_grad) {
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;

  PD_CHECK(y_grad.dtype() == output_type);

  auto sizes = x.dims();
  PD_CHECK(sizes.size() >= 2);
  PD_CHECK(y_grad.dims() == sizes);

  int64_t rows = 1;
  for (size_t i = 0; i + 1 < sizes.size(); ++i) {
    rows *= sizes[i];
  }
  auto cols = sizes[sizes.size() - 1];

  auto hidden_size = scale.numel();

  PD_CHECK(scale.numel() == cols);

  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(scale_grad);

  auto place = x.place();

  LaunchNormBwd<T, Context>(dev_ctx,
                            dev_ctx.stream(),
                            place,
                            /* x_ptr */ x.data(),
                            /* scale_ptr */ scale.data(),
                            /* mean_ptr */ nullptr,
                            /* invvar_ptr */ invvar.data(),
                            /* y_grad_ptr */ y_grad.data(),
                            /* x_grad_ptr */ x_grad->data(),
                            /* scale_grad_ptr */ scale_grad->data(),
                            /* dbias_ptr */ nullptr,
                            weight_type,
                            input_type,
                            output_type,
                            compute_type,
                            hidden_size,
                            rows,
                            cols,
                            epsilon);
}

}  // namespace phi
PD_REGISTER_KERNEL(fast_rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSLnBwdKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
