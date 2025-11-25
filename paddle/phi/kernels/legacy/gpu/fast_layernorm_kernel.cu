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
void LnFwdKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& scale,
                 const DenseTensor& bias,
                 float epsilon,
                 DenseTensor* y,
                 DenseTensor* mean,
                 DenseTensor* invvar) {
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;

  PD_CHECK(bias.type() == weight_type);

  auto sizes = x.dims();
  PD_CHECK(sizes.size() >= 2);

  const int cols = sizes[sizes.size() - 1];
  const int rows = x.numel() / cols;
  auto hidden_size = scale.numel();

  PD_CHECK(scale.dims() == bias.dims());
  PD_CHECK(hidden_size == cols);

  PD_CHECK(epsilon >= 0.f);

  auto place = x.place();
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<float>(mean);
  dev_ctx.template Alloc<float>(invvar);

  LaunchNormFwd<T, Context>(dev_ctx,
                            dev_ctx.stream(),
                            place,
                            /* x_ptr */ x.data(),
                            /* scale_ptr */ scale.data(),
                            /* bias_ptr */ bias.data(),
                            /* y_ptr */ y->data(),
                            /* mean_ptr */ mean->data(),
                            /* invvar_ptr */ invvar->data(),
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

PD_REGISTER_KERNEL(fast_ln,
                   GPU,
                   ALL_LAYOUT,
                   phi::LnFwdKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
