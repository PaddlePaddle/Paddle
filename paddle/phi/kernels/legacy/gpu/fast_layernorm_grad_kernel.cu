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
void LnBwdKernel(const Context &dev_ctx,
                 const DenseTensor &x,
                 const DenseTensor &scale,
                 const DenseTensor &mean,
                 const DenseTensor &invvar,
                 const DenseTensor &dy,
                 const float epsilon,
                 DenseTensor *dx,
                 DenseTensor *dscale,
                 DenseTensor *dbias) {
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;

  PD_CHECK(dy.dtype() == output_type);
  PD_CHECK(mean.dtype() == compute_type);
  PD_CHECK(invvar.dtype() == compute_type);

  PD_CHECK(!x.is_cpu());
  PD_CHECK(!dy.is_cpu());
  PD_CHECK(!mean.is_cpu());
  PD_CHECK(!invvar.is_cpu());
  PD_CHECK(!scale.is_cpu());

  auto sizes = x.shape();
  PD_CHECK(sizes.size() >= 2);
  PD_CHECK(dy.shape() == sizes);

  int64_t rows = 1;
  for (size_t i = 0; i + 1 < sizes.size(); ++i) {
    rows *= sizes[i];
  }
  auto cols = sizes[sizes.size() - 1];

  auto hidden_size = scale.numel();

  PD_CHECK(mean.numel() == rows);
  PD_CHECK(mean.shape() == invvar.shape());

  PD_CHECK(scale.numel() == cols);

  auto dx = paddle::empty_like(x);
  auto dscale = paddle::empty_like(scale);
  auto dbias = paddle::empty_like(scale);

  auto place = x.place();

  LaunchNormBwd(x.stream(),
                place,
                /* x_ptr */ x.data(),
                /* scale_ptr */ scale.data(),
                /* mean_ptr */ mean.data(),
                /* invvar_ptr */ invvar.data(),
                /* dy_ptr */ dy.data(),
                /* dx_ptr */ dx.data(),
                /* dscale_ptr */ dscale.data(),
                /* dbias_ptr */ dbias.data(),
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

PD_REGISTER_KERNEL(fast_ln_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LnBwdKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
