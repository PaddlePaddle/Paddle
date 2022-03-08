// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void ClipByNormKernel(const Context& ctx,
                      const DenseTensor& x_in,
                      float max_norm,
                      DenseTensor* out_p) {
  ctx.template Alloc<T>(out_p);
  auto x = EigenVector<T>::Flatten(x_in);
  auto out = EigenVector<T>::Flatten(*out_p);
  auto x_norm = x.square().sum().sqrt();
  auto& place = *ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<T>();
  auto epsilon = ((x_norm <= static_cast<T>(1e-30)).all().template cast<T>()) *
                 static_cast<T>(1e-6);

  auto scaling =
      temp + (static_cast<T>(1) - temp) * max_norm / (x_norm + epsilon);
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(x_in.numel());
  if (ctx.GetPlace() == phi::CPUPlace()) {
    out.device(place) = x * scaling.reshape(one_dim).eval().broadcast(m_dsize);
  } else {
    out.device(place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
  }
}

template <typename T, typename Context>
void ClipByNormSparseKernel(const Context& ctx,
                            const SelectedRows& x,
                            float max_norm,
                            SelectedRows* out) {
  // merge ids in selected rows first
  paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
  phi::SelectedRows merged_input;
  merge_func(ctx, x, &merged_input);
  auto input = merged_input.value();

  phi::SelectedRows* output_selected_rows = out;
  output_selected_rows->set_rows(merged_input.rows());
  output_selected_rows->set_height(merged_input.height());
  auto output = output_selected_rows->mutable_value();
  output->Resize(merged_input.value().dims());
  output->mutable_data<T>(ctx.GetPlace());

  ClipByNormKernel<T>(ctx, input, max_norm, output);
}

}  // namespace phi
