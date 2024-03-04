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

#pragma once

#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FCKernel(const Context& dev_ctx,
              const DenseTensor& input,
              const DenseTensor& w,
              const paddle::optional<DenseTensor>& bias,
              const int in_num_col_dims,
              const std::string& activation_type,
              const bool padding_weights,
              DenseTensor* out) {
  bool with_relu = (activation_type == "relu") ? true : false;

  auto w_dims = w.dims();

  std::vector<int64_t> output_dims;
  phi::funcs::FCOutputSize(
      input.dims(), w_dims, output_dims, in_num_col_dims, padding_weights);
  out->Resize(common::make_ddim(output_dims));
  out->set_lod(input.lod());

  auto out_dims = out->dims();
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  int M = common::product(out_dims) / w_dims1;

  const T* input_data = input.data<T>();
  const T* w_data = w.data<T>();
  auto* output_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx,
     M,
     w_dims1,
     w_dims0,
     input_data,
     w_data,
     output_data,
     bias ? bias->data<T>() : NULL,
     with_relu,
     padding_weights);
}
}  // namespace fusion
}  // namespace phi
