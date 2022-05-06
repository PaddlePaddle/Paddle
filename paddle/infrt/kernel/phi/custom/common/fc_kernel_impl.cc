
/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/infrt/kernel/phi/custom/common/fc_kernel_impl.h"

#include "paddle/phi/core/ddim.h"

void infrt::FcInferMeta(const phi::MetaTensor& input,
                        const phi::MetaTensor& weight,
                        const phi::MetaTensor& > bias,
                        int in_num_col_dims,
                        phi::MetaTensor* out) {
  auto w_dims = weight.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Weight of fc is expected to be a 2-D tensor. "
          "But received the number of Weight's dimensions is %d, "
          "Weight's shape is %s.",
          w_dims.size(),
          w_dims));

  auto bias_dims = bias.dims();
  auto w_dims1 = w_dims[1];

  PADDLE_ENFORCE_LE(
      bias_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Bias of fc is expected to be a 1-D or 2-D tensor. But "
          "received the number of Bias's dimensions is %d, "
          "Bias's shape is %s.",
          bias_dims.size(),
          bias_dims));

  PADDLE_ENFORCE_EQ(
      bias_dims[bias_dims.size() - 1],
      w_dims1,
      phi::errors::InvalidArgument(
          "The last dimension of input Bias is expected be equal "
          "to the actual width of input Weight. But received the last "
          "dimension of Bias is %d, Bias's shape is %s; "
          "the actual width of Weight is %d, Weight's shape is %s.",
          bias_dims[bias_dims.size() - 1],
          bias_dims,
          w_dims1,
          w_dims));

  if (bias_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        bias_dims[0],
        1,
        phi::errors::InvalidArgument(
            "The first dimension of input Bias is expected to be 1, "
            "but received %d, Bias's shape is %s.",
            bias_dims[0],
            bias_dims));
  }

  auto in_dims = input.dims();
  PADDLE_ENFORCE_LT(
      in_num_col_dims,
      in_dims.size(),
      phi::errors::InvalidArgument(
          "The attribute in_num_col_dims used to flatten Input to "
          "a 2-D tensor, is expected to be less than the number of "
          "Input's dimensions. But recieved in_num_col_dims is %d, "
          "the number of Input's dimensions is %d, Input's shape is %s.",
          in_num_col_dims,
          in_dims.size(),
          in_dims));

  auto in_mat_dims = phi::flatten_to_2d(in_dims, in_num_col_dims);
  auto w_dims0 = w_dims[0];
  auto w_dims1 = w_dims[1];
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1],
      w_dims0,
      phi::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But recieved input's second dimension is "
          "%d, input's shape is %s; weight's first dimension is %d, weight's "
          "shape is %s.",
          in_mat_dims[1],
          in_mat_dims,
          w_dims0,
          phi::make_ddim({w_dims0, w_dims1})));
  std::vector<int64_t> out_dims;
  out_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dims.push_back(in_dims[i]);
  }
  out_dims.push_back(w_dims1);
  out->set_dims(phi::make_ddim(out_dims));
  out->set_dtype(input.dtype());
  out->share_lod(input);
}
