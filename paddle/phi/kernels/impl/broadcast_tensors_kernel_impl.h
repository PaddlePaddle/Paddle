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

#include "paddle/phi/kernels/broadcast_tensors_kernel.h"

#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define SWITCH_OUT_RANK_CASE(n)                                        \
  case n: {                                                            \
    ApplyBroadcast<T, Context, n>(ctx, in_tensors[i], out_tensors[i]); \
    break;                                                             \
  }

namespace phi {

template <typename T, typename Context, int OutRank>
void ApplyBroadcast(const Context& ctx,
                    const DenseTensor* input_tensor,
                    DenseTensor* output_tensor) {
  const auto& input_dims = input_tensor->dims();
  const auto& output_dims = output_tensor->dims();

  int in_rank = input_dims.size();
  int out_rank = output_dims.size();

  // 1. Collect bcast_dims, each element of which indicates how many
  // times we need to replicate along the corresponding dimension
  // 2. Collect new_input_dims_vec. Eigen::broadcast requires same rank for
  // both input and output tensors, so we need to initialize input X with
  // expanded dims: "new_input_dims_vec"
  Eigen::DSizes<Eigen::DenseIndex, OutRank> bcast_dims;
  std::vector<int64_t> new_input_dims_vec(out_rank);
  for (int j = 0; j < out_rank; j++) {
    int out_axis = out_rank - j - 1;
    int in_axis = in_rank - j - 1;

    bcast_dims[out_axis] = output_dims[out_axis];
    new_input_dims_vec[out_axis] = 1;
    if (in_axis >= 0 && input_dims[in_axis] == output_dims[out_axis]) {
      bcast_dims[out_axis] = 1;
      new_input_dims_vec[out_axis] = input_dims[in_axis];
    }
  }
  auto new_input_dims = phi::make_ddim(new_input_dims_vec);

  // Initialize input X with new_input_dims_vec, so it's rank-aligned with the
  // output
  auto x = EigenTensor<T, OutRank>::From(*input_tensor, new_input_dims);

  ctx.template Alloc<T>(output_tensor);
  auto y = EigenTensor<T, OutRank>::From(*output_tensor, output_dims);

  auto& place = *ctx.eigen_device();
  funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, OutRank>::Eval(
      place, y, x, bcast_dims);
}

template <typename T, typename Context>
void BroadcastTensorsKernel(const Context& ctx,
                            const std::vector<const DenseTensor*>& x,
                            std::vector<DenseTensor*> out) {
  const auto& in_tensors = x;
  auto out_tensors = out;
  size_t num_ins = in_tensors.size();

  PADDLE_ENFORCE_GT(
      num_ins,
      1,
      errors::InvalidArgument(
          "Expected at least 2 input tensors, but only received d%.",
          in_tensors.size()));

  PADDLE_ENFORCE_EQ(num_ins,
                    out_tensors.size(),
                    errors::InvalidArgument(
                        "BroadcastTensorsOp expects equal number of inputs and "
                        "outputs,but received: %d inputs v.s %d outputs",
                        num_ins,
                        out_tensors.size()));

  // Eigen has no support for dynamic ranked tensor
  // Thus we perform static expansion for each possible ranks
  for (size_t i = 0; i < num_ins; i++) {
    int out_rank = out_tensors[i]->dims().size();
    switch (out_rank) {
      SWITCH_OUT_RANK_CASE(1)
      SWITCH_OUT_RANK_CASE(2)
      SWITCH_OUT_RANK_CASE(3)
      SWITCH_OUT_RANK_CASE(4)
      SWITCH_OUT_RANK_CASE(5)
      default: {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Target tensor rank out of range"
            "Maximum supported rank for broadcast is: 5"));
      }
    }
  }
}

}  // namespace phi
