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

#include <algorithm>
#include <vector>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6

namespace phi {

template <typename Context, typename T, int Rank>
void ExpandAs(const Context& context,
              const DenseTensor& x,
              const std::vector<int>& target_shape,
              DenseTensor* out) {
  auto in_dims = x.dims();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = target_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());
  if (Rank == 0) {
    phi::Copy<Context>(context, x, context.GetPlace(), false, out);
    return;
  }
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(
        target_shape[i],
        0,
        errors::InvalidArgument("The value of target shape cannot be zero."));
    if (i < diff) {
      PADDLE_ENFORCE_GT(
          target_shape[i],
          0,
          errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_as_v2 op.",
              target_shape[i]));
      repeat_times[i] = target_shape[i];
    } else if (target_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            target_shape[i],
            errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_as_v2 op.",
                vec_in_dims[i],
                target_shape[i]));
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = target_shape[i];
      }
    } else {
      PADDLE_ENFORCE_EQ(
          target_shape[i],
          -1,
          errors::InvalidArgument(
              "When the value in shape is negative for expand_as_v2 op, "
              "only -1 is supported, but the value received is %d.",
              target_shape[i]));
      repeat_times[i] = 1;
    }
  }
  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }

  phi::DDim new_in_dims = phi::make_ddim(vec_in_dims);
  phi::DDim out_dims = phi::make_ddim(target_shape);

  out->Resize(out_dims);
  context.template Alloc<T>(out);
  auto x0 = EigenTensor<T, Rank>::From(x, new_in_dims);
  auto y = EigenTensor<T, Rank>::From(*out, out_dims);
  auto& place = *context.eigen_device();
  funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
      place, y, x0, bcast_dims);
}

template <typename T, typename Context>
void ExpandAsKernel(const Context& ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& y,
                    const std::vector<int>& target_shape,
                    DenseTensor* out) {
  auto rank = x.dims().size();
  auto target_rank = target_shape.size();
  PADDLE_ENFORCE_GE(target_rank,
                    rank,
                    errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be greater than or equal to "
                        "the rank (%d) of the input 'x'.",
                        target_rank,
                        rank));
  PADDLE_ENFORCE_GE(
      rank,
      0,
      errors::InvalidArgument("The rank (%d) of the input 'x' for "
                              "expand_as_v2 op must be positive.",
                              rank));
  PADDLE_ENFORCE_LE(target_rank,
                    MAX_RANK_SUPPORTED,
                    errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be less than or equal to %d.",
                        target_rank,
                        MAX_RANK_SUPPORTED));

  std::vector<int> real_target_shape = target_shape;
  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (target_shape[i] == -1) {
      if (y) {
        if (y->IsInitialized()) {
          real_target_shape = phi::vectorize<int>(y->dims());
        }
      }
      break;
    }
  }

  switch (target_rank) {
    case 0:
      ExpandAs<Context, T, 0>(ctx, x, real_target_shape, out);
      break;
    case 1:
      ExpandAs<Context, T, 1>(ctx, x, real_target_shape, out);
      break;
    case 2:
      ExpandAs<Context, T, 2>(ctx, x, real_target_shape, out);
      break;
    case 3:
      ExpandAs<Context, T, 3>(ctx, x, real_target_shape, out);
      break;
    case 4:
      ExpandAs<Context, T, 4>(ctx, x, real_target_shape, out);
      break;
    case 5:
      ExpandAs<Context, T, 5>(ctx, x, real_target_shape, out);
      break;
    case 6:
      ExpandAs<Context, T, 6>(ctx, x, real_target_shape, out);
      break;
  }
}

}  // namespace phi
