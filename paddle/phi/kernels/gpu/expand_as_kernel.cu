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

#include "paddle/phi/kernels/expand_as_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T, typename Context>
void ExpandAsKernel(const Context& ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& y,
                    const std::vector<int>& target_shape_t,
                    DenseTensor* out) {
  std::vector<int> target_shape = target_shape_t;

  if (y.get_ptr()) {
    target_shape = phi::vectorize<int>(y.get_ptr()->dims());
  }

  int rank = x.dims().size();
  int target_rank = static_cast<int>(target_shape.size());
  auto vec_in_dims = common::vectorize<int>(x.dims());

  unsigned int diff = target_rank - rank;
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);

  for (unsigned int i = 0; i < vec_in_dims.size(); ++i) {
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
      }
    } else {
      PADDLE_ENFORCE_EQ(
          target_shape[i],
          -1,
          errors::InvalidArgument(
              "When the value in shape is negative for expand_as_v2 op, "
              "only -1 is supported, but the value received is %d.",
              target_shape[i]));
    }
  }

  ExpandKernel<T, Context>(ctx, x, target_shape, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(expand_as,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandAsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16) {}
