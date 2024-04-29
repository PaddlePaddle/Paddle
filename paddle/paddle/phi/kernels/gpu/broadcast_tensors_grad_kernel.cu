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

#include "paddle/phi/kernels/broadcast_tensors_grad_kernel.h"

#include <vector>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void BroadcastTensorsGradKernel(const Context& ctx,
                                const std::vector<const DenseTensor*>& inputs,
                                const std::vector<const DenseTensor*>& dout,
                                std::vector<DenseTensor*> dx) {
  (void)inputs;
  // Find reduce dimensions
  const auto& in_tensors = dout;
  auto& out_tensors = dx;

  size_t num_ins = in_tensors.size();

  PADDLE_ENFORCE_GT(
      num_ins,
      1,
      errors::InvalidArgument(
          "Expected at least 2 input tensors, but only received d%.",
          in_tensors.size()));

  PADDLE_ENFORCE_EQ(
      num_ins,
      out_tensors.size(),
      errors::InvalidArgument(
          "BroadcastTensorsOp expects equal number of inputs and outputs,"
          "but received: %d inputs v.s %d outputs",
          num_ins,
          out_tensors.size()));

  // For each In-Out tensor pair,
  // Prepare and apply broadcast dims array
  for (size_t i = 0; i < num_ins; i++) {
    auto* input_tensor = in_tensors[i];
    auto* output_tensor = out_tensors[i];

    const DDim& input_dims = input_tensor->dims();
    const DDim& output_dims = output_tensor->dims();

    int in_rank = input_dims.size();
    int out_rank = output_dims.size();

    // Collect reduce_dims
    // Example:
    // dX  = [1,1,1,1]
    // dOut = [1,1,1,4]
    //
    // reduce_dims  = [3] // reduce along the broadcasted axis
    std::vector<int> reduce_dims_vec;
    for (int j = 0; j < in_rank; j++) {
      int out_axis = out_rank - j - 1;
      int in_axis = in_rank - j - 1;

      if (out_axis < 0 || output_dims[out_axis] != input_dims[in_axis]) {
        reduce_dims_vec.push_back(in_axis);
      }
    }

    bool just_copy = (reduce_dims_vec.size() == 0);
    ctx.template Alloc<T>(output_tensor);
    if (just_copy) {
      // Turns out to be a No-Op, simply copy tensors
      phi::Copy(ctx, *input_tensor, ctx.GetPlace(), false, output_tensor);
    } else {
      // reduce_sum implementation on CUDA
      phi::SumKernel<T, Context>(ctx,
                                 *input_tensor,
                                 reduce_dims_vec,
                                 output_tensor->dtype(),
                                 false,
                                 output_tensor);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(broadcast_tensors_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BroadcastTensorsGradKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
