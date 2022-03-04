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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define SWITCH_RESHAPE_DIMS(n)                                                \
  case n: {                                                                   \
    Eigen::DSizes<Eigen::DenseIndex, n> reshape_dims;                         \
    for (size_t i = 0; i < reshape_dims_vec.size(); ++i) {                    \
      reshape_dims[i] = reshape_dims_vec[i];                                  \
    }                                                                         \
    dX.device(place) =                                                        \
        dOut.reshape(reshape_dims).sum(reduce_dims).reshape(dX.dimensions()); \
    break;                                                                    \
  }

#define UPPER_SWITCH_REDUCE_DIMS(m)                       \
  case m: {                                               \
    Eigen::DSizes<Eigen::DenseIndex, m> reduce_dims;      \
    for (size_t i = 0; i < reduce_dims_vec.size(); ++i) { \
      reduce_dims[i] = reduce_dims_vec[i];                \
    }                                                     \
    switch (reshape_size) {
#define LOWER_SWITCH_REDUCE_DIMS                             \
  default: {                                                 \
    PADDLE_THROW(errors::InvalidArgument(                    \
        "Detected reshape size: %d out of range"             \
        "Minimum value should be larger than reduce size %d" \
        "While maximum supported is: 5",                     \
        reshape_size,                                        \
        reduce_size));                                       \
  }                                                          \
    }                                                        \
    break;                                                   \
    }

namespace phi {

template <typename T, typename Context>
void BroadcastTensorsGradKernel(const Context& ctx,
                                const std::vector<const DenseTensor*>& dout,
                                std::vector<DenseTensor*> dx) {
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

  PADDLE_ENFORCE_EQ(num_ins,
                    out_tensors.size(),
                    errors::InvalidArgument(
                        "BroadcastTensorsOp expects equal number of inputs and "
                        "outputs, but received: %d inputs v.s %d outputs",
                        num_ins,
                        out_tensors.size()));

  // For each In-Out tensor pair,
  // Prepare and apply broadcast dims array
  for (size_t i = 0; i < num_ins; i++) {
    const auto* input_tensor = in_tensors[i];
    auto* output_tensor = out_tensors[i];

    const auto& input_dims = input_tensor->dims();
    const auto& output_dims = output_tensor->dims();

    int in_rank = input_dims.size();
    int out_rank = output_dims.size();

    // BroadcastTensorsGrad is simply a reduce_sum along broadcasted axes
    // Here we perform the following Eigen operations:
    // dOut(Flattened) -> reshape(reshape_dims) -> reduce(reduce_dims) ->
    // reshape(dX_shape) -> dX
    // Note the last "reshape(dX_shape)" will be performed implicitly,
    // and we only need to collect reduce_dims and reshape_dims
    std::vector<int> reduce_dims_vec;
    std::vector<int> reshape_dims_vec;
    for (int j = 0; j < in_rank; j++) {
      int out_axis = out_rank - j - 1;
      int in_axis = in_rank - j - 1;

      reshape_dims_vec.push_back(input_dims[j]);
      if (out_axis < 0 || output_dims[out_axis] != input_dims[in_axis]) {
        reduce_dims_vec.push_back(in_axis);
      }
    }

    size_t reduce_size = reduce_dims_vec.size();
    size_t reshape_size = reshape_dims_vec.size();
    bool just_copy = (reduce_dims_vec.size() == 0);
    ctx.template Alloc<T>(output_tensor);
    if (just_copy) {
      // If this turns out to be a No-Op, simply perform a tensor copy
      paddle::framework::TensorCopy(
          *input_tensor, ctx.GetPlace(), ctx, output_tensor);
    } else {
      PADDLE_ENFORCE_GE(
          reduce_dims_vec.size(),
          1,
          errors::InvalidArgument("The number of dimensions of the input "
                                  "'Out@GRAD' for Op(broadcast_tensors)"
                                  " must be greater than or equal to 1, but "
                                  "the value received is %d.",
                                  reduce_dims_vec.size()));
      PADDLE_ENFORCE_LE(
          reduce_dims_vec.size(),
          5,
          errors::InvalidArgument(
              "The number of dimensions of the input 'Out@GRAD' "
              "for Op(broadcast_tensors) must be less than or equal "
              "to 5, but the value received is %d.",
              reduce_dims_vec.size()));

      // Overall:
      // dOut(Flattened) -> reshape(reshape_dims) -> reduce(reduce_dims) ->
      // reshape(dX_shape) -> dX
      auto dX = EigenVector<T>::Flatten(*output_tensor);
      auto dOut = EigenVector<T>::Flatten(*input_tensor);
      auto& place = *ctx.eigen_device();

      // Expand ReduceSize and ReshapeSize into static values
      switch (reduce_size) {
        UPPER_SWITCH_REDUCE_DIMS(1)
        SWITCH_RESHAPE_DIMS(1)
        SWITCH_RESHAPE_DIMS(2)
        SWITCH_RESHAPE_DIMS(3)
        SWITCH_RESHAPE_DIMS(4)
        SWITCH_RESHAPE_DIMS(5)
        LOWER_SWITCH_REDUCE_DIMS

        UPPER_SWITCH_REDUCE_DIMS(2)
        SWITCH_RESHAPE_DIMS(2)
        SWITCH_RESHAPE_DIMS(3)
        SWITCH_RESHAPE_DIMS(4)
        SWITCH_RESHAPE_DIMS(5)
        LOWER_SWITCH_REDUCE_DIMS

        UPPER_SWITCH_REDUCE_DIMS(3)
        SWITCH_RESHAPE_DIMS(3)
        SWITCH_RESHAPE_DIMS(4)
        SWITCH_RESHAPE_DIMS(5)
        LOWER_SWITCH_REDUCE_DIMS

        UPPER_SWITCH_REDUCE_DIMS(4)
        SWITCH_RESHAPE_DIMS(4)
        SWITCH_RESHAPE_DIMS(5)
        LOWER_SWITCH_REDUCE_DIMS

        UPPER_SWITCH_REDUCE_DIMS(5)
        SWITCH_RESHAPE_DIMS(5)
        LOWER_SWITCH_REDUCE_DIMS

        default: {
          PADDLE_THROW(
              errors::InvalidArgument("Detected reduce size: %d out of range"
                                      "While maximum supported is: 5",
                                      reduce_size));
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(broadcast_tensors_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BroadcastTensorsGradKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16) {}
