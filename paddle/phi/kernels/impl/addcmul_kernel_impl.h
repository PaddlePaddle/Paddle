// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <limits>
#include <type_traits>

#include "paddle/common/errors.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
inline typename dtype::MPTypeTrait<T>::Type GetAddcmulValue(
    const Scalar& value) {
  using MPType = typename dtype::MPTypeTrait<T>::Type;
  if constexpr (std::is_integral<T>::value) {
    const double numeric_value = value.to<double>();
    const double lowest = static_cast<double>(std::numeric_limits<T>::lowest());
    const double highest = static_cast<double>(std::numeric_limits<T>::max());
    const bool is_within_upper_bound =
        std::numeric_limits<T>::digits >= std::numeric_limits<double>::digits
            ? numeric_value < highest
            : numeric_value <= highest;
    const bool is_valid_integer = std::isfinite(numeric_value) &&
                                  std::trunc(numeric_value) == numeric_value &&
                                  numeric_value >= lowest &&
                                  is_within_upper_bound;
    PADDLE_ENFORCE_EQ(
        is_valid_integer,
        true,
        common::errors::InvalidArgument(
            "For integral input tensors, the argument value of addcmul must "
            "be an integer within the range of the input data type, but got "
            "%s.",
            value.ToRawString()));
  }
  return value.to<MPType>();
}

template <typename T, typename Context, size_t D>
static void AddcmulFunction(const Context& dev_ctx,
                            const DenseTensor& input,
                            const DenseTensor& tensor1,
                            const DenseTensor& tensor2,
                            typename dtype::MPTypeTrait<T>::Type value,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& out_dims = out->dims();
  auto input_dims = funcs::ExtendDims2Rank(input.dims(), D);
  auto t1_dims = funcs::ExtendDims2Rank(tensor1.dims(), D);
  auto t2_dims = funcs::ExtendDims2Rank(tensor2.dims(), D);

  Eigen::DSizes<int, D> input_bcast_dims;
  Eigen::DSizes<int, D> t1_bcast_dims;
  Eigen::DSizes<int, D> t2_bcast_dims;

  funcs::GetBroadcastDims<D>(input_dims, out_dims, &input_bcast_dims);
  funcs::GetBroadcastDims<D>(t1_dims, out_dims, &t1_bcast_dims);
  funcs::GetBroadcastDims<D>(t2_dims, out_dims, &t2_bcast_dims);

  auto eigen_input = phi::EigenTensor<T, D>::From(input, input_dims);
  auto eigen_t1 = phi::EigenTensor<T, D>::From(tensor1, t1_dims);
  auto eigen_t2 = phi::EigenTensor<T, D>::From(tensor2, t2_dims);
  auto eigen_out = phi::EigenTensor<T, D>::From(*out);

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();

  // out = input + value * t1 * t2
  eigen_out.device(place) =
      (eigen_input.broadcast(input_bcast_dims).template cast<MPType>() +
       static_cast<MPType>(value) *
           eigen_t1.broadcast(t1_bcast_dims).template cast<MPType>() *
           eigen_t2.broadcast(t2_bcast_dims).template cast<MPType>())
          .template cast<T>();
}

template <typename T, typename Context>
static void AddcmulFunctionZero(const Context& dev_ctx,
                                const DenseTensor& input,
                                const DenseTensor& tensor1,
                                const DenseTensor& tensor2,
                                typename dtype::MPTypeTrait<T>::Type value,
                                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto dim = common::make_ddim(std::vector<int64_t>(1, 1));
  auto eigen_input = phi::EigenTensor<T, 1>::From(input, dim);
  auto eigen_t1 = phi::EigenTensor<T, 1>::From(tensor1, dim);
  auto eigen_t2 = phi::EigenTensor<T, 1>::From(tensor2, dim);
  auto eigen_out = phi::EigenTensor<T, 1>::From(*out, dim);

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();

  eigen_out.device(place) =
      (eigen_input.template cast<MPType>() +
       static_cast<MPType>(value) * eigen_t1.template cast<MPType>() *
           eigen_t2.template cast<MPType>())
          .template cast<T>();
}

template <typename T, typename Context>
void AddcmulKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& tensor1,
                   const DenseTensor& tensor2,
                   const Scalar& value_scalar,
                   DenseTensor* out) {
  auto value = GetAddcmulValue<T>(value_scalar);
  int rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      common::errors::InvalidArgument(
          "The number of dimensions for AddcmulOp must be "
          "greater than or equal to 0, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      common::errors::InvalidArgument(
          "The number of dimensions for AddcmulOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));

  switch (rank) {
    case 0:
      AddcmulFunctionZero<T, Context>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 1:
      AddcmulFunction<T, Context, 1>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 2:
      AddcmulFunction<T, Context, 2>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 3:
      AddcmulFunction<T, Context, 3>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 4:
      AddcmulFunction<T, Context, 4>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 5:
      AddcmulFunction<T, Context, 5>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
    case 6:
      AddcmulFunction<T, Context, 6>(
          dev_ctx, input, tensor1, tensor2, value, out);
      break;
  }
}

}  // namespace phi
