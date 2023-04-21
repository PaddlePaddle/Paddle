// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lerp_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct LerpElementWiseDirectCUDAFunctor {
  HOSTDEVICE inline T operator()(const T x, const T y, const T weight) const {
    return x + weight * (y - x);
  }
};

template <typename T>
struct LerpScalarDirectCUDAFunctor {
  const T *weight_;

  HOSTDEVICE inline LerpScalarDirectCUDAFunctor(const T *weight)
      : weight_(weight) {}

  HOSTDEVICE inline T operator()(const T x, const T y) const {
    return x + weight_[0] * (y - x);
  }
};

template <typename Context, typename T>
static void LerpFunction(const Context &ctx,
                         const DenseTensor &x,
                         const DenseTensor &y,
                         const DenseTensor &weight,
                         DenseTensor *out) {
  std::vector<DenseTensor *> outputs;
  outputs.reserve(1);
  outputs.emplace_back(out);
  ctx.template Alloc<T>(out);

  std::vector<const DenseTensor *> inputs;

  if (weight.dims().size() == 0) {
    const T *weight_ptr = weight.data<T>();
    inputs.reserve(2);
    inputs.emplace_back(&x);
    inputs.emplace_back(&y);
    auto functor = LerpScalarDirectCUDAFunctor<T>(weight_ptr);
    funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
        ctx, inputs, &outputs, 1, functor);
  } else {
    inputs.reserve(3);
    inputs.emplace_back(&x);
    inputs.emplace_back(&y);
    inputs.emplace_back(&weight);
    auto functor = LerpElementWiseDirectCUDAFunctor<T>();
    funcs::BroadcastKernel<ElementwiseType::kTernary, T, T>(
        ctx, inputs, &outputs, 1, functor);
  }
}

template <typename Context, typename T>
static void LerpFunctionZero(const Context &ctx,
                             const DenseTensor &x,
                             const DenseTensor &y,
                             const DenseTensor &weight,
                             DenseTensor *out) {
  ctx.template Alloc<T>(out);

  auto dim = make_ddim(std::vector<int64_t>(1, 1));
  auto eigen_x = phi::EigenTensor<T, 1>::From(x, dim);
  auto eigen_y = phi::EigenTensor<T, 1>::From(y, dim);
  auto eigen_w = phi::EigenTensor<T, 1>::From(weight, dim);
  auto eigen_out = phi::EigenTensor<T, 1>::From(*out, dim);

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto &place = *ctx.eigen_device();
  eigen_out.device(place) =
      (eigen_x.template cast<MPType>() +
       eigen_w.template cast<MPType>() *
           (eigen_y.template cast<MPType>() - eigen_x.template cast<MPType>()))
          .template cast<T>();
}

template <typename T, typename Context>
void LerpKernel(const Context &ctx,
                const DenseTensor &x,
                const DenseTensor &y,
                const DenseTensor &weight,
                DenseTensor *out) {
  int rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "greater than or equal to 0, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));

  if (rank == 0) {
    LerpFunctionZero<Context, T>(ctx, x, y, weight, out);
  } else {
    LerpFunction<Context, T>(ctx, x, y, weight, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(lerp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LerpKernel,
                   phi::dtype::float16,
                   float,
                   double) {}
