// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
using EnableIfInteger =
    typename std::enable_if<std::is_integral<T>::value, int>::type;

template <typename T>
using EnableIfNonInteger =
    typename std::enable_if<!std::is_integral<T>::value, int>::type;

// Here if keepdim=True, this will fallback to a simplified version of
// take_along_axis. However, if keepdim=False (by default), indices will
// not have equal rank will the input values (and values_grad), therefore
// needs an unsqueeze operation by shallow copying indices and Resize
#define DEFINE_WITH_INDEX_GRAD_KERNEL(OpType)                                \
  template <typename T, typename Context, EnableIfNonInteger<T> = 0>         \
  void OpType##WithIndexGradKernel(const Context& dev_ctx,                   \
                                   const DenseTensor& x,                     \
                                   const DenseTensor& values,                \
                                   const DenseTensor& indices,               \
                                   const DenseTensor& values_grad,           \
                                   const Scalar& dim,                        \
                                   bool keepdim,                             \
                                   DenseTensor* x_grad) {                    \
    x_grad->Resize(x.dims());                                                \
    dev_ctx.template Alloc<T>(x_grad);                                       \
    if (x_grad->numel() == 0) {                                              \
      return;                                                                \
    }                                                                        \
    int64_t dim_val = dim.to<int64_t>();                                     \
    if (dim_val < 0) {                                                       \
      dim_val += x.dims().size();                                            \
    }                                                                        \
    DenseTensor shallow_copied_inds(indices);                                \
    if (!keepdim) {                                                          \
      auto indices_dim = x.dims();                                           \
      indices_dim[dim_val] = 1;                                              \
      shallow_copied_inds.Resize(indices_dim);                               \
    }                                                                        \
    phi::funcs::SetConstant<Context, T> functor;                             \
    functor(dev_ctx, x_grad, static_cast<T>(0));                             \
    phi::funcs::gpu_scatter_add_kernel<T, int64_t>(                          \
        *x_grad, dim_val, shallow_copied_inds, values_grad, true, dev_ctx);  \
  }                                                                          \
  template <typename T, typename Context, EnableIfInteger<T> = 0>            \
  void OpType##WithIndexGradKernel(const Context& dev_ctx,                   \
                                   const DenseTensor& x,                     \
                                   const DenseTensor& values,                \
                                   const DenseTensor& indices,               \
                                   const DenseTensor& values_grad,           \
                                   const Scalar& dim,                        \
                                   bool keepdim,                             \
                                   DenseTensor* x_grad) {                    \
    std::string dtype_name = phi::DataTypeToString(values.dtype());          \
    PADDLE_ENFORCE_EQ(                                                       \
        0,                                                                   \
        1,                                                                   \
        phi::errors::InvalidArgument(                                        \
            "Integer type '%s' is not allowed to have stop_gradient=False.", \
            dtype_name.c_str()));                                            \
  }

DEFINE_WITH_INDEX_GRAD_KERNEL(Max)
DEFINE_WITH_INDEX_GRAD_KERNEL(Min)

#undef DEFINE_WITH_INDEX_GRAD_KERNEL

}  // namespace phi

PD_REGISTER_KERNEL(max_with_index_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaxWithIndexGradKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(min_with_index_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MinWithIndexGradKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
