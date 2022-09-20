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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void DivCooScalarKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        float scalar,
                        SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);

  auto eigen_out =
      phi::EigenVector<T>::Flatten(*(out->mutable_non_zero_elements()));
  auto eigen_x = phi::EigenVector<T>::Flatten(x.non_zero_elements());
  auto& dev = *dev_ctx.eigen_device();

  phi::funcs::EigenDiv<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x, static_cast<T>(scalar));
}

template <typename T, typename Context>
void DivCsrScalarKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        float scalar,
                        SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);

  auto eigen_out =
      phi::EigenVector<T>::Flatten(*(out->mutable_non_zero_elements()));
  auto eigen_x = phi::EigenVector<T>::Flatten(x.non_zero_elements());
  auto& dev = *dev_ctx.eigen_device();

  phi::funcs::EigenDiv<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x, static_cast<T>(scalar));
}

}  // namespace sparse
}  // namespace phi

#define PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(name, prefix)          \
  PD_REGISTER_KERNEL(name##_coo,                                   \
                     CPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CooKernel,               \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_REGISTER_KERNEL(name##_csr,                                   \
                     CPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CsrKernel,               \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sin, Sin)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(tan, Tan)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(asin, Asin)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(atan, Atan)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sinh, Sinh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(tanh, Tanh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(asinh, Asinh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(atanh, Atanh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sqrt, Sqrt)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(square, Square)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(log1p, Log1p)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(relu, Relu)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(abs, Abs)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(pow, Pow)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(scale, Scale)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(expm1, Expm1)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(relu6, Relu6)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(leaky_relu, LeakyRelu)

PD_REGISTER_KERNEL(divide_coo_scalar,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCooScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_csr_scalar,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCsrScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(cast_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(cast_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
