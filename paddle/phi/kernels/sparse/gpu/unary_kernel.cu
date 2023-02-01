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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi {
namespace sparse {

template <typename T>
struct DivScalarFunctor {
  T value_;

  explicit DivScalarFunctor(T value) : value_(value) {}

  __device__ __forceinline__ T operator()(const T x) const {
    return x / value_;
  }
};

template <typename T, typename Context>
void DivScalarCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        float scalar,
                        SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);

  std::vector<const DenseTensor*> ins = {&(x.values())};
  std::vector<DenseTensor*> outs = {out->mutable_values()};
  DivScalarFunctor<T> func(static_cast<T>(scalar));
  funcs::ElementwiseKernel<T, DivScalarFunctor<T>>(dev_ctx, ins, &outs, func);
}

template <typename T, typename Context>
void DivScalarCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        float scalar,
                        SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);

  std::vector<const DenseTensor*> ins = {&(x.values())};
  std::vector<DenseTensor*> outs = {out->mutable_values()};
  DivScalarFunctor<T> func(static_cast<T>(scalar));
  funcs::ElementwiseKernel<T, DivScalarFunctor<T>>(dev_ctx, ins, &outs, func);
}

}  // namespace sparse
}  // namespace phi

#define PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(name, prefix)          \
  PD_REGISTER_KERNEL(name##_coo,                                   \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CooKernel,               \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_REGISTER_KERNEL(name##_csr,                                   \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CsrKernel,               \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sin, Sin)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(tan, Tan)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(asin, Asin)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(atan, Atan)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sinh, Sinh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(tanh, Tanh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(asinh, Asinh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(atanh, Atanh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sqrt, Sqrt)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(square, Square)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(log1p, Log1p)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(relu, Relu)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(abs, Abs)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(pow, Pow)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(scale, Scale)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(expm1, Expm1)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(relu6_raw, Relu6Raw)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(relu6, Relu6)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(leaky_relu, LeakyRelu)

PD_REGISTER_KERNEL(divide_scalar_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DivScalarCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_scalar_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DivScalarCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(cast_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(cast_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
