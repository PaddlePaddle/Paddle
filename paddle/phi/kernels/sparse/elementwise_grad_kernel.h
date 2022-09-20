/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD(name)          \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD_WITH_TYPE(name, Csr) \
                                                           \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD_WITH_TYPE(name, Coo)

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC(name)          \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC_WITH_TYPE(name, Csr) \
                                                           \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC_WITH_TYPE(name, Coo)

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD_WITH_TYPE(name, type)            \
  template <typename T, typename Context>                                    \
  void ElementWise##name##type##GradKernel(const Context& dev_ctx,           \
                                           const Sparse##type##Tensor& x,    \
                                           const Sparse##type##Tensor& y,    \
                                           const Sparse##type##Tensor& dout, \
                                           Sparse##type##Tensor* dx,         \
                                           Sparse##type##Tensor* dy);

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC_WITH_TYPE(name, type)  \
  template <typename T, typename Context>                          \
  std::vector<Sparse##type##Tensor> ElementWise##name##type##Grad( \
      const Context& dev_ctx,                                      \
      const Sparse##type##Tensor& x,                               \
      const Sparse##type##Tensor& y,                               \
      const Sparse##type##Tensor& dout) {                          \
    Sparse##type##Tensor dx;                                       \
    Sparse##type##Tensor dy;                                       \
    MetaTensor meta_dx(&dx), meta_dy(&dy);                         \
    phi::UnchangedInferMeta(x, &meta_dx);                          \
    phi::UnchangedInferMeta(y, &meta_dy);                          \
    ElementWise##name##type##GradKernel<T, Context>(               \
        dev_ctx, x, y, dout, &dx, &dy);                            \
    return std::vector<Sparse##type##Tensor>{dx, dy};              \
  }

DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD(Add)
DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD(Subtract)
DEFINE_ELEMENTWISE_GRAD_KERNEL_HEAD(Multiply)

DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC(Add)
DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC(Subtract)
DEFINE_ELEMENTWISE_GRAD_KERNEL_FUNC(Multiply)

template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& out,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy);

template <typename T, typename Context>
void ElementWiseDivideCooGradKernel(const Context& dev_ctx,
                                    const SparseCooTensor& x,
                                    const SparseCooTensor& y,
                                    const SparseCooTensor& out,
                                    const SparseCooTensor& dout,
                                    SparseCooTensor* dx,
                                    SparseCooTensor* dy);

template <typename T, typename Context>
std::vector<SparseCsrTensor> ElementWiseDivideCsrGrad(
    const Context& dev_ctx,
    const SparseCsrTensor& x,
    const SparseCsrTensor& y,
    const SparseCsrTensor& out,
    const SparseCsrTensor& dout) {
  SparseCsrTensor dx;
  SparseCsrTensor dy;
  MetaTensor meta_dx(&dx), meta_dy(&dy);
  phi::UnchangedInferMeta(x, &meta_dx);
  phi::UnchangedInferMeta(y, &meta_dy);
  ElementWiseDivideCsrGradKernel<T, Context>(
      dev_ctx, x, y, out, dout, &dx, &dy);
  return std::vector<SparseCsrTensor>{dx, dy};
}

template <typename T, typename Context>
std::vector<SparseCooTensor> ElementWiseDivideCooGrad(
    const Context& dev_ctx,
    const SparseCooTensor& x,
    const SparseCooTensor& y,
    const SparseCooTensor& out,
    const SparseCooTensor& dout) {
  SparseCooTensor dx;
  SparseCooTensor dy;
  MetaTensor meta_dx(&dx), meta_dy(&dy);
  phi::UnchangedInferMeta(x, &meta_dx);
  phi::UnchangedInferMeta(y, &meta_dy);
  ElementWiseDivideCooGradKernel<T, Context>(
      dev_ctx, x, y, out, dout, &dx, &dy);
  return std::vector<SparseCooTensor>{dx, dy};
}

}  // namespace sparse
}  // namespace phi
