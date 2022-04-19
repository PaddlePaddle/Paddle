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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/sparse/copy_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ElementWiseAddCsrGradKernel(const Context& dev_ctx,
                                 const SparseCsrTensor& x,
                                 const SparseCsrTensor& y,
                                 const SparseCsrTensor& dout,
                                 SparseCsrTensor* dx,
                                 SparseCsrTensor* dy) {
  // Special case when y_grad is not needed
  if (dx != nullptr && dy == nullptr) {
    VLOG(4) << "Special case when dy is not needed";
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  } else if (dx == nullptr && dy != nullptr) {
    VLOG(4) << "Special case when dx is not needed";
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  } else {
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  }
}

template <typename T, typename Context>
void ElementWiseSubtractCsrGradKernel(const Context& dev_ctx,
                                      const SparseCsrTensor& x,
                                      const SparseCsrTensor& y,
                                      const SparseCsrTensor& dout,
                                      SparseCsrTensor* dx,
                                      SparseCsrTensor* dy) {
  if (dx) {
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::OppositeKernel<T, Context>(
        dev_ctx, dout.non_zero_elements(), dy->mutable_non_zero_elements());
  }
}

template <typename T, typename Context>
void ElementWiseMultiplyCsrGradKernel(const Context& dev_ctx,
                                      const SparseCsrTensor& x,
                                      const SparseCsrTensor& y,
                                      const SparseCsrTensor& dout,
                                      SparseCsrTensor* dx,
                                      SparseCsrTensor* dy) {
  if (dx) {
    //    dout*y
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    dout*x
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, x, dy);
  }
}

template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& out,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  if (dx) {
    //    dout/y
    sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    -dout * out / y
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::OppositeKernel<T, Context>(
        dev_ctx, dout.non_zero_elements(), dy->mutable_non_zero_elements());
    auto tmp = sparse::ElementWiseMultiplyCsr<T, Context>(dev_ctx, *dy, out);
    sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, tmp, y, dy);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_elementwise_add_grad_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_sub_grad_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_mul_grad_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_div_grad_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
