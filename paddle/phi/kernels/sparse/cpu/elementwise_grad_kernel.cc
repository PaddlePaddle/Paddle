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

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/sparse/elementwise_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT, typename Context>
void AllocCsrPtr(const Context& dev_ctx,
                 const SparseCsrTensor& x,
                 SparseCsrTensor* dx) {
  DenseTensor dx_crows = phi::EmptyLike<IntT>(dev_ctx, x.crows());
  DenseTensor dx_cols = phi::EmptyLike<IntT>(dev_ctx, x.cols());
  DenseTensor dx_values = phi::EmptyLike<T>(dev_ctx, x.values());
  dx->SetMember(dx_crows, dx_cols, dx_values, x.dims());
}

template <typename T, typename IntT, typename Context>
void AllocCooPtr(const Context& dev_ctx,
                 const SparseCooTensor& x,
                 SparseCooTensor* dx) {
  DenseTensor dx_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
  DenseTensor dx_values = phi::EmptyLike<T>(dev_ctx, x.values());
  dx->SetMember(dx_indices, dx_values, x.dims(), true);
}

template <typename T, typename IntT, typename Context>
void ElementWiseAddCsrGradCPUKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  // Special case when y_grad is not needed
  if (dx != nullptr && dy == nullptr) {
    VLOG(4) << "Special case when dy is not needed";
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  } else if (dx == nullptr && dy != nullptr) {
    VLOG(4) << "Special case when dx is not needed";
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  } else {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseSubtractCsrGradCPUKernel(const Context& dev_ctx,
                                         const SparseCsrTensor& x,
                                         const SparseCsrTensor& y,
                                         const SparseCsrTensor& dout,
                                         SparseCsrTensor* dx,
                                         SparseCsrTensor* dy) {
  if (dx) {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseMultiplyCsrGradCPUKernel(const Context& dev_ctx,
                                         const SparseCsrTensor& x,
                                         const SparseCsrTensor& y,
                                         const SparseCsrTensor& dout,
                                         SparseCsrTensor* dx,
                                         SparseCsrTensor* dy) {
  if (dx) {
    //    dout*y
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    dout*x
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, x, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseDivideCsrGradCPUKernel(const Context& dev_ctx,
                                       const SparseCsrTensor& x,
                                       const SparseCsrTensor& y,
                                       const SparseCsrTensor& out,
                                       const SparseCsrTensor& dout,
                                       SparseCsrTensor* dx,
                                       SparseCsrTensor* dy) {
  if (dx) {
    //    dout/y
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    -dout * out / y
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
    auto tmp = sparse::ElementWiseMultiplyCsr<T, Context>(dev_ctx, *dy, out);
    sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, tmp, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseAddCooGradCPUKernel(const Context& dev_ctx,
                                    const SparseCooTensor& x,
                                    const SparseCooTensor& y,
                                    const SparseCooTensor& dout,
                                    SparseCooTensor* dx,
                                    SparseCooTensor* dy) {
  //     Special case when y_grad is not needed*/
  if (dx != nullptr && dy == nullptr) {
    VLOG(4) << "Special case when dy is not needed";
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  } else if (dx == nullptr && dy != nullptr) {
    VLOG(4) << "Special case when dx is not needed";
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  } else {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseSubtractCooGradCPUKernel(const Context& dev_ctx,
                                         const SparseCooTensor& x,
                                         const SparseCooTensor& y,
                                         const SparseCooTensor& dout,
                                         SparseCooTensor* dx,
                                         SparseCooTensor* dy) {
  if (dx) {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseMultiplyCooGradCPUKernel(const Context& dev_ctx,
                                         const SparseCooTensor& x,
                                         const SparseCooTensor& y,
                                         const SparseCooTensor& dout,
                                         SparseCooTensor* dx,
                                         SparseCooTensor* dy) {
  if (dx) {
    //    dout*y
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    sparse::ElementWiseMultiplyCooKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    dout*x
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    sparse::ElementWiseMultiplyCooKernel<T, Context>(dev_ctx, dout, x, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseDivideCooGradCPUKernel(const Context& dev_ctx,
                                       const SparseCooTensor& x,
                                       const SparseCooTensor& y,
                                       const SparseCooTensor& out,
                                       const SparseCooTensor& dout,
                                       SparseCooTensor* dx,
                                       SparseCooTensor* dy) {
  if (dx) {
    //    dout/y
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    sparse::ElementWiseDivideCooKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    -dout * out / y
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
    auto tmp = sparse::ElementWiseMultiplyCoo<T, Context>(dev_ctx, *dy, out);
    sparse::ElementWiseDivideCooKernel<T, Context>(dev_ctx, tmp, y, dy);
  }
}
// CPU Kernel end

// Kernel
template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& out,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.crows().dtype(), "ElementWiseDivideCsrGradCPUKernel", ([&] {
        ElementWiseDivideCsrGradCPUKernel<T, data_t>(
            dev_ctx, x, y, out, dout, dx, dy);
      }));
}
template <typename T, typename Context>
void ElementWiseDivideCooGradKernel(const Context& dev_ctx,
                                    const SparseCooTensor& x,
                                    const SparseCooTensor& y,
                                    const SparseCooTensor& out,
                                    const SparseCooTensor& dout,
                                    SparseCooTensor* dx,
                                    SparseCooTensor* dy) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "ElementWiseDivideCooGradCPUKernel", ([&] {
        ElementWiseDivideCooGradCPUKernel<T, data_t>(
            dev_ctx, x, y, out, dout, dx, dy);
      }));
}

#define DEFINE_ELEMENTWISE_GRAD_KERNEL(name) \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_CSR(name)   \
                                             \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_COO(name)

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_CSR(name)                         \
  template <typename T, typename Context>                                \
  void ElementWise##name##CsrGradKernel(const Context& dev_ctx,          \
                                        const SparseCsrTensor& x,        \
                                        const SparseCsrTensor& y,        \
                                        const SparseCsrTensor& dout,     \
                                        SparseCsrTensor* dx,             \
                                        SparseCsrTensor* dy) {           \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                        \
        x.crows().dtype(), "ElementWise##name##CsrGradCPUKernel", ([&] { \
          ElementWise##name##CsrGradCPUKernel<T, data_t>(                \
              dev_ctx, x, y, dout, dx, dy);                              \
        }));                                                             \
  }

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_COO(name)                           \
  template <typename T, typename Context>                                  \
  void ElementWise##name##CooGradKernel(const Context& dev_ctx,            \
                                        const SparseCooTensor& x,          \
                                        const SparseCooTensor& y,          \
                                        const SparseCooTensor& dout,       \
                                        SparseCooTensor* dx,               \
                                        SparseCooTensor* dy) {             \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                          \
        x.indices().dtype(), "ElementWise##name##CooGradCPUKernel", ([&] { \
          ElementWise##name##CooGradCPUKernel<T, data_t>(                  \
              dev_ctx, x, y, dout, dx, dy);                                \
        }));                                                               \
  }

DEFINE_ELEMENTWISE_GRAD_KERNEL(Add)
DEFINE_ELEMENTWISE_GRAD_KERNEL(Subtract)
DEFINE_ELEMENTWISE_GRAD_KERNEL(Multiply)

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(add_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(subtract_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(multiply_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(divide_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(3).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(add_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(subtract_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(multiply_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(3).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
