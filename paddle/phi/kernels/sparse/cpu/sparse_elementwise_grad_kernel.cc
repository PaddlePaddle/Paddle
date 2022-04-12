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

//#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"
//#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
//#include "paddle/phi/kernels/funcs/sparse/sparse_elementwise_base.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/sparse/copy_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

#define CSR_ELEMENTWISE_GRAD_API_NAME(name) sparse_elementwise_##name##_grad

/*template <typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type
ElementwiseAddCsrGrad(const CPUContext& ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& y,
                      const SparseCsrTensor& out,
                      const SparseCsrTensor& dout,
                      SparseCsrTensor* dx,
                      SparseCsrTensor* dy) {
  *//*  auto blas = phi::funcs::GetBlas<CPUContext, T>(ctx);
    if (dx) {
      blas.VCOPY(
          dout.numel(), dout.data<T>(), dx->mutable_data<T>(ctx.GetPlace()));
    }

    if (dy) {
      blas.VCOPY(
          dout.numel(), dout.data<T>(), dy->mutable_data<T>(ctx.GetPlace()));
    }*//*
}

template <typename T>
void AddGradFunc(const CPUContext& dev_ctx,
                 const SparseCsrTensor& x,
                 const SparseCsrTensor& y,
                 const SparseCsrTensor& out,
                 const SparseCsrTensor& dout,
                 SparseCsrTensor* dx,
                 SparseCsrTensor* dy) {
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    ElementwiseAddCsrGrad<T>(dev_ctx, x, y, out, dout, dx, dy);
  } else {
    *//*    ElemwiseExplicitGradCompute<T, IdentityGrad<T>, IdentityGrad<T>>(
            dev_ctx,
            x,
            y,
            out,
            dout,
            axis,
            dx,
            dy,
            IdentityGrad<T>(),
            IdentityGrad<T>());
      }*//*
  }
}*/

template <typename T, typename Context>
void ElementwiseAddCsrGradImpl(const Context& dev_ctx,
                               const SparseCsrTensor& x,
                               const SparseCsrTensor& y,
                               const SparseCsrTensor& out_grad,
                               SparseCsrTensor* x_grad,
                               SparseCsrTensor* y_grad) {
  //  // Special case when y_grad is not needed and x_grad doesn't reduce
  //  if (x_grad != nullptr && y_grad == nullptr &&
  //      x_grad->dims() == out_grad.dims()) {
  //    VLOG(4) << "Special case when y_grad is not needed and x_grad doesn't "
  //               "reduce";
  //    CopyCsr(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  //  } else if (x_grad == nullptr && y_grad != nullptr &&
  //             y_grad->dims() == out_grad.dims()) {
  //    VLOG(4) << "Special case when x_grad is not needed and y_grad doesn't "
  //               "reduce";
  //    CopyCsr(dev_ctx, out_grad, dev_ctx.GetPlace(), false, y_grad);
  //  } else {
  //    grad_func(dev_ctx, x, y, *out, out_grad dout, x_grad dx, y_grad dy);
  if (x_grad) {
    //      dev_ctx.template Alloc<T>(x_grad);
    //      blas.VCOPY(
    //          dout.numel(), dout.data<T>(),
    //          dx->mutable_data<T>(ctx.GetPlace()));
    CopyCsr(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  }

  if (y_grad) {
    //      dev_ctx.template Alloc<T>(y_grad);
    //      blas.VCOPY(
    //          dout.numel(), dout.data<T>(),
    //          dy->mutable_data<T>(ctx.GetPlace()));
    CopyCsr(dev_ctx, out_grad, dev_ctx.GetPlace(), false, y_grad);
  }
  //  }
}

template <typename T, typename Context>
void ElementWiseAddCsrGradKernel(const Context& dev_ctx,
                                 const SparseCsrTensor& x,
                                 const SparseCsrTensor& y,
                                 const SparseCsrTensor& dout,
                                 SparseCsrTensor* dx,
                                 SparseCsrTensor* dy) {
  //  ElementwiseAddCsrGradImpl<T,Context>(dev_ctx, x, y, dout, dx, dy);
  if (dx) {
    //      dev_ctx.template Alloc<T>(x_grad);
    //      blas.VCOPY(
    //          dout.numel(), dout.data<T>(),
    //          dx->mutable_data<T>(ctx.GetPlace()));
    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    //      dev_ctx.template Alloc<T>(y_grad);
    //      blas.VCOPY(
    //          dout.numel(), dout.data<T>(),
    //          dy->mutable_data<T>(ctx.GetPlace()));
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
    //    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    //    dout*x
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, x, dy);
    //    CopyCsr(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  }
}

template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  if (dx) {
    //    dout/y
    sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, dout, y, dx);
  }

  if (dy) {
    //    dout/x
    sparse::ElementWiseMultiplyCsrKernel<T, Context>(dev_ctx, dout, x, dy);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_GRAD_API_NAME(add),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrGradKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_GRAD_API_NAME(sub),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrGradKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_GRAD_API_NAME(mul),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrGradKernel,
                   float,
                   double) {}