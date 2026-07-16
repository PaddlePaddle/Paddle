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

#include "paddle/phi/kernels/sparse/addmm_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/sparse/matmul_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
struct ScaleFunctor {
  ScaleFunctor(const T scale, T* data) : scale_(scale), data_(data) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    data_[idx] = data_[idx] * scale_;
  }

 private:
  const T scale_;
  T* data_;
};

template <typename T, typename Context>
void AddmmCooDenseGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const SparseCooTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& dout,
                             float alpha,
                             float beta,
                             DenseTensor* dinput,
                             SparseCooTensor* dx,
                             DenseTensor* dy) {
  if (dinput) {
    dinput->Resize(input.dims());
    dev_ctx.template Alloc<T>(dinput);

    memory_utils::Copy(dev_ctx.GetPlace(),
                       dinput->data<T>(),
                       dev_ctx.GetPlace(),
                       dout.data<T>(),
                       static_cast<size_t>(input.numel()) * sizeof(T),
                       dev_ctx.stream());
    funcs::ForRange<Context> for_range(dev_ctx, input.numel());
    ScaleFunctor<T> functor(static_cast<T>(beta), dinput->data<T>());
    for_range(functor);
  }
  DenseTensor dout_scale = EmptyLike<T, Context>(dev_ctx, dout);
  memory_utils::Copy(dev_ctx.GetPlace(),
                     dout_scale.data<T>(),
                     dev_ctx.GetPlace(),
                     dout.data<T>(),
                     static_cast<size_t>(dout.numel()) * sizeof(T),
                     dev_ctx.stream());
  funcs::ForRange<Context> for_range(dev_ctx, dout.numel());
  ScaleFunctor<T> functor(static_cast<T>(alpha), dout_scale.data<T>());
  for_range(functor);
  MatmulCooDenseGradKernel<T, Context>(dev_ctx, x, y, dout_scale, dx, dy);
}

// Backward of "DENSE + CSR @ DENSE -> DENSE"
template <typename T, typename Context>
void AddmmCsrDenseGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const SparseCsrTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& dout,
                             float alpha,
                             float beta,
                             DenseTensor* dinput,
                             SparseCsrTensor* dx,
                             DenseTensor* dy) {
  if (dinput) {
    dinput->Resize(input.dims());
    dev_ctx.template Alloc<T>(dinput);

    memory_utils::Copy(dev_ctx.GetPlace(),
                       dinput->data<T>(),
                       dev_ctx.GetPlace(),
                       dout.data<T>(),
                       static_cast<size_t>(input.numel()) * sizeof(T),
                       dev_ctx.stream());
    funcs::ForRange<Context> for_range(dev_ctx, input.numel());
    ScaleFunctor<T> functor(static_cast<T>(beta), dinput->data<T>());
    for_range(functor);
  }
  DenseTensor dout_scale = EmptyLike<T, Context>(dev_ctx, dout);
  memory_utils::Copy(dev_ctx.GetPlace(),
                     dout_scale.data<T>(),
                     dev_ctx.GetPlace(),
                     dout.data<T>(),
                     static_cast<size_t>(dout.numel()) * sizeof(T),
                     dev_ctx.stream());
  funcs::ForRange<Context> for_range(dev_ctx, dout.numel());
  ScaleFunctor<T> functor(static_cast<T>(alpha), dout_scale.data<T>());
  for_range(functor);
  MatmulCsrDenseGradKernel<T, Context>(dev_ctx, x, y, dout_scale, dx, dy);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(addmm_coo_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCooDenseGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(addmm_csr_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCsrDenseGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
