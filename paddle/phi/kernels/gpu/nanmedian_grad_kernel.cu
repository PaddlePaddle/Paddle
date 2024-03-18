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

#include "paddle/phi/kernels/nanmedian_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/nanmedian_utils.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;
inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelNanmedianMeanGrad(const int64_t* medians_ptr,
                                        const T* out_grad_ptr,
                                        T* dx_data,
                                        int64_t stride,
                                        int64_t pre_dim) {
  CUDA_KERNEL_LOOP(index, pre_dim) {
    int64_t offset = index * stride;

    if (medians_ptr[2 * index] >= 0) {
      if (medians_ptr[2 * index] == medians_ptr[2 * index + 1]) {
        dx_data[offset + medians_ptr[2 * index]] = out_grad_ptr[index];
      } else {
        dx_data[offset + medians_ptr[2 * index]] =
            out_grad_ptr[index] / static_cast<T>(2.0);
        dx_data[offset + medians_ptr[2 * index + 1]] =
            out_grad_ptr[index] / static_cast<T>(2.0);
      }
    }
  }
}

template <typename T>
__global__ void KernelNanmedianMinGrad(const int64_t* medians_ptr,
                                       const T* out_grad_ptr,
                                       T* dx_data,
                                       int64_t stride,
                                       int64_t pre_dim) {
  CUDA_KERNEL_LOOP(index, pre_dim) {
    int64_t offset = index * stride;

    if (medians_ptr[index] >= 0) {
      dx_data[offset + medians_ptr[index]] = out_grad_ptr[index];
    }
  }
}

template <typename T, typename Context>
void CalcMedianGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& median_index,
                          const DenseTensor& out_grad,
                          const std::string& mode,
                          DenseTensor* x_grad) {
  T* dx_data = dev_ctx.template Alloc<T>(x_grad);
  if (!dx_data) return;

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, x_grad, static_cast<T>(0));
  // VLOG(0) << "x_grad->dims():  " << x_grad->dims();

  auto stream = dev_ctx.stream();
  const T* x_data = x.data<T>();
  const int64_t* m_data = median_index.data<int64_t>();
  const T* out_grad_ptr = out_grad.data<T>();

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;

  if (mode == "avg") {
    KernelNanmedianMeanGrad<T>
        <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            m_data, out_grad_ptr, dx_data, stride, pre_dim);
  } else {  // mode == "min"
    KernelNanmedianMinGrad<T>
        <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            m_data, out_grad_ptr, dx_data, stride, pre_dim);
  }
}

template <typename T, typename Context>
void NanmedianGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& median_index,
                         const DenseTensor& out_grad,
                         const IntArray& axes,
                         bool keepdim UNUSED,
                         const std::string& mode,
                         DenseTensor* x_grad) {
  DenseTensor tmp_x;
  auto rank = x.dims().size();
  if ((axes.size() == 0) || rank <= 1) {
    tmp_x = x;
    tmp_x.Resize({x.numel()});
    CalcMedianGradKernel<T, Context>(
        dev_ctx, tmp_x, median_index, out_grad, mode, x_grad);
  } else {
    funcs::PreprocessMedianKernel<T, Context>(dev_ctx, x, axes, &tmp_x);

    DenseTensor tmp_x_grad;
    tmp_x_grad.Resize(x_grad->dims());
    CalcMedianGradKernel<T, Context>(
        dev_ctx, tmp_x, median_index, out_grad, mode, &tmp_x_grad);

    dev_ctx.template Alloc<T>(x_grad);
    funcs::PostprocessMedianGradKernel<T, Context>(
        dev_ctx, &tmp_x_grad, axes, x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(nanmedian_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::NanmedianGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
