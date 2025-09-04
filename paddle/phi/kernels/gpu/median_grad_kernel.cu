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

#include "paddle/phi/kernels/median_grad_kernel.h"

#include <math.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/nanmedian_utils.h"
#include "paddle/phi/kernels/gpu/reduce_amin_amax_common.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;
inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelMedianMeanGrad(const int64_t* medians_ptr,
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
__global__ void KernelMedianMinGrad(const int64_t* medians_ptr,
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

template <typename T>
__global__ void KernelMedianGradEvenly(const T* medians_ptr,
                                       const int64_t* median_index_ptr,
                                       const T* out_grad_ptr,
                                       T* x,
                                       T* dx_data,
                                       int64_t stride,
                                       int64_t pre_dim) {
  CUDA_KERNEL_LOOP(index, pre_dim) {
    int64_t offset = index * stride;
    if (median_index_ptr[2 * index] >= 0 &&
        !isnan(static_cast<float>(medians_ptr[index]))) {
      x[offset + median_index_ptr[2 * index]] = medians_ptr[index];

      x[offset + median_index_ptr[2 * index + 1]] = medians_ptr[index];
    }
  }
}

template <typename T, typename Context>
void CalcMedianGradKernel_GPU(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& median_data,
                              const DenseTensor& median_index,
                              const DenseTensor& out_grad,
                              const std::string& mode,
                              const bool evenly,
                              DenseTensor* x_grad) {
  T* dx_data = dev_ctx.template Alloc<T>(x_grad);
  if (!dx_data) return;

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, x_grad, static_cast<T>(0));
  // VLOG(0) << "x_grad->dims():  " << x_grad->dims();

  auto stream = dev_ctx.stream();
  const T* x_data = x.data<T>();
  const int64_t* m_index = median_index.data<int64_t>();
  const T* m_data = median_data.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;
  if (!evenly) {
    if (mode == "avg") {
      KernelMedianMeanGrad<T>
          <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              m_index, out_grad_ptr, dx_data, stride, pre_dim);
    } else {  // mode == "min"
      KernelMedianMinGrad<T>
          <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              m_index, out_grad_ptr, dx_data, stride, pre_dim);
    }
  } else {
    std::vector<int64_t> dims;
    dims.push_back(-1);
    DenseTensor tmp_x(x);
    dev_ctx.template Alloc<T>(&tmp_x);
    T* tmp_x_data = tmp_x.data<T>();
    if (mode == "avg") {
      KernelMedianGradEvenly<T>
          <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              m_data,
              m_index,
              out_grad_ptr,
              tmp_x_data,
              dx_data,
              stride,
              pre_dim);
    }
    auto grad_dim = x_grad->dims();
    x_grad->Resize(x.dims());
    ReduceCudaAMaxAMinGrad<T, Context>(
        dev_ctx, tmp_x, median_data, out_grad, dims, true, false, x_grad, true);
    x_grad->Resize(grad_dim);
  }
}

template <typename T, typename Context>
void MedianGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& median_data,
                      const DenseTensor& median_index,
                      const DenseTensor& out_grad,
                      const IntArray& axes,
                      bool keepdim UNUSED,
                      const std::string& mode,
                      DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  bool evenly = (axes.size() != 1 || mode == "avg");
  DenseTensor tmp_x;
  auto rank = x.dims().size();
  if ((axes.size() == 0) || rank <= 1) {
    tmp_x = x;
    tmp_x.Resize({x.numel()});
    CalcMedianGradKernel_GPU<T, Context>(dev_ctx,
                                         tmp_x,
                                         median_data,
                                         median_index,
                                         out_grad,
                                         mode,
                                         evenly,
                                         x_grad);
  } else {
    funcs::PreprocessMedianKernel<T, Context>(dev_ctx, x, axes, &tmp_x);

    DenseTensor tmp_x_grad;
    tmp_x_grad.Resize(x_grad->dims());
    CalcMedianGradKernel_GPU<T, Context>(dev_ctx,
                                         tmp_x,
                                         median_data,
                                         median_index,
                                         out_grad,
                                         mode,
                                         evenly,
                                         &tmp_x_grad);
    dev_ctx.template Alloc<T>(x_grad);
    funcs::PostprocessMedianGradKernel<T, Context>(
        dev_ctx, &tmp_x_grad, axes, x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(median_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MedianGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
