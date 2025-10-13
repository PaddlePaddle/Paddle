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

#include <math.h>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/nanmedian_utils.h"

namespace phi {

template <typename T>
void CalcNanMedianMinGrad(int64_t pre_dim,
                          int64_t stride,
                          const int64_t* m_data,
                          T* dx_data,
                          const T* dout_data) {
  int64_t i = 0;
  int64_t offset = 0;
  for (i = 0; i < pre_dim; i++) {
    if (m_data[i] >= 0) {
      dx_data[offset + m_data[i]] = dout_data[i];
    }
    offset += stride;
  }
}

template <typename T>
void CalcNanMedianGradEvenly(int64_t pre_dim,
                             int64_t stride,
                             const DenseTensor& x,
                             const T* m_data,
                             const int64_t* m_index,
                             T* dx_data,
                             const T* dout_data) {
  int64_t i = 0, j = 0;
  int64_t offset = 0;
  std::vector<int64_t> data_index;
  const T* x_data = x.data<T>();
  for (i = 0; i < pre_dim; i++) {
    data_index.clear();
    for (j = 0; j < stride; j++) {
      if ((m_data[i] == x_data[offset + j]) ||
          (isnan(static_cast<float>(m_data[i])) &&
           isnan(static_cast<float>(x_data[offset + j])))) {
        data_index.push_back(offset + j);
      }
    }
    if (data_index.size() == 0) {
      if (m_index[2 * i] == m_index[2 * i + 1]) {
        dx_data[offset + m_index[2 * i]] = dout_data[i];
      } else {
        dx_data[offset + m_index[2 * i]] = dout_data[i] / static_cast<T>(2.0);
        dx_data[offset + m_index[2 * i + 1]] =
            dout_data[i] / static_cast<T>(2.0);
      }
    } else {
      for (j = 0; j < static_cast<int64_t>(data_index.size()); j++) {
        dx_data[data_index[j]] =
            dout_data[i] / static_cast<T>(data_index.size());
      }
    }

    offset += stride;
  }
}

template <typename T, typename Context>
void CalcNanMedianGradKernel_CPU(const Context& dev_ctx,
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

  const int64_t* m_index = median_index.data<int64_t>();
  const T* m_data = median_data.data<T>();
  const T* dout_data = out_grad.data<T>();
  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t rank = x_dim.size();
  int64_t stride = x_dim[static_cast<int>(rank - 1)];
  int64_t pre_dim = numel / stride;
  if (!evenly) {
    CalcNanMedianMinGrad(pre_dim, stride, m_index, dx_data, dout_data);
  } else {
    CalcNanMedianGradEvenly(
        pre_dim, stride, x, m_data, m_index, dx_data, dout_data);
  }
}

template <typename T, typename Context>
void NanmedianGradKernel(const Context& dev_ctx,
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
    CalcNanMedianGradKernel_CPU<T, Context>(dev_ctx,
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
    CalcNanMedianGradKernel_CPU<T, Context>(dev_ctx,
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

PD_REGISTER_KERNEL(nanmedian_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::NanmedianGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
