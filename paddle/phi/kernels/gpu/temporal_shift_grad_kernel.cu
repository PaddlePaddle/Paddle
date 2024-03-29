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

#include "paddle/phi/kernels/temporal_shift_grad_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void KeTemporalShiftBwNCHW(const T* output_grad,
                                      T* input_grad,
                                      const int ntchw,
                                      const int tchw,
                                      const int chw,
                                      const int hw,
                                      const int t,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < ntchw; tid += stride) {
    int it = (tid % tchw) / chw;
    int ic = (tid % chw) / hw;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[tid] = output_grad[tid + (src_it - it) * chw];
    } else {
      input_grad[tid] = 0;
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftBwNHWC(const T* output_grad,
                                      T* input_grad,
                                      const int nthwc,
                                      const int thwc,
                                      const int hwc,
                                      const int t,
                                      const int c,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < nthwc; tid += stride) {
    int it = (tid % thwc) / hwc;
    int ic = tid % c;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[tid] = output_grad[tid + (src_it - it) * hwc];
    } else {
      input_grad[tid] = 0;
    }
  }
}

template <typename T, typename Context>
void TemporalShiftGradKernel(const Context& dev_ctx,
                             const DenseTensor& out_grad,
                             int seg_num,
                             float shift_ratio,
                             const std::string& data_format_str,
                             DenseTensor* x_grad) {
  auto* input_grad = x_grad;
  auto* output_grad = &out_grad;
  int t = seg_num;
  const DataLayout data_layout = common::StringToDataLayout(data_format_str);

  const int nt = output_grad->dims()[0];
  const int c = (data_layout == DataLayout::kNCHW ? output_grad->dims()[1]
                                                  : output_grad->dims()[3]);
  const int h = (data_layout == DataLayout::kNCHW ? output_grad->dims()[2]
                                                  : output_grad->dims()[1]);
  const int w = (data_layout == DataLayout::kNCHW ? output_grad->dims()[3]
                                                  : output_grad->dims()[2]);

  const int hw = h * w;
  const int chw = c * hw;
  const int tchw = t * chw;
  const int ntchw = nt * chw;

  const int c1 = static_cast<int>(c * shift_ratio);
  const int c2 = static_cast<int>(c * 2 * shift_ratio);

  DDim in_grad_dims =
      (data_layout == DataLayout::kNCHW ? common::make_ddim({nt, c, h, w})
                                        : common::make_ddim({nt, h, w, c}));
  const T* output_grad_data = output_grad->data<T>();
  input_grad->Resize(in_grad_dims);
  T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);

  int pixelNum = nt * chw;
  int threads = 1024;
  int grid = (pixelNum + threads - 1) / threads;
  int blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() / threads;
  grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

  if (data_layout == DataLayout::kNCHW) {
    KeTemporalShiftBwNCHW<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        output_grad_data, input_grad_data, ntchw, tchw, chw, hw, t, c1, c2);
  } else {
    KeTemporalShiftBwNHWC<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        output_grad_data, input_grad_data, ntchw, tchw, chw, t, c, c1, c2);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(temporal_shift_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TemporalShiftGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
