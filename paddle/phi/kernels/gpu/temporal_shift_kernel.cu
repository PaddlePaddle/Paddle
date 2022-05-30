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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/temporal_shift_kernel.h"

namespace phi {

template <typename T>
__global__ void KeTemporalShiftFwNCHW(const T* input,
                                      T* output,
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
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[tid] = 0;
    } else {
      output[tid] = input[tid + (src_it - it) * chw];
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftFwNHWC(const T* input,
                                      T* output,
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
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[tid] = 0;
    } else {
      output[tid] = input[tid + (src_it - it) * hwc];
    }
  }
}

template <typename T, typename Context>
void TemporalShiftKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int seg_num,
                         float shift_ratio,
                         const std::string& data_format_str,
                         DenseTensor* out) {
  auto* input = &x;
  auto* output = out;
  int t = seg_num;
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format_str);

  const int nt = input->dims()[0];
  const int c =
      (data_layout == DataLayout::kNCHW ? input->dims()[1] : input->dims()[3]);
  const int h =
      (data_layout == DataLayout::kNCHW ? input->dims()[2] : input->dims()[1]);
  const int w =
      (data_layout == DataLayout::kNCHW ? input->dims()[3] : input->dims()[2]);

  const int hw = h * w;
  const int chw = c * hw;
  const int tchw = t * chw;
  const int ntchw = nt * chw;

  const int c1 = static_cast<int>(c * shift_ratio);
  const int c2 = static_cast<int>(c * 2 * shift_ratio);

  DDim out_dims =
      (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                        : phi::make_ddim({nt, h, w, c}));
  const T* input_data = input->data<T>();
  T* output_data = output->mutable_data<T>(out_dims, dev_ctx.GetPlace());

  int pixelNum = nt * chw;
  int threads = 1024;
  int grid = (pixelNum + threads - 1) / threads;
  int blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() / threads;
  grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

  if (data_layout == DataLayout::kNCHW) {
    KeTemporalShiftFwNCHW<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        input_data, output_data, ntchw, tchw, chw, hw, t, c1, c2);
  } else {
    KeTemporalShiftFwNHWC<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        input_data, output_data, ntchw, tchw, chw, t, c, c1, c2);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(temporal_shift,
                   GPU,
                   ALL_LAYOUT,
                   phi::TemporalShiftKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
