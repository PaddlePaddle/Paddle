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

#include "paddle/phi/kernels/temporal_shift_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IndexT>
__global__ void KeTemporalShiftFwNCHW(const T* input,
                                      T* output,
                                      const IndexT ntchw,
                                      const IndexT tchw,
                                      const IndexT chw,
                                      const IndexT hw,
                                      const int t,
                                      const IndexT c1,
                                      const IndexT c2) {
  IndexT tid = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT stride = static_cast<IndexT>(blockDim.x) * gridDim.x;
  IndexT src_it = 0;

  for (; tid < ntchw; tid += stride) {
    IndexT it = (tid % tchw) / chw;
    IndexT ic = (tid % chw) / hw;

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

template <typename T, typename IndexT>
__global__ void KeTemporalShiftFwNHWC(const T* input,
                                      T* output,
                                      const IndexT nthwc,
                                      const IndexT thwc,
                                      const IndexT hwc,
                                      const int t,
                                      const IndexT c,
                                      const IndexT c1,
                                      const IndexT c2) {
  IndexT tid = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT stride = static_cast<IndexT>(blockDim.x) * gridDim.x;
  IndexT src_it = 0;

  for (; tid < nthwc; tid += stride) {
    IndexT it = (tid % thwc) / hwc;
    IndexT ic = tid % c;

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
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto* input = &x;
  auto* output = out;
  int t = seg_num;
  const DataLayout data_layout = common::StringToDataLayout(data_format_str);

  const int64_t nt = input->dims()[0];
  const int64_t c =
      (data_layout == DataLayout::kNCHW ? input->dims()[1] : input->dims()[3]);
  const int64_t h =
      (data_layout == DataLayout::kNCHW ? input->dims()[2] : input->dims()[1]);
  const int64_t w =
      (data_layout == DataLayout::kNCHW ? input->dims()[3] : input->dims()[2]);

  const int64_t hw = h * w;
  const int64_t chw = c * hw;
  const int64_t tchw = t * chw;
  const int64_t ntchw = nt * chw;

  const int64_t c1 = static_cast<int64_t>(c * shift_ratio);
  const int64_t c2 = static_cast<int64_t>(c * 2 * shift_ratio);

  DDim out_dims =
      (data_layout == DataLayout::kNCHW ? common::make_ddim({nt, c, h, w})
                                        : common::make_ddim({nt, h, w, c}));
  const T* input_data = input->data<T>();
  output->Resize(out_dims);
  T* output_data = dev_ctx.template Alloc<T>(output);

  int64_t pixelNum = nt * chw;
  int64_t threads = 1024;
  int64_t grid = (pixelNum + threads - 1) / threads;
  int64_t blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() / threads;
  grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

  if (data_layout == DataLayout::kNCHW) {
    if (x.numel() < std::numeric_limits<int32_t>::max()) {
      KeTemporalShiftFwNCHW<T, int32_t><<<grid, threads, 0, dev_ctx.stream()>>>(
          input_data, output_data, ntchw, tchw, chw, hw, t, c1, c2);
    } else {
      KeTemporalShiftFwNCHW<T, int64_t><<<grid, threads, 0, dev_ctx.stream()>>>(
          input_data, output_data, ntchw, tchw, chw, hw, t, c1, c2);
    }
  } else {
    if (x.numel() < std::numeric_limits<int32_t>::max()) {
      KeTemporalShiftFwNHWC<T, int32_t><<<grid, threads, 0, dev_ctx.stream()>>>(
          input_data, output_data, ntchw, tchw, chw, t, c, c1, c2);
    } else {
      KeTemporalShiftFwNHWC<T, int64_t><<<grid, threads, 0, dev_ctx.stream()>>>(
          input_data, output_data, ntchw, tchw, chw, t, c, c1, c2);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(temporal_shift,
                   GPU,
                   ALL_LAYOUT,
                   phi::TemporalShiftKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
