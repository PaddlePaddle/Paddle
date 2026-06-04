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

#include "paddle/phi/kernels/psroi_pool_grad_kernel.h"

#include <algorithm>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void PsroiPoolGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& rois,
                         const optional<DenseTensor>& rois_num,
                         const DenseTensor& dout,
                         int pooled_height,
                         int pooled_width,
                         int output_channels,
                         float spatial_scale,
                         DenseTensor* dx) {
  if (dx) {
    const auto& in_dims = x.dims();
    int64_t input_channels = in_dims[1];
    int64_t height = in_dims[2];
    int64_t width = in_dims[3];
    int64_t rois_num_t = rois.dims()[0];

    // set roi batch id
    DenseTensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num_t});
    int* rois_batch_id_data = dev_ctx.template Alloc<int>(&rois_batch_id_list);
    int64_t rois_batch_size = 0;
    if (rois_num.get_ptr()) {
      rois_batch_size = rois_num->numel();
      auto* rois_num_t_data = rois_num->data<int>();
      int64_t start = 0;
      for (int64_t n = 0; n < rois_batch_size; ++n) {
        for (int64_t i = start; i < start + rois_num_t_data[n]; ++i) {
          rois_batch_id_data[i] = n;
        }
        start += rois_num_t_data[n];
      }
    } else {
      auto rois_lod = rois.lod().back();
      rois_batch_size = static_cast<int64_t>(rois_lod.size()) - 1;
      // calculate batch id index for each roi according to LoD
      for (int64_t n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
      }
    }
    const T* input_rois = rois.data<T>();
    const T* dout_data = dout.data<T>();
    T* dx_data = dev_ctx.template Alloc<T>(dx);

    // set gradient of X to be 0. before backpropagate.
    funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dx, static_cast<T>(0));

    // backpropagate gradient per output pixel
    int64_t dout_size = dout.numel();
    for (int64_t i = 0; i < dout_size; ++i) {
      // The output is in order (n, c, ph, pw)
      int64_t pw = i % pooled_width;
      int64_t ph = (i / pooled_width) % pooled_height;
      int64_t c = (i / pooled_width / pooled_height) % output_channels;
      int64_t n = i / pooled_width / pooled_height / output_channels;

      // set roi_batch_id
      int64_t roi_batch_id = rois_batch_id_data[n];
      int64_t input_channel =
          (static_cast<int64_t>(c) * pooled_height + ph) * pooled_width + pw;
      int64_t input_offset =
          (static_cast<int64_t>(roi_batch_id) * input_channels +
           input_channel) *
          height * width;
      T* offset_dx_data = dx_data + input_offset;

      // [start, end) interval for spatial sampling
      const T* offset_input_rois = input_rois + static_cast<int64_t>(n) * 4;
      T roi_start_w =
          static_cast<T>(round(offset_input_rois[0])) * spatial_scale;
      T roi_start_h =
          static_cast<T>(round(offset_input_rois[1])) * spatial_scale;
      T roi_end_w =
          static_cast<T>(round(offset_input_rois[2]) + 1.) * spatial_scale;
      T roi_end_h =
          static_cast<T>(round(offset_input_rois[3]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      T roi_height = std::max(roi_end_h - roi_start_h, (T)0.1);  // avoid 0
      T roi_width = std::max(roi_end_w - roi_start_w, (T)0.1);

      // Compute w and h at input feature map
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);

      int64_t hstart = floor(bin_size_h * static_cast<T>(ph) + roi_start_h);
      int64_t wstart = floor(bin_size_w * static_cast<T>(pw) + roi_start_w);
      int64_t hend = ceil(bin_size_h * static_cast<T>(ph + 1) + roi_start_h);
      int64_t wend = ceil(bin_size_w * static_cast<T>(pw + 1) + roi_start_w);

      // Add roi offsets and clip to input boundaries
      hstart = std::min(std::max(hstart, static_cast<int64_t>(0)), height);
      hend = std::min(std::max(hend, static_cast<int64_t>(0)), height);
      wstart = std::min(std::max(wstart, static_cast<int64_t>(0)), width);
      wend = std::min(std::max(wend, static_cast<int64_t>(0)), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Accumulate diff_val into input data
      T bin_area =
          static_cast<T>(static_cast<int64_t>(hend - hstart) * (wend - wstart));
      T diff_val = is_empty ? 0. : dout_data[i] / bin_area;
      for (int64_t ih = hstart; ih < hend; ++ih) {
        for (int64_t iw = wstart; iw < wend; ++iw) {
          int64_t input_index = static_cast<int64_t>(ih) * width + iw;
          offset_dx_data[input_index] += diff_val;
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    psroi_pool_grad, CPU, ALL_LAYOUT, phi::PsroiPoolGradKernel, float, double) {
  kernel->InputAt(2).SetDataType(phi::CppTypeToDataType<int>::Type());
}
