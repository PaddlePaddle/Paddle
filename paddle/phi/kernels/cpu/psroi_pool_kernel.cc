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

#include "paddle/phi/kernels/psroi_pool_kernel.h"

#include <algorithm>

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void PsroiPoolKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& rois,
                     const optional<DenseTensor>& rois_num,
                     int pooled_height,
                     int pooled_width,
                     int output_channels,
                     float spatial_scale,
                     DenseTensor* out) {
  auto in_dims = x.dims();
  int64_t batch_size = in_dims[0];
  int64_t input_channels = in_dims[1];
  int64_t height = in_dims[2];
  int64_t width = in_dims[3];
  int64_t rois_num_t = rois.dims()[0];

  PADDLE_ENFORCE_EQ(
      input_channels,
      static_cast<int64_t>(output_channels) * pooled_height * pooled_width,
      errors::InvalidArgument(
          "Expected the channels of input X to be equal to output_channels * "
          "pooled_height * pooled_width, but received input_channels: %ld, "
          "output_channels: %d, pooled_height: %d, pooled_width: %d, "
          "expected channels: %ld.",
          input_channels,
          output_channels,
          pooled_height,
          pooled_width,
          static_cast<int64_t>(output_channels) * pooled_height *
              pooled_width));

  auto in_stride = stride(in_dims);
  auto out_stride = stride(out->dims());

  const T* input_data = x.data<T>();

  DenseTensor rois_batch_id_list;
  rois_batch_id_list.Resize({rois_num_t});
  int* rois_batch_id_data = dev_ctx.template Alloc<int>(&rois_batch_id_list);

  int64_t rois_batch_size = 0;
  if (rois_num.get_ptr()) {
    rois_batch_size = rois_num->numel();
    auto* rois_num_data = rois_num->data<int>();
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The batch size of rois and the batch size of images "
            " must be the same. But received the batch size of rois is %d, "
            "and the batch size of images is %d",
            rois_batch_size,
            batch_size));
    int rois_num_count = 0;
    for (int64_t i = 0; i < rois_batch_size; ++i) {
      rois_num_count += rois_num_data[i];
    }
    PADDLE_ENFORCE_EQ(
        rois_num_count,
        rois_num_t,
        errors::InvalidArgument(
            "the rois_num from input and RoisNum must be the same"));
    int start = 0;
    for (int64_t n = 0; n < rois_batch_size; ++n) {
      for (int64_t i = start; i < start + rois_num_data[n]; ++i) {
        rois_batch_id_data[i] = n;
      }
      start += rois_num_data[n];
    }
  } else {
    auto rois_lod = rois.lod().back();
    rois_batch_size = static_cast<int64_t>(rois_lod.size()) - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch_size,
        errors::InvalidArgument("the rois_batch_size and input(X) "
                                "batch_size should be the same."));
    int64_t rois_num_with_lod = static_cast<int64_t>(rois_lod[rois_batch_size]);
    PADDLE_ENFORCE_EQ(rois_num_with_lod,
                      rois_num_t,
                      errors::InvalidArgument(
                          "the rois_num from input and lod must be the same"));
    // calculate batch id index for each roi according to LoD
    for (int64_t n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        rois_batch_id_data[i] = n;
      }
    }
  }
  T* output_data = dev_ctx.template Alloc<T>(out);
  const T* input_rois = rois.data<T>();

  // calculate psroipooling, parallel processing can be implemented per ROI
  for (int64_t n = 0; n < rois_num_t; ++n) {
    // set roi batch id
    int roi_batch_id = rois_batch_id_data[n];

    // [start, end) interval for spatial sampling
    const T* offset_input_rois = input_rois + static_cast<int64_t>(n) * 4;
    T roi_start_w = static_cast<T>(round(offset_input_rois[0])) * spatial_scale;
    T roi_start_h = static_cast<T>(round(offset_input_rois[1])) * spatial_scale;
    T roi_end_w =
        static_cast<T>(round(offset_input_rois[2]) + 1.) * spatial_scale;
    T roi_end_h =
        static_cast<T>(round(offset_input_rois[3]) + 1.) * spatial_scale;
    // Force too small rois to be 1 x 1
    T roi_height = std::max(roi_end_h - roi_start_h, (T)0.1);  // avoid 0
    T roi_width = std::max(roi_end_w - roi_start_w, (T)0.1);

    // Compute bin size w and h at input feature map
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    // calculate each pixel of the output feature map.
    int64_t out_roi_offset = n * out_stride[0];
    for (int c = 0; c < output_channels; ++c) {
      // per category
      int64_t out_plane_offset = out_roi_offset + c * out_stride[1];
      for (int ph = 0; ph < pooled_height; ++ph) {
        int64_t out_row_offset = out_plane_offset + ph * out_stride[2];
        for (int pw = 0; pw < pooled_width; ++pw) {
          // calculate w and h at input feature map
          int64_t hstart = floor(static_cast<T>(ph) * bin_size_h + roi_start_h);
          int64_t wstart = floor(static_cast<T>(pw) * bin_size_w + roi_start_w);
          int64_t hend =
              ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
          int64_t wend =
              ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w);
          //  Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart, static_cast<int64_t>(0)), height);
          wstart = std::min(std::max(wstart, static_cast<int64_t>(0)), width);
          hend = std::min(std::max(hend, static_cast<int64_t>(0)), height);
          wend = std::min(std::max(wend, static_cast<int64_t>(0)), width);

          int64_t output_index = out_row_offset + pw;
          int64_t input_channel =
              (static_cast<int64_t>(c) * pooled_height + ph) * pooled_width +
              pw;
          int64_t input_plane_offset =
              roi_batch_id * in_stride[0] + input_channel * in_stride[1];
          const T* offset_input_data = input_data + input_plane_offset;
          T out_sum = 0.;
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          for (int64_t ih = hstart; ih < hend; ++ih) {
            for (int64_t iw = wstart; iw < wend; ++iw) {
              int64_t input_index = ih * in_stride[2] + iw;
              out_sum += offset_input_data[input_index];
            }
          }
          T bin_area = static_cast<int64_t>(hend - hstart) * (wend - wstart);
          output_data[output_index] = is_empty ? 0. : out_sum / bin_area;
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    psroi_pool, CPU, ALL_LAYOUT, phi::PsroiPoolKernel, float, double) {
  kernel->InputAt(2).SetDataType(phi::CppTypeToDataType<int>::Type());
}
