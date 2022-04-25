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

#include "paddle/phi/kernels/roi_pool_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void RoiPoolKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& boxes,
                   paddle::optional<const DenseTensor&> boxes_num,
                   int pooled_height,
                   int pooled_width,
                   float spatial_scale,
                   DenseTensor* out,
                   DenseTensor* arg_max) {
  auto x_dims = x.dims();
  int batch_size = x_dims[0];
  int channels = x_dims[1];
  int height = x_dims[2];
  int width = x_dims[3];
  int rois_num = boxes.dims()[0];

  auto in_stride = phi::stride(x_dims);
  auto arg_max_stride = phi::stride(arg_max->dims());
  auto box_stride = phi::stride(boxes.dims());
  auto out_stride = phi::stride(out->dims());

  const T* input_data = x.data<T>();

  DenseTensor box_batch_id_list = Empty<int>(dev_ctx, {rois_num});
  int* box_batch_id_data = box_batch_id_list.data<int>();

  int boxes_batch_size;
  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument("The boxes_batch_size and imgs "
                                     "batch_size must be the same."));
    auto* boxes_num_data = boxes_num->data<int>();
    int start = 0;
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        box_batch_id_data[i] = n;
      }
      start += boxes_num_data[n];
    }
  } else {
    auto boxes_lod = boxes.lod().back();
    boxes_batch_size = boxes_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument("The boxes_batch_size and imgs "
                                     "batch_size must be the same."));
    int rois_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(
        rois_num,
        rois_num_with_lod,
        phi::errors::InvalidArgument("The rois_num from input "
                                     "and lod must be the same."));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_id_data[i] = n;
      }
    }
  }

  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* arg_max_data = dev_ctx.template Alloc<int64_t>(arg_max);

  const T* boxes_data = boxes.data<T>();
  for (int n = 0; n < rois_num; ++n) {
    int box_batch_id = box_batch_id_data[n];
    int box_start_w = round(boxes_data[0] * spatial_scale);
    int box_start_h = round(boxes_data[1] * spatial_scale);
    int box_end_w = round(boxes_data[2] * spatial_scale);
    int box_end_h = round(boxes_data[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int box_height = std::max(box_end_h - box_start_h + 1, 1);
    int box_width = std::max(box_end_w - box_start_w + 1, 1);

    const float bin_size_h =
        static_cast<float>(box_height) / static_cast<float>(pooled_height);
    const float bin_size_w =
        static_cast<float>(box_width) / static_cast<float>(pooled_width);

    const T* batch_data = input_data + box_batch_id * in_stride[0];

    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          //  Compute pooling region for this output unit:
          //  start (included) = floor(ph * box_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * box_height / pooled_height_)
          int hstart =
              static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
          int wstart =
              static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
          int hend =
              static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
          int wend =
              static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

          hstart = std::min(std::max(hstart + box_start_h, 0), height);
          hend = std::min(std::max(hend + box_start_h, 0), height);
          wstart = std::min(std::max(wstart + box_start_w, 0), width);
          wend = std::min(std::max(wend + box_start_w, 0), width);

          const int pool_index = ph * pooled_width + pw;

          // Define an empty pooling region to be zero
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          output_data[pool_index] =
              is_empty ? 0 : -std::numeric_limits<T>::max();
          arg_max_data[pool_index] = -1;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              if (batch_data[index] > output_data[pool_index]) {
                output_data[pool_index] = batch_data[index];
                arg_max_data[pool_index] = index;
              }
            }
          }
        }
      }

      batch_data += in_stride[1];
      output_data += out_stride[1];
      arg_max_data += arg_max_stride[1];
    }
    // Increment ROI data pointer
    boxes_data += box_stride[0];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_pool, CPU, ALL_LAYOUT, phi::RoiPoolKernel, float, double, int) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
