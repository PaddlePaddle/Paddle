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

#include "paddle/phi/kernels/roi_align_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <class T>
void bilinear_interpolate_gradient(const int height,
                                   const int width,
                                   T y,
                                   T x,
                                   const T out_grad_this_bin,
                                   const T count,
                                   T* batch_grad_data) {
  int x_low, y_low, x_high, y_high;
  T w1, w2, w3, w4;
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0;
    x_low = x_high = y_low = y_high = -1;
    return;
  }
  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low, lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  T diff1 = out_grad_this_bin * w1 / count;
  T diff2 = out_grad_this_bin * w2 / count;
  T diff3 = out_grad_this_bin * w3 / count;
  T diff4 = out_grad_this_bin * w4 / count;
  if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
    *(batch_grad_data + y_low * width + x_low) += diff1;
    *(batch_grad_data + y_low * width + x_high) += diff2;
    *(batch_grad_data + y_high * width + x_low) += diff3;
    *(batch_grad_data + y_high * width + x_high) += diff4;
  }
}

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& boxes,
                        paddle::optional<const DenseTensor&> boxes_num,
                        const DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        DenseTensor* dx) {
  auto in_dims = x.dims();
  int channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int rois_num = boxes.dims()[0];

  if (!dx) {
    return;
  }

  DenseTensor roi_batch_id_list = Empty<int>(dev_ctx, {rois_num});
  int* box_batch_id_data = roi_batch_id_list.data<int>();

  int boxes_batch_size;
  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
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
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_id_data[i] = n;
      }
    }
  }
  dev_ctx.template Alloc<T>(dx);

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, dx, static_cast<T>(0));

  int output_grad_size = out_grad.numel();

  if ((!out_grad.IsInitialized()) || (output_grad_size <= 0)) {
    return;
  }

  const T* boxes_data = boxes.data<T>();
  const T* out_grad_data = out_grad.data<T>();
  T* dx_data = dev_ctx.template Alloc<T>(dx);

  auto in_stride = phi::stride(x.dims());
  auto roi_stride = phi::stride(boxes.dims());
  auto out_stride = phi::stride(out_grad.dims());

  T roi_offset = aligned ? T(0.5) : 0;
  for (int n = 0; n < rois_num; ++n) {
    int box_batch_idx = box_batch_id_data[n];
    T roi_xmin = boxes_data[0] * spatial_scale - roi_offset;
    T roi_ymin = boxes_data[1] * spatial_scale - roi_offset;
    T roi_xmax = boxes_data[2] * spatial_scale - roi_offset;
    T roi_ymax = boxes_data[3] * spatial_scale - roi_offset;

    T roi_width = roi_xmax - roi_xmin;
    T roi_height = roi_ymax - roi_ymin;
    roi_width = std::max(roi_width, static_cast<T>(1.));
    roi_height = std::max(roi_height, static_cast<T>(1.));
    if (!aligned) {
      roi_width = std::max(roi_width, static_cast<T>(1.));
      roi_height = std::max(roi_height, static_cast<T>(1.));
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    for (int c = 0; c < channels; ++c) {
      T* batch_grad_data =
          dx_data + box_batch_idx * in_stride[0] + c * in_stride[1];
      const T* batch_out_grad_data =
          out_grad_data + n * out_stride[0] + c * out_stride[1];
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int pool_index = ph * pooled_width + pw;
          T out_grad_this_bin = batch_out_grad_data[pool_index];
          int roi_bin_grid_h = (sampling_ratio > 0)
                                   ? sampling_ratio
                                   : ceil(roi_height / pooled_height);
          int roi_bin_grid_w = (sampling_ratio > 0)
                                   ? sampling_ratio
                                   : ceil(roi_width / pooled_width);
          T count = roi_bin_grid_h * roi_bin_grid_w;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_ymin + ph * bin_size_h +
                        static_cast<T>(iy + .5f) * bin_size_h /
                            static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              const T x = roi_xmin + pw * bin_size_w +
                          static_cast<T>(ix + .5f) * bin_size_w /
                              static_cast<T>(roi_bin_grid_w);
              bilinear_interpolate_gradient(height,
                                            width,
                                            y,
                                            x,
                                            out_grad_this_bin,
                                            count,
                                            batch_grad_data);
            }
          }
        }
      }
    }
    boxes_data += roi_stride[0];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(roi_align_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RoiAlignGradKernel,
                   float,
                   double,
                   int) {}
