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

#include "paddle/phi/kernels/roi_align_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

constexpr size_t GetOffset(size_t x, size_t y, size_t width) {
  return y * width + x;
}

template <class T>
struct OffsetsAndRatios {
  OffsetsAndRatios() = default;
  OffsetsAndRatios(std::size_t xy,
                   std::size_t xY,
                   std::size_t Xy,
                   std::size_t XY,
                   T xy_ratio,
                   T xY_ratio,
                   T Xy_ratio,
                   T XY_ratio)
      : xy(xy),
        xY(xY),
        Xy(Xy),
        XY(XY),
        xy_ratio(xy_ratio),
        xY_ratio(xY_ratio),
        Xy_ratio(Xy_ratio),
        XY_ratio(XY_ratio) {}

  std::size_t xy = 0;
  std::size_t xY = 0;
  std::size_t Xy = 0;
  std::size_t XY = 0;
  T xy_ratio = 0.0f;
  T xY_ratio = 0.0f;
  T Xy_ratio = 0.0f;
  T XY_ratio = 0.0f;
};

template <typename T>
std::vector<OffsetsAndRatios<T>> GetIndexesAndRatios(
    std::size_t width,
    std::size_t height,
    const T roi_width,
    const T roi_height,
    const T roi_xmin,
    const T roi_ymin,
    std::size_t pooled_width,
    std::size_t roi_bin_grid_w,
    std::size_t pooled_height,
    std::size_t roi_bin_grid_h) {
  const auto ind_num =
      pooled_width * roi_bin_grid_w * pooled_height * roi_bin_grid_h;

  std::vector<OffsetsAndRatios<T>> interpolation_cords;
  interpolation_cords.reserve(ind_num);

  const auto bin_w = roi_width / pooled_width;
  const auto bin_h = roi_height / pooled_height;

  for (std::size_t py = 0; py < pooled_height; py++) {
    for (std::size_t px = 0; px < pooled_width; px++) {
      for (std::size_t iy = 0; iy < roi_bin_grid_h; iy++) {
        // calculate x of sample points
        auto y = roi_ymin + bin_h * (py + static_cast<T>(iy + .5f) /  // NOLINT
                                              static_cast<T>(roi_bin_grid_h));
        for (std::size_t ix = 0; ix < roi_bin_grid_w; ix++) {
          // calculate x of sample points
          auto x =
              roi_xmin + bin_w * (px + static_cast<T>(ix + .5f) /  // NOLINT
                                           static_cast<T>(roi_bin_grid_w));

          // deal with elements out of map
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            interpolation_cords.emplace_back();
            continue;
          }
          y = y <= 0 ? 0 : y;
          x = x <= 0 ? 0 : x;

          std::size_t x_low_index = static_cast<std::size_t>(x);
          std::size_t x_high_index;
          if (x_low_index >= width - 1) {
            x_high_index = x_low_index = width - 1;
            x = static_cast<T>(x_low_index);
          } else {
            x_high_index = x_low_index + 1;
          }
          T x_ratio = x_high_index - x;

          std::size_t y_low_index = static_cast<std::size_t>(y);
          std::size_t y_high_index;
          if (y_low_index >= height - 1) {
            y_high_index = y_low_index = height - 1;
            y = static_cast<T>(y_low_index);
          } else {
            y_high_index = y_low_index + 1;
          }
          T y_ratio = y_high_index - y;

          auto xy = GetOffset(x_low_index, y_low_index, width);
          auto xY = GetOffset(x_low_index, y_high_index, width);
          auto Xy = GetOffset(x_high_index, y_low_index, width);
          auto XY = GetOffset(x_high_index, y_high_index, width);

          auto xy_ratio = x_ratio * y_ratio;
          auto xY_ratio = x_ratio * (1 - y_ratio);
          auto Xy_ratio = (1 - x_ratio) * y_ratio;
          auto XY_ratio = (1 - x_ratio) * (1 - y_ratio);

          interpolation_cords.emplace_back(
              xy, xY, Xy, XY, xy_ratio, xY_ratio, Xy_ratio, XY_ratio);
        }
      }
    }
  }
  return interpolation_cords;
}

template <typename T>
void Interpolate(std::vector<T>& interpolated_values,  // NOLINT
                 const std::vector<OffsetsAndRatios<T>>& interpolation_cords,
                 const T* data) {
  for (auto& ic : interpolation_cords) {
    auto xlyl_offset = ic.xy;
    auto xhyl_offset = ic.Xy;
    auto xlyh_offset = ic.xY;
    auto xhyh_offset = ic.XY;

    auto xlyl_ratio = ic.xy_ratio;
    auto xhyl_ratio = ic.Xy_ratio;
    auto xlyh_ratio = ic.xY_ratio;
    auto xhyh_ratio = ic.XY_ratio;

    interpolated_values.emplace_back(
        xlyl_ratio * data[xlyl_offset] + xhyl_ratio * data[xhyl_offset] +
        xlyh_ratio * data[xlyh_offset] + xhyh_ratio * data[xhyh_offset]);
  }
}

template <typename T>
void AvgPool(const std::vector<T>& interpolated_values,
             T* output_data,
             int roi_bin_grid_w,
             int roi_bin_grid_h,
             int pooled_width,
             int pooled_height) {
  const auto data_amount = pooled_width * pooled_height;
  const auto grid_points = roi_bin_grid_w * roi_bin_grid_h;
  const T count = 1.0 / grid_points;
  auto val_begin = interpolated_values.cbegin();
  for (auto i = 0; i < data_amount; ++i) {
    T sum = 0.0;
    auto val_end = val_begin + grid_points;
    sum = std::accumulate(val_begin, val_end, sum);
    val_begin = val_end;
    output_data[i] = sum * count;
  }
}

template <typename T, typename Context>
void RoiAlignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& boxes,
                    const paddle::optional<DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    DenseTensor* out) {
  auto in_dims = x.dims();
  int batch_size = static_cast<int>(in_dims[0]);
  int channels = static_cast<int>(in_dims[1]);
  int height = static_cast<int>(in_dims[2]);
  int width = static_cast<int>(in_dims[3]);
  int rois_num = static_cast<int>(boxes.dims()[0]);

  if (rois_num == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto in_stride = common::stride(in_dims);
  auto roi_stride = common::stride(boxes.dims());
  auto out_stride = common::stride(out->dims());

  const T* input_data = x.data<T>();
  DenseTensor roi_batch_id_list = Empty<int>(dev_ctx, {rois_num});
  int* roi_batch_id_data = roi_batch_id_list.data<int>();
  int boxes_batch_size;
  if (boxes_num) {
    boxes_batch_size = static_cast<int>(boxes_num->numel());
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The batch size of rois and the batch size of images "
            " must be the same. But received the batch size of rois is %d, "
            "and the batch size of images is %d",
            boxes_batch_size,
            batch_size));
    if (boxes_num->dtype() == phi::DataType::INT64) {
      auto* boxes_num_data = boxes_num->data<int64_t>();
      int64_t start = 0;
      for (int64_t n = 0; n < boxes_batch_size; ++n) {
        for (int64_t i = start; i < start + boxes_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += boxes_num_data[n];
      }
    } else if (boxes_num->dtype() == phi::DataType::INT32) {
      auto* boxes_num_data = boxes_num->data<int>();
      int start = 0;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (int i = start; i < start + boxes_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += boxes_num_data[n];
      }
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(),
        false,
        errors::InvalidArgument("Input(ROIs) Tensor of ROIAlignOp "
                                "does not contain LoD information."));
    auto boxes_lod = lod.back();
    int boxes_batch_size = static_cast<int>(boxes_lod.size() - 1);
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The boxes_batch_size and imgs "
            "batch_size must be the same. But received boxes_batch_size = %d, "
            "batch_size = %d",
            boxes_batch_size,
            batch_size));
    int boxes_num_with_lod = static_cast<int>(boxes_lod[boxes_batch_size]);
    PADDLE_ENFORCE_EQ(
        rois_num,
        boxes_num_with_lod,
        errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            rois_num,
            boxes_num_with_lod));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }
  T* output_data = dev_ctx.template Alloc<T>(out);
  const T* boxes_data = boxes.data<T>();
  T roi_offset = aligned ? T(0.5) : 0;
  for (int n = 0; n < rois_num; ++n) {
    int roi_batch_id = roi_batch_id_data[n];
    T roi_xmin = boxes_data[0] * spatial_scale - roi_offset;
    T roi_ymin = boxes_data[1] * spatial_scale - roi_offset;
    T roi_xmax = boxes_data[2] * spatial_scale - roi_offset;
    T roi_ymax = boxes_data[3] * spatial_scale - roi_offset;

    T roi_width = roi_xmax - roi_xmin;
    T roi_height = roi_ymax - roi_ymin;
    if (!aligned) {
      roi_width = std::max(roi_width, static_cast<T>(1.));
      roi_height = std::max(roi_height, static_cast<T>(1.));
    }

    const T* batch_data = input_data + roi_batch_id * in_stride[0];

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    auto interpolation_cords = GetIndexesAndRatios(width,
                                                   height,
                                                   roi_width,
                                                   roi_height,
                                                   roi_xmin,
                                                   roi_ymin,
                                                   pooled_width,
                                                   roi_bin_grid_w,
                                                   pooled_height,
                                                   roi_bin_grid_h);

    std::vector<T> interpolated_values;
    interpolated_values.reserve(interpolation_cords.size());
    for (auto channel = 0; channel < channels; ++channel) {
      Interpolate(interpolated_values, interpolation_cords, batch_data);
      AvgPool(interpolated_values,
              output_data,
              roi_bin_grid_w,
              roi_bin_grid_h,
              pooled_width,
              pooled_height);
      batch_data += in_stride[1];
      output_data += out_stride[1];
      interpolated_values.clear();
    }
    boxes_data += roi_stride[0];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_align, CPU, ALL_LAYOUT, phi::RoiAlignKernel, float, double, int) {}
