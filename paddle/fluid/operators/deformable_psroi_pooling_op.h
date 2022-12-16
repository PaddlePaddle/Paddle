// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
//
// Part of the following code in this file refs to
// https://github.com/msracver/Deformable-ConvNets/blob/master/faster_rcnn/operator_cxx/deformable_psroi_pooling.cu
//
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// \file deformable_psroi_pooling.cu
// \brief
// \author Yi Li, Guodong Zhang, Jifeng Dai

#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
T bilinear_interp(
    const T* data, const T x, const T y, const int width, const int height) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  T dist_x = static_cast<T>(x - x1);
  T dist_y = static_cast<T>(y - y1);
  T value11 = data[y1 * width + x1];
  T value12 = data[y2 * width + x1];
  T value21 = data[y1 * width + x2];
  T value22 = data[y2 * width + x2];
  T value = (1 - dist_x) * (1 - dist_y) * value11 +
            (1 - dist_x) * dist_y * value12 + dist_x * (1 - dist_y) * value21 +
            dist_x * dist_y * value22;
  return value;
}

template <typename T>
void DeformablePSROIPoolForwardCPUKernel(const int count,
                                         const T* bottom_data,
                                         const T spatial_scale,
                                         const int channels,
                                         const int height,
                                         const int width,
                                         const int pooled_height,
                                         const int pooled_width,
                                         const T* bottom_rois,
                                         const T* bottom_trans,
                                         const bool no_trans,
                                         const float trans_std,
                                         const int sample_per_part,
                                         const int output_dim,
                                         const int group_height,
                                         const int group_width,
                                         const int part_height,
                                         const int part_width,
                                         const int num_classes,
                                         const int channels_each_class,
                                         T* top_data,
                                         T* top_count,
                                         const int batch_size,
                                         int* roi_batch_id_data,
                                         const phi::DenseTensor* rois) {
  for (int ix = 0; ix < count; ix++) {
    int pw = ix % pooled_width;
    int ph = (ix / pooled_width) % pooled_height;
    int ctop = (ix / pooled_width / pooled_height) % output_dim;
    int n = ix / pooled_width / pooled_height / output_dim;
    const T* offset_bottom_rois = bottom_rois + n * 4;

    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w =
        static_cast<T>(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h =
        static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w =
        static_cast<T>(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h =
        static_cast<T>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;

    //  width and height of roi
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1));
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));

    //  width and height of each bin
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    //  sampling interval in each bin
    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

    //  obtain offset of roi
    int part_h = floor(static_cast<T>(ph) / pooled_height * part_height);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_width);
    int class_id = ctop / channels_each_class;

    T trans_x =
        no_trans
            ? static_cast<T>(0)
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_height +
                            part_h) *
                               part_width +
                           part_w] *
                  static_cast<T>(trans_std);
    T trans_y = no_trans
                    ? static_cast<T>(0)
                    : bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                                        part_height +
                                    part_h) *
                                       part_width +
                                   part_w] *
                          static_cast<T>(trans_std);

    //  location of start after adding offset
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    T sum = 0;
    int num_sample = 0;
    int gw = floor(static_cast<T>(pw) * group_width / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_height / pooled_height);
    gw = std::min(std::max(gw, 0), group_width - 1);
    gh = std::min(std::max(gh, 0), group_height - 1);
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels) * height * width;

    //  sampling in each bin
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = std::min(std::max(w, T(0.)), T(width - 1.));
        h = std::min(std::max(h, T(0.)), height - T(1.));
        int c = (ctop * group_height + gh) * group_width + gw;
        // bilinear interpolation to get value
        T val = bilinear_interp(
            offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        num_sample++;
      }
    }
    top_data[ix] = num_sample == 0 ? static_cast<T>(0) : sum / num_sample;
    top_count[ix] = num_sample;
  }
}

template <typename DeviceContext, typename T>
class DeformablePSROIPoolCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* rois = ctx.Input<phi::DenseTensor>("ROIs");
    auto* trans = ctx.Input<phi::DenseTensor>("Trans");
    auto* out = ctx.Output<phi::DenseTensor>("Output");
    out->mutable_data<T>(ctx.GetPlace());
    auto* top_count = ctx.Output<phi::DenseTensor>("TopCount");
    top_count->mutable_data<T>(ctx.GetPlace());

    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    set_zero(dev_ctx, out, static_cast<T>(0));
    set_zero(dev_ctx, top_count, static_cast<T>(0));

    const int num_rois = rois->dims()[0];
    PADDLE_ENFORCE_EQ(
        num_rois,
        out->dims()[0],
        platform::errors::InvalidArgument(
            "The number of Input(ROIs) should be same with the number of "
            "Output(Output), but received ROIs number is:%d, Output number "
            "is:%d.",
            num_rois,
            out->dims()[0]));
    phi::DenseTensor roi_batch_id_list;
    roi_batch_id_list.Resize({num_rois});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());
    auto no_trans = ctx.Attr<bool>("no_trans");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto output_dim = ctx.Attr<int>("output_dim");
    auto group_size = ctx.Attr<std::vector<int>>("group_size");
    auto group_height = group_size[0];
    auto group_width = group_size[1];
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto part_size = ctx.Attr<std::vector<int>>("part_size");
    auto part_height = part_size[0];
    auto part_width = part_size[1];
    auto sample_per_part = ctx.Attr<int>("sample_per_part");
    auto trans_std = ctx.Attr<float>("trans_std");

    int batch = static_cast<int>(input->dims()[0]);
    int channels = static_cast<int>(input->dims()[1]);
    int height = static_cast<int>(input->dims()[2]);
    int width = static_cast<int>(input->dims()[3]);
    int channels_trans = no_trans ? 2 : trans->dims()[1];
    auto count = num_rois * output_dim * pooled_height * pooled_width;
    auto num_classes = no_trans ? 1 : channels_trans / 2;
    auto channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    PADDLE_ENFORCE_GE(channels_each_class,
                      1,
                      platform::errors::InvalidArgument(
                          "channels_each_class should not be lower than 1, but "
                          "channels_each_class is:%d.",
                          channels_each_class));

    const T* bottom_data = input->data<T>();
    const T* bottom_rois = rois->data<T>();
    const T* bottom_trans = no_trans ? NULL : trans->data<T>();

    T* top_data = out->mutable_data<T>(ctx.GetPlace());
    T* top_count_data = top_count->mutable_data<T>(ctx.GetPlace());

    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch,
        platform::errors::InvalidArgument(
            "rois_batch_size should be equal to the batch_size, but "
            "rois_batch_size is:%d, batch_size is:%d.",
            rois_batch_size,
            batch));
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_rois,
                      rois_num_with_lod,
                      platform::errors::InvalidArgument(
                          "The rois_num from input and lod must be same, but"
                          "rois_num from input is:%d, rois_num from lod is:%d.",
                          num_rois,
                          rois_num_with_lod));
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }

    DeformablePSROIPoolForwardCPUKernel(count,
                                        bottom_data,
                                        (T)spatial_scale,
                                        channels,
                                        height,
                                        width,
                                        pooled_height,
                                        pooled_width,
                                        bottom_rois,
                                        bottom_trans,
                                        no_trans,
                                        trans_std,
                                        sample_per_part,
                                        output_dim,
                                        group_height,
                                        group_width,
                                        part_height,
                                        part_width,
                                        num_classes,
                                        channels_each_class,
                                        top_data,
                                        top_count_data,
                                        batch,
                                        roi_batch_id_data,
                                        rois);
  }
};

template <typename T>
void DeformablePSROIPoolBackwardAccCPUKernel(const int count,
                                             const T* top_diff,
                                             const T* top_count,
                                             const int num_rois,
                                             const T spatial_scale,
                                             const int channels,
                                             const int height,
                                             const int width,
                                             const int pooled_height,
                                             const int pooled_width,
                                             const int output_dim,
                                             T* bottom_data_diff,
                                             T* bottom_trans_diff,
                                             const T* bottom_data,
                                             const T* bottom_rois,
                                             const T* bottom_trans,
                                             const bool no_trans,
                                             const float trans_std,
                                             const int sample_per_part,
                                             const int group_height,
                                             const int group_width,
                                             const int part_height,
                                             const int part_width,
                                             const int num_classes,
                                             const int channels_each_class,
                                             const int batch_size,
                                             int* roi_batch_id_data,
                                             const phi::DenseTensor* rois) {
  for (int index = 0; index < count; index++) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    //  location of roi on feature map
    const T* offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w =
        static_cast<T>(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h =
        static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w =
        static_cast<T>(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h =
        static_cast<T>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;

    //  width and height of roi
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1));
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));

    //  width and height of each bin
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    //  sampling interval in each bin
    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

    //  obtain offset of roi
    int part_h = floor(static_cast<T>(ph) / pooled_height * part_height);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_height);
    int class_id = ctop / channels_each_class;

    T trans_x =
        no_trans
            ? static_cast<T>(0)
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_height +
                            part_h) *
                               part_width +
                           part_w] *
                  static_cast<T>(trans_std);
    T trans_y = no_trans
                    ? static_cast<T>(0)
                    : bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                                        part_height +
                                    part_h) *
                                       part_width +
                                   part_w] *
                          static_cast<T>(trans_std);

    //  location of start after adding offset
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0) {
      continue;
    }

    T diff_val = top_diff[index] / top_count[index];
    const T* offset_bottom_data =
        bottom_data + roi_batch_ind * channels * height * width;
    int gw = floor(static_cast<T>(pw) * group_width / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_height / pooled_height);
    gw = std::min(std::max(gw, 0), group_width - 1);
    gh = std::min(std::max(gh, 0), group_height - 1);

    //  sampling in each bin
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = std::min(std::max(w, T(0.)), T(width - 1.));
        h = std::min(std::max(h, T(0.)), T(height - 1.));
        int c = (ctop * group_height + gh) * group_width + gw;
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);

        //  compute coefficient of gradient
        T dist_x = w - x0, dist_y = h - y0;
        T q00 = (1 - dist_x) * (1 - dist_y);
        T q01 = (1 - dist_x) * dist_y;
        T q10 = dist_x * (1 - dist_y);
        T q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;

        //  compute gradient of input
        if (bottom_data_diff != NULL) {
          T* offset_bottom_data_diff_addr00 =
              bottom_data_diff + roi_batch_ind * channels * height * width +
              bottom_index_base + y0 * width + x0;
          T* offset_bottom_data_diff_addr01 =
              bottom_data_diff + roi_batch_ind * channels * height * width +
              bottom_index_base + y1 * width + x0;
          T* offset_bottom_data_diff_addr10 =
              bottom_data_diff + roi_batch_ind * channels * height * width +
              bottom_index_base + y0 * width + x1;
          T* offset_bottom_data_diff_addr11 =
              bottom_data_diff + roi_batch_ind * channels * height * width +
              bottom_index_base + y1 * width + x1;
          *offset_bottom_data_diff_addr00 =
              *offset_bottom_data_diff_addr00 + q00 * diff_val;
          *offset_bottom_data_diff_addr01 =
              *offset_bottom_data_diff_addr01 + q01 * diff_val;
          *offset_bottom_data_diff_addr10 =
              *offset_bottom_data_diff_addr10 + q10 * diff_val;
          *offset_bottom_data_diff_addr11 =
              *offset_bottom_data_diff_addr11 + q11 * diff_val;
        }

        //  compute gradient of trans
        if (no_trans || bottom_trans_diff == NULL) {
          continue;
        }

        T u00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        T u01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        T u10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        T u11 = offset_bottom_data[bottom_index_base + y1 * width + x1];

        T diff_x = (u11 * dist_y + u10 * (1 - dist_y) - u01 * dist_y -
                    u00 * (1 - dist_y)) *
                   trans_std * diff_val;
        diff_x *= roi_width;
        T diff_y = (u11 * dist_x + u01 * (1 - dist_x) - u10 * dist_x -
                    u00 * (1 - dist_x)) *
                   trans_std * diff_val;
        diff_y *= roi_height;
        T* offset_bottom_trans_diff_x =
            bottom_trans_diff +
            (((n * num_classes + class_id) * 2) * part_height + part_h) *
                part_width +
            part_w;
        T* offset_bottom_trans_diff_y =
            bottom_trans_diff +
            (((n * num_classes + class_id) * 2 + 1) * part_height + part_h) *
                part_width +
            part_w;

        *offset_bottom_trans_diff_x = *offset_bottom_trans_diff_x + diff_x;
        *offset_bottom_trans_diff_y = *offset_bottom_trans_diff_y + diff_y;
      }
    }
  }
}

template <typename DeviceContext, typename T>
class DeformablePSROIPoolGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* rois = ctx.Input<phi::DenseTensor>("ROIs");
    auto* trans = ctx.Input<phi::DenseTensor>("Trans");
    auto* top_count = ctx.Input<phi::DenseTensor>("TopCount");
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Output"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, input_grad, static_cast<T>(.0));
    }
    auto* trans_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Trans"));
    if (trans_grad) {
      trans_grad->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, trans_grad, static_cast<T>(.0));
    }
    auto no_trans = ctx.Attr<bool>("no_trans");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto output_dim = ctx.Attr<int>("output_dim");
    auto group_size = ctx.Attr<std::vector<int>>("group_size");
    auto group_height = group_size[0];
    auto group_width = group_size[1];
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto part_size = ctx.Attr<std::vector<int>>("part_size");
    auto part_height = part_size[0];
    auto part_width = part_size[1];
    auto sample_per_part = ctx.Attr<int>("sample_per_part");
    auto trans_std = ctx.Attr<float>("trans_std");

    const int batch = static_cast<int>(input->dims()[0]);
    const int channels = static_cast<int>(input->dims()[1]);
    const int height = static_cast<int>(input->dims()[2]);
    const int width = static_cast<int>(input->dims()[3]);
    const int channels_trans = no_trans ? 2 : trans->dims()[1];
    const int num_rois = rois->dims()[0];
    const int count = num_rois * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class =
        no_trans ? output_dim : output_dim / num_classes;
    phi::DenseTensor roi_batch_id_list;
    roi_batch_id_list.Resize({num_rois});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

    const T* top_diff = output_grad->data<T>();
    const T* bottom_data = input->data<T>();
    const T* bottom_rois = rois->data<T>();
    const T* bottom_trans = no_trans ? NULL : trans->data<T>();

    T* bottom_data_diff = NULL;
    T* bottom_trans_diff = NULL;
    if (input_grad) {
      bottom_data_diff = input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (trans_grad) {
      bottom_trans_diff =
          no_trans ? NULL : trans_grad->mutable_data<T>(ctx.GetPlace());
    }

    const T* top_count_data = top_count->data<T>();
    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_rois,
                      rois_num_with_lod,
                      platform::errors::InvalidArgument(
                          "The rois_num from input and lod must be same, but"
                          "rois_num from input is:%d, rois_num from lod is:%d.",
                          num_rois,
                          rois_num_with_lod));
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }

    DeformablePSROIPoolBackwardAccCPUKernel(count,
                                            top_diff,
                                            top_count_data,
                                            num_rois,
                                            (T)spatial_scale,
                                            channels,
                                            height,
                                            width,
                                            pooled_height,
                                            pooled_width,
                                            output_dim,
                                            bottom_data_diff,
                                            bottom_trans_diff,
                                            bottom_data,
                                            bottom_rois,
                                            bottom_trans,
                                            no_trans,
                                            (T)trans_std,
                                            sample_per_part,
                                            group_height,
                                            group_width,
                                            part_height,
                                            part_width,
                                            num_classes,
                                            channels_each_class,
                                            batch,
                                            roi_batch_id_data,
                                            rois);
  }
};

}  // namespace operators
}  // namespace paddle
