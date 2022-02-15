/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CPUPSROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto output_channels = ctx.Attr<int>("output_channels");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    PADDLE_ENFORCE_EQ(input_channels,
                      output_channels * pooled_height * pooled_width,
                      platform::errors::InvalidArgument(
                          "the channels of input "
                          "X should equal the product of "
                          "output_channels x pooled_height x pooled_width"));

    auto in_stride = framework::stride(in_dims);
    auto out_stride = framework::stride(out->dims());

    const T* input_data = in->data<T>();

    framework::Tensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num});
    int* rois_batch_id_data =
        rois_batch_id_list.mutable_data<int>(ctx.GetPlace());
    int rois_batch_size;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      auto* rois_num_data = rois_num_t->data<int>();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              rois_batch_size, batch_size));
      int rois_num_count = 0;
      for (int i = 0; i < rois_batch_size; ++i) {
        rois_num_count += rois_num_data[i];
      }
      PADDLE_ENFORCE_EQ(
          rois_num_count, rois_num,
          platform::errors::InvalidArgument(
              "the rois_num from input and RoisNum must be the same"));
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_data[n]; ++i) {
          rois_batch_id_data[i] = n;
        }
        start += rois_num_data[n];
      }
    } else {
      auto rois_lod = rois->lod().back();
      rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument("the rois_batch_size and input(X) "
                                            "batch_size should be the same."));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num_with_lod, rois_num,
          platform::errors::InvalidArgument(
              "the rois_num from input and lod must be the same"));
      // calculate batch id index for each roi according to LoD
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
      }
    }
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* input_rois = rois->data<T>();

    // calculate psroipooling, parallel processing can be implemented per ROI
    for (int n = 0; n < rois_num; ++n) {
      // set roi batch id
      int roi_batch_id = rois_batch_id_data[n];

      // [start, end) interval for spatial sampling
      const T* offset_input_rois = input_rois + n * 4;
      T roi_start_w =
          static_cast<T>(round(offset_input_rois[0])) * spatial_scale;
      T roi_start_h =
          static_cast<T>(round(offset_input_rois[1])) * spatial_scale;
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
      int out_roi_offset = n * out_stride[0];
      for (int c = 0; c < output_channels; ++c) {
        // per category
        int out_plane_offset = out_roi_offset + c * out_stride[1];
        for (int ph = 0; ph < pooled_height; ++ph) {
          int out_row_offset = out_plane_offset + ph * out_stride[2];
          for (int pw = 0; pw < pooled_width; ++pw) {
            // calculate w and h at input feature map
            int hstart = floor(static_cast<T>(ph) * bin_size_h + roi_start_h);
            int wstart = floor(static_cast<T>(pw) * bin_size_w + roi_start_w);
            int hend = ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
            int wend = ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w);
            //  Add roi offsets and clip to input boundaries
            hstart = std::min(std::max(hstart, 0), height);
            wstart = std::min(std::max(wstart, 0), width);
            hend = std::min(std::max(hend, 0), height);
            wend = std::min(std::max(wend, 0), width);

            int output_index = out_row_offset + pw;
            int input_channel = (c * pooled_height + ph) * pooled_width + pw;
            int input_plane_offset =
                roi_batch_id * in_stride[0] + input_channel * in_stride[1];
            const T* offset_input_data = input_data + input_plane_offset;
            T out_sum = 0.;
            bool is_empty = (hend <= hstart) || (wend <= wstart);
            for (int ih = hstart; ih < hend; ++ih) {
              for (int iw = wstart; iw < wend; ++iw) {
                int input_index = ih * in_stride[2] + iw;
                out_sum += offset_input_data[input_index];
              }
            }
            T bin_area = (hend - hstart) * (wend - wstart);
            output_data[output_index] = is_empty ? 0. : out_sum / bin_area;
          }
        }
      }
    }
    return;
  }
};

template <typename DeviceContext, typename T>
class CPUPSROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto output_channels = ctx.Attr<int>("output_channels");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    if (input_grad) {
      auto in_dims = in->dims();
      int input_channels = in_dims[1];
      int height = in_dims[2];
      int width = in_dims[3];
      int rois_num = rois->dims()[0];

      // set roi batch id
      framework::Tensor rois_batch_id_list;
      rois_batch_id_list.Resize({rois_num});
      int* rois_batch_id_data =
          rois_batch_id_list.mutable_data<int>(ctx.GetPlace());
      int rois_batch_size;
      if (ctx.HasInput("RoisNum")) {
        auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
        rois_batch_size = rois_num_t->numel();
        auto* rois_num_data = rois_num_t->data<int>();
        int start = 0;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (int i = start; i < start + rois_num_data[n]; ++i) {
            rois_batch_id_data[i] = n;
          }
          start += rois_num_data[n];
        }
      } else {
        auto rois_lod = rois->lod().back();
        rois_batch_size = rois_lod.size() - 1;
        // calculate batch id index for each roi according to LoD
        for (int n = 0; n < rois_batch_size; ++n) {
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            rois_batch_id_data[i] = n;
          }
        }
      }
      const T* input_rois = rois->data<T>();
      const T* output_grad_data = output_grad->data<T>();
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

      // set gradient of X to be 0. before backpropagate.
      pten::funcs::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.template device_context<DeviceContext>(), input_grad,
               static_cast<T>(0));

      // backpropagate gradient per output pixel
      int output_grad_size = output_grad->numel();
      for (int i = 0; i < output_grad_size; ++i) {
        // The output is in order (n, c, ph, pw)
        int pw = i % pooled_width;
        int ph = (i / pooled_width) % pooled_height;
        int c = (i / pooled_width / pooled_height) % output_channels;
        int n = i / pooled_width / pooled_height / output_channels;

        // set roi_batch_id
        int roi_batch_id = rois_batch_id_data[n];
        int input_channel = (c * pooled_height + ph) * pooled_width + pw;
        int input_offset =
            (roi_batch_id * input_channels + input_channel) * height * width;
        T* offset_input_grad_data = input_grad_data + input_offset;

        // [start, end) interval for spatial sampling
        const T* offset_input_rois = input_rois + n * 4;
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

        int hstart = floor(bin_size_h * static_cast<T>(ph) + roi_start_h);
        int wstart = floor(bin_size_w * static_cast<T>(pw) + roi_start_w);
        int hend = ceil(bin_size_h * static_cast<T>(ph + 1) + roi_start_h);
        int wend = ceil(bin_size_w * static_cast<T>(pw + 1) + roi_start_w);

        // Add roi offsets and clip to input boundaries
        hstart = std::min(std::max(hstart, 0), height);
        hend = std::min(std::max(hend, 0), height);
        wstart = std::min(std::max(wstart, 0), width);
        wend = std::min(std::max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Accumulate diff_val into input data
        T bin_area = static_cast<T>((hend - hstart) * (wend - wstart));
        T diff_val = is_empty ? 0. : output_grad_data[i] / bin_area;
        for (int ih = hstart; ih < hend; ++ih) {
          for (int iw = wstart; iw < wend; ++iw) {
            int input_index = ih * width + iw;
            offset_input_grad_data[input_index] += diff_val;
          }
        }
      }
    }
    return;
  }
};

}  // namespace operators
}  // namespace paddle
