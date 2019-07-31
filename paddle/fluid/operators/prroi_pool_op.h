/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
HOSTDEVICE T PrRoIPoolingGetData(const T* data, const int h, const int w,
                                 const int height, const int width) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  T retVal = overflow ? 0.0f : data[h * width + w];
  return retVal;
}

template <typename T>
HOSTDEVICE T PrRoIPoolingGetCoeff(T dh, T dw) {
  dw = dw > 0 ? dw : -dw;
  dh = dh > 0 ? dh : -dh;
  return (1.0f - dh) * (1.0f - dw);
}

template <typename T>
HOSTDEVICE T PrRoIPoolingSingleCoorIntegral(T s, T t, T c1, T c2) {
  return 0.5 * (t * t - s * s) * c2 + (t - 0.5 * t * t - s + 0.5 * s * s) * c1;
}

template <typename T>
HOSTDEVICE T PrRoIPoolingInterpolation(const T* data, const T h, const T w,
                                       const int height, const int width) {
  T retVal = 0.0f;
  int h1 = floorf(h);
  int w1 = floorf(w);
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w);
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h);
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  return retVal;
}

template <typename T>
HOSTDEVICE T PrRoIPoolingMatCalculation(const T* this_data, const int s_h,
                                        const int s_w, const int e_h,
                                        const int e_w, const T y0, const T x0,
                                        const T y1, const T x1, const int h0,
                                        const int w0) {
  T alpha, beta, lim_alpha, lim_beta, tmp;
  T sum_out = 0;

  alpha = x0 - T(s_w);
  beta = y0 - T(s_h);
  lim_alpha = x1 - T(s_w);
  lim_beta = y1 - T(s_h);
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp;

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp;

  alpha = x0 - T(s_w);
  beta = T(e_h) - y1;
  lim_alpha = x1 - T(s_w);
  lim_beta = T(e_h) - y0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp;

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp;

  return sum_out;
}

template <typename DeviceContext, typename T>
class CPUPRROIPoolOpKernel : public framework::OpKernel<T> {
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

    auto in_stride = framework::stride(in_dims);
    auto out_stride = framework::stride(out->dims());

    const T* input_data = in->data<T>();

    framework::Tensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num});
    int* rois_batch_id_data =
        rois_batch_id_list.mutable_data<int>(ctx.GetPlace());

    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size, batch_size,
        "the rois_batch_size and input(X) batch_size should be the same.");
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(rois_num_with_lod, rois_num,
                      "the rois_num from input and lod must be the same");

    PADDLE_ENFORCE_EQ(input_channels,
                      output_channels * pooled_height * pooled_width,
                      "the channels of input X should equal the product of "
                      "output_channels x pooled_height x pooled_width");

    // calculate batch id index for each roi according to LoD
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        rois_batch_id_data[i] = n;
      }
    }

    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* input_rois = rois->data<T>();

    // calculate prroipooling, parallel processing can be implemented per ROI
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
            for (int h_iter = hstart; h_iter < hend; ++h_iter) {
              for (int w_iter = wstart; w_iter < wend; ++w_iter) {
                out_sum += PrRoIPoolingMatCalculation(
                    offset_input_data, h_iter, w_iter, h_iter + 1, w_iter + 1,
                    std::max(hstart, T(h_iter)), std::max(wstart, T(w_iter)),
                    std::min(hend, T(h_iter) + 1.0),
                    std::min(wend, T(w_iter + 1.0)), height, width);
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
class CPUPRROIPoolGradOpKernel : public framework::OpKernel<T> {
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
      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      // calculate batch id index for each roi according to LoD
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
      }

      const T* input_rois = rois->data<T>();
      const T* output_grad_data = output_grad->data<T>();
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

      // set gradient of X to be 0. before backpropagate.
      math::SetConstant<DeviceContext, T> set_zero;
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
