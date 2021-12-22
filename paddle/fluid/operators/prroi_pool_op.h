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
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
inline HOSTDEVICE T PrRoIPoolingGetData(const T* data, const int h, const int w,
                                        const int height, const int width) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  T retVal = overflow ? 0.0f : data[h * width + w];
  return retVal;
}

template <typename T>
inline HOSTDEVICE T PrRoIPoolingMatCalculation(const T* this_data,
                                               const int s_h, const int s_w,
                                               const int e_h, const int e_w,
                                               const T y0, const T x0,
                                               const T y1, const T x1,
                                               const int h0, const int w0) {
  T alpha, beta, lim_alpha, lim_beta, tmp;
  T sum_out = 0;

  alpha = x0 - static_cast<T>(s_w);
  beta = y0 - static_cast<T>(s_h);
  lim_alpha = x1 - static_cast<T>(s_w);
  lim_beta = y1 - static_cast<T>(s_h);
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp;

  alpha = static_cast<T>(e_w) - x1;
  lim_alpha = static_cast<T>(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp;

  alpha = x0 - static_cast<T>(s_w);
  beta = static_cast<T>(e_h) - y1;
  lim_alpha = x1 - static_cast<T>(s_w);
  lim_beta = static_cast<T>(e_h) - y0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp;

  alpha = static_cast<T>(e_w) - x1;
  lim_alpha = static_cast<T>(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp;

  return sum_out;
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
DEVICE void PrRoIPoolingDistributeDiff(T* diff, const T top_diff, const int h,
                                       const int w, const int height,
                                       const int width, const T coeff) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  if (!overflow) {
    paddle::platform::CudaAtomicAdd(diff + h * width + w, top_diff * coeff);
  }
}
#else
template <typename T>
inline HOSTDEVICE void PrRoIPoolingDistributeDiff(T* diff, const T top_diff,
                                                  const int h, const int w,
                                                  const int height,
                                                  const int width,
                                                  const T coeff) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  if (!overflow) {
    *(diff + h * width + w) += top_diff * coeff;
  }
}
#endif

template <typename T>
HOSTDEVICE void PrRoIPoolingMatDistributeDiff(T* diff, const T top_diff,
                                              const int s_h, const int s_w,
                                              const int e_h, const int e_w,
                                              const T y0, const T x0,
                                              const T y1, const T x1,
                                              const int h0, const int w0) {
  T alpha, beta, lim_alpha, lim_beta, tmp;

  alpha = x0 - static_cast<T>(s_w);
  beta = y0 - static_cast<T>(s_h);
  lim_alpha = x1 - static_cast<T>(s_w);
  lim_beta = y1 - static_cast<T>(s_h);
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff<T>(diff, top_diff, s_h, s_w, h0, w0, tmp);

  alpha = static_cast<T>(e_w) - x1;
  lim_alpha = static_cast<T>(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff<T>(diff, top_diff, s_h, e_w, h0, w0, tmp);

  alpha = x0 - static_cast<T>(s_w);
  beta = static_cast<T>(e_h) - y1;
  lim_alpha = x1 - static_cast<T>(s_w);
  lim_beta = static_cast<T>(e_h) - y0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff<T>(diff, top_diff, e_h, s_w, h0, w0, tmp);

  alpha = static_cast<T>(e_w) - x1;
  lim_alpha = static_cast<T>(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff<T>(diff, top_diff, e_h, e_w, h0, w0, tmp);
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
DEVICE void AccumulateRois(T* offset, T data) {
  paddle::platform::CudaAtomicAdd(offset, data);
}
#else
template <typename T>
inline HOSTDEVICE void AccumulateRois(T* offset, T data) {
  *offset += data;
}
#endif

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
DEVICE T MaxFunctor(const T x, const T y) {
  return max(x, y);
}
template <typename T>
DEVICE T MinFunctor(const T x, const T y) {
  return min(x, y);
}
#else
template <typename T>
inline HOSTDEVICE T MaxFunctor(const T x, const T y) {
  return std::max(x, y);
}
template <typename T>
inline HOSTDEVICE T MinFunctor(const T x, const T y) {
  return std::max(x, y);
}
#endif

template <typename T>
inline HOSTDEVICE static T PrRoIPoolingGetCoeff(T dh, T dw) {
  dw = dw > 0 ? dw : -dw;
  dh = dh > 0 ? dh : -dh;
  return (1.0f - dh) * (1.0f - dw);
}

template <typename T, typename H, typename W>
inline HOSTDEVICE static T PrRoIPoolingInterpolation(const T* data, const H h,
                                                     const W w,
                                                     const int height,
                                                     const int width) {
  T retVal = 0.0f;
  int h1 = floorf(h);
  int w1 = floorf(w);
  retVal +=
      PrRoIPoolingGetData(data, h1, w1, height, width) *
      PrRoIPoolingGetCoeff(h - static_cast<T>(h1), w - static_cast<T>(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w);
  retVal +=
      PrRoIPoolingGetData(data, h1, w1, height, width) *
      PrRoIPoolingGetCoeff(h - static_cast<T>(h1), w - static_cast<T>(w1));
  h1 = floorf(h);
  w1 = floorf(w) + 1;
  retVal +=
      PrRoIPoolingGetData(data, h1, w1, height, width) *
      PrRoIPoolingGetCoeff(h - static_cast<T>(h1), w - static_cast<T>(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w) + 1;
  retVal +=
      PrRoIPoolingGetData(data, h1, w1, height, width) *
      PrRoIPoolingGetCoeff(h - static_cast<T>(h1), w - static_cast<T>(w1));
  return retVal;
}

template <typename T>
inline HOSTDEVICE T PrRoIPoolingSingleCoorIntegral(T s, T t, T c1, T c2) {
  return 0.5f * (t * t - s * s) * c2 +
         (t - 0.5f * t * t - s + 0.5f * s * s) * c1;
}

template <typename T>
inline HOSTDEVICE void PrRoIPoolingCoorBackward(
    int s_w, int e_w, int s_h, int e_h, int width, int height, T win_start_w,
    T win_start_h, T win_end_w, T win_end_h, int pw, int ph,
    const int pooled_width, const int pooled_height, T win_size,
    const float spatial_scale, const T* this_bottom_data,
    const T* this_top_data, T* this_data_grad, const T* this_out_grad) {
  T g_x1_y = 0.f;
  T g_x2_y = 0.f;
  T g_x_y1 = 0.f;
  T g_x_y2 = 0.f;

  for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
    g_x1_y += PrRoIPoolingSingleCoorIntegral(
        MaxFunctor<T>(win_start_h, static_cast<T>(h_iter)) - h_iter,
        MinFunctor<T>(win_end_h, static_cast<T>(h_iter + 1)) - h_iter,
        PrRoIPoolingInterpolation(this_bottom_data, h_iter, win_start_w, height,
                                  width),
        PrRoIPoolingInterpolation(this_bottom_data, h_iter + 1, win_start_w,
                                  height, width));

    g_x2_y += PrRoIPoolingSingleCoorIntegral(
        MaxFunctor<T>(win_start_h, static_cast<T>(h_iter)) - h_iter,
        MinFunctor<T>(win_end_h, static_cast<T>(h_iter + 1)) - h_iter,
        PrRoIPoolingInterpolation(this_bottom_data, h_iter, win_end_w, height,
                                  width),
        PrRoIPoolingInterpolation(this_bottom_data, h_iter + 1, win_end_w,
                                  height, width));
  }

  for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
    g_x_y1 += PrRoIPoolingSingleCoorIntegral(
        MaxFunctor<T>(win_start_w, static_cast<T>(w_iter)) - w_iter,
        MinFunctor<T>(win_end_w, static_cast<T>(w_iter + 1)) - w_iter,
        PrRoIPoolingInterpolation(this_bottom_data, win_start_h, w_iter, height,
                                  width),
        PrRoIPoolingInterpolation(this_bottom_data, win_start_h, w_iter + 1,
                                  height, width));

    g_x_y2 += PrRoIPoolingSingleCoorIntegral(
        MaxFunctor<T>(win_start_w, static_cast<T>(w_iter)) - w_iter,
        MinFunctor<T>(win_end_w, static_cast<T>(w_iter + 1)) - w_iter,
        PrRoIPoolingInterpolation(this_bottom_data, win_end_h, w_iter, height,
                                  width),
        PrRoIPoolingInterpolation(this_bottom_data, win_end_h, w_iter + 1,
                                  height, width));
  }

  float partial_x1 = -g_x1_y + (win_end_h - win_start_h) * (*this_top_data);
  float partial_y1 = -g_x_y1 + (win_end_w - win_start_w) * (*this_top_data);
  float partial_x2 = g_x2_y - (win_end_h - win_start_h) * (*this_top_data);
  float partial_y2 = g_x_y2 - (win_end_w - win_start_w) * (*this_top_data);

  partial_x1 = partial_x1 / win_size * spatial_scale;
  partial_x2 = partial_x2 / win_size * spatial_scale;
  partial_y1 = partial_y1 / win_size * spatial_scale;
  partial_y2 = partial_y2 / win_size * spatial_scale;

  AccumulateRois<T>(
      this_data_grad + 0,
      (partial_x1 * (1.0 - static_cast<T>(pw) / pooled_width) +
       partial_x2 * (1.0 - static_cast<T>(pw + 1) / pooled_width)) *
          (*this_out_grad));
  AccumulateRois<T>(
      this_data_grad + 1,
      (partial_y1 * (1.0 - static_cast<T>(ph) / pooled_height) +
       partial_y2 * (1.0 - static_cast<T>(ph + 1) / pooled_height)) *
          (*this_out_grad));
  AccumulateRois<T>(this_data_grad + 2,
                    (partial_x2 * static_cast<T>(pw + 1) / pooled_width +
                     partial_x1 * static_cast<T>(pw) / pooled_width) *
                        (*this_out_grad));
  AccumulateRois<T>(this_data_grad + 3,
                    (partial_y2 * static_cast<T>(ph + 1) / pooled_height +
                     partial_y1 * static_cast<T>(ph) / pooled_height) *
                        (*this_out_grad));
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
    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    auto output_channels = input_channels;
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];
    if (rois_num == 0) return;

    auto in_stride = framework::stride(in_dims);
    auto out_stride = framework::stride(out->dims());

    const T* input_data = in->data<T>();

    framework::Tensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num});
    int* rois_batch_id_data =
        rois_batch_id_list.mutable_data<int>(ctx.GetPlace());
    if (ctx.HasInput("BatchRoINums") || rois->lod().empty()) {
      auto* batchroinum = ctx.Input<framework::Tensor>("BatchRoINums");
      auto* batch_index = batchroinum->data<int64_t>();
      int rois_batch_size = batchroinum->dims()[0];
      size_t c = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int64_t k = 0; k < batch_index[n]; ++k) {
          rois_batch_id_data[c] = n;
          c = c + 1;
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(rois->lod().empty(), false,
                        platform::errors::InvalidArgument(
                            "The lod of Input ROIs should not be empty when "
                            "BatchRoINums is None!"));
      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(rois_batch_size, batch_size,
                        platform::errors::InvalidArgument(
                            "The rois_batch_size and input(X)'s "
                            "batch_size should be the same but received"
                            "rois_batch_size: %d and batch_size: %d",
                            rois_batch_size, batch_size));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num_with_lod, rois_num,
          platform::errors::InvalidArgument("The rois_num from input should be "
                                            "equal to the rois_num from lod, "
                                            "but received rois_num from input: "
                                            "%d and the rois_num from lod: %d.",
                                            rois_num_with_lod, rois_num));

      // calculate batch id index for each roi according to LoD
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
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
      T roi_start_w = static_cast<T>(offset_input_rois[0]) * spatial_scale;
      T roi_start_h = static_cast<T>(offset_input_rois[1]) * spatial_scale;
      T roi_end_w = static_cast<T>(offset_input_rois[2]) * spatial_scale;
      T roi_end_h = static_cast<T>(offset_input_rois[3]) * spatial_scale;

      T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(0.0));
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.0));

      // Compute w and h at input feature map
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);
      T win_size = std::max(static_cast<T>(0.0), bin_size_w * bin_size_h);

      // calculate each pixel of the output feature map.
      int out_roi_offset = n * out_stride[0];
      for (int c = 0; c < output_channels; ++c) {
        // per category
        int out_plane_offset = out_roi_offset + c * out_stride[1];
        for (int ph = 0; ph < pooled_height; ++ph) {
          int out_row_offset = out_plane_offset + ph * out_stride[2];
          for (int pw = 0; pw < pooled_width; ++pw) {
            // calculate w and h at input feature map
            T win_start_h = static_cast<T>(ph) * bin_size_h + roi_start_h;
            T win_start_w = static_cast<T>(pw) * bin_size_w + roi_start_w;
            T win_end_h = win_start_h + bin_size_h;
            T win_end_w = win_start_w + bin_size_w;
            //  Add roi offsets and clip to input boundaries
            int s_w = std::floor(win_start_w);
            int e_w = std::ceil(win_end_w);
            int s_h = std::floor(win_start_h);
            int e_h = std::ceil(win_end_h);

            int output_index = out_row_offset + pw;
            int input_channel = c;
            int input_plane_offset =
                roi_batch_id * in_stride[0] + input_channel * in_stride[1];
            const T* offset_input_data = input_data + input_plane_offset;
            T sum_out = 0.;

            if (win_size > static_cast<T>(0.0)) {
              for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
                for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
                  sum_out += PrRoIPoolingMatCalculation(
                      offset_input_data, h_iter, w_iter, h_iter + 1, w_iter + 1,
                      std::max(win_start_h, static_cast<T>(h_iter)),
                      std::max(win_start_w, static_cast<T>(w_iter)),
                      std::min(win_end_h,
                               static_cast<T>(h_iter) + static_cast<T>(1.0)),
                      std::min(win_end_w,
                               static_cast<T>(w_iter) + static_cast<T>(1.0)),
                      height, width);
                }
              }

              output_data[output_index] = sum_out / win_size;
            } else {
              output_data[output_index] = 0.;
            }
          }
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class CPUPRROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Input<framework::Tensor>("Out");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* input_roi_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("ROIs"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    if (input_grad || input_roi_grad) {
      auto in_dims = in->dims();
      auto* in_data = in->data<T>();
      auto* out_data = out->data<T>();

      int input_channels = in_dims[1];
      auto output_channels = input_channels;
      int height = in_dims[2];
      int width = in_dims[3];
      int rois_num = rois->dims()[0];

      // set roi batch id
      framework::Tensor rois_batch_id_list;
      rois_batch_id_list.Resize({rois_num});
      int* rois_batch_id_data =
          rois_batch_id_list.mutable_data<int>(ctx.GetPlace());
      if (ctx.HasInput("BatchRoINums") || rois->lod().empty()) {
        auto* batchroinum = ctx.Input<framework::Tensor>("BatchRoINums");
        auto* batch_index = batchroinum->data<int64_t>();
        int rois_batch_size = batchroinum->dims()[0];
        size_t c = 0;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (int64_t k = 0; k < batch_index[n]; ++k) {
            rois_batch_id_data[c] = n;
            c = c + 1;
          }
        }
      } else {
        auto rois_lod = rois->lod().back();
        int rois_batch_size = rois_lod.size() - 1;
        // calculate batch id index for each roi according to LoD
        for (int n = 0; n < rois_batch_size; ++n) {
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            rois_batch_id_data[i] = n;
          }
        }
      }

      const T* input_rois = rois->data<T>();
      const T* output_grad_data = output_grad->data<T>();

      input_grad->mutable_data<T>(ctx.GetPlace());
      input_roi_grad->mutable_data<T>(ctx.GetPlace());
      // set gradient of X to be 0. before backpropagate.
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.template device_context<DeviceContext>(), input_grad,
               static_cast<T>(0));
      set_zero(ctx.template device_context<DeviceContext>(), input_roi_grad,
               static_cast<T>(0));

      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      T* input_roi_grad_data = input_roi_grad->mutable_data<T>(ctx.GetPlace());

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
        int input_channel = c;
        int input_offset =
            (roi_batch_id * input_channels + input_channel) * height * width;
        T* offset_input_grad_data = input_grad_data + input_offset;
        const T* offset_output_grad_data = output_grad_data + i;
        const T* offset_out_data = out_data + i;

        // [start, end) interval for spatial sampling
        const T* offset_input_rois = input_rois + n * 4;
        T roi_start_w = static_cast<T>(offset_input_rois[0]) * spatial_scale;
        T roi_start_h = static_cast<T>(offset_input_rois[1]) * spatial_scale;
        T roi_end_w = static_cast<T>(offset_input_rois[2]) * spatial_scale;
        T roi_end_h = static_cast<T>(offset_input_rois[3]) * spatial_scale;
        T* offset_input_roi_grad_data = input_roi_grad_data + n * 4;

        T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(0.0));
        T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.0));

        // Compute w and h at input feature map
        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        T win_start_w = roi_start_w + bin_size_w * pw;
        T win_start_h = roi_start_h + bin_size_h * ph;
        T win_end_w = win_start_w + bin_size_w;
        T win_end_h = win_start_h + bin_size_h;

        T win_size = std::max(static_cast<T>(0.0), bin_size_w * bin_size_h);

        T sum_out = win_size == static_cast<T>(0.)
                        ? static_cast<T>(0.)
                        : *offset_output_grad_data / win_size;

        int s_w = std::floor(win_start_w);
        int e_w = std::ceil(win_end_w);
        int s_h = std::floor(win_start_h);
        int e_h = std::ceil(win_end_h);

        for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
          for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
            PrRoIPoolingMatDistributeDiff<T>(
                offset_input_grad_data, sum_out, h_iter, w_iter, h_iter + 1,
                w_iter + 1, std::max(win_start_h, static_cast<T>(h_iter)),
                std::max(win_start_w, static_cast<T>(w_iter)),
                std::min(win_end_h,
                         static_cast<T>(h_iter) + static_cast<T>(1.0)),
                std::min(win_end_w,
                         static_cast<T>(w_iter) + static_cast<T>(1.0)),
                height, width);
          }
        }

        const T* offset_in_data = in_data + input_offset;
        PrRoIPoolingCoorBackward<T>(
            s_w, e_w, s_h, e_h, width, height, win_start_w, win_start_h,
            win_end_w, win_end_h, pw, ph, pooled_width, pooled_height, win_size,
            spatial_scale, offset_in_data, offset_out_data,
            offset_input_roi_grad_data, offset_output_grad_data);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
