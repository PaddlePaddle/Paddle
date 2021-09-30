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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
T bezier_curve(const T p0, const T p1, const T p2, const T p3, const T u) {
  return ((1. - u) * (1. - u) * (1. - u) * p0 +
          3. * u * (1. - u) * (1. - u) * p1 + 3. * u * u * (1. - u) * p2 +
          u * u * u * p3);
}

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int iy_upper, const int ix_upper, T p0_x,
    T p0_y, T p1_x, T p1_y, T p2_x, T p2_y, T p3_x, T p3_y, T p4_x, T p4_y,
    T p5_x, T p5_y, T p6_x, T p6_y, T p7_x, T p7_y, T bin_size_h, T bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      // compute the coords
      const T u = pw / static_cast<T>(pooled_width);
      const T v = ph / static_cast<T>(pooled_height);
      const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
      const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
      const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
      const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
      const T x_center = x1 * v + x0 * (1. - v);
      const T y_center = y1 * v + y0 * (1. - v);
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = y_center - (T)0.5 * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                         static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = x_center - (T)0.5 * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                           static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename DeviceContext, typename T>
class CPUBezierAlignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<float>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    auto output_channels = input_channels;
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];
    if (rois_num == 0) return;

    const T* input = in->data<T>();

    T* output = out->mutable_data<T>(ctx.GetPlace());

    // get roi batch id
    int rois_batch_size;
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

    const T* input_rois = rois->data<T>();
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              rois_batch_size, batch_size));

      auto* rois_num_data = rois_num_t->data<int>();
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_data[n];
      }
    }

    for (int n = 0; n < rois_num; n++) {
      int index_n = n * output_channels * pooled_width * pooled_height;
      const T* offset_rois = input_rois + n * 16;
      int roi_batch_ind = roi_batch_id_data[n];
      T offset = aligned ? (T)0.5 : (T)0.0;
      // Do not use rounding; this implementation detail is critical
      T p0_x = offset_rois[0] * spatial_scale - offset;
      T p0_y = offset_rois[1] * spatial_scale - offset;
      T p1_x = offset_rois[2] * spatial_scale - offset;
      T p1_y = offset_rois[3] * spatial_scale - offset;
      T p2_x = offset_rois[4] * spatial_scale - offset;
      T p2_y = offset_rois[5] * spatial_scale - offset;
      T p3_x = offset_rois[6] * spatial_scale - offset;
      T p3_y = offset_rois[7] * spatial_scale - offset;
      T p4_x = offset_rois[14] * spatial_scale - offset;
      T p4_y = offset_rois[15] * spatial_scale - offset;
      T p5_x = offset_rois[12] * spatial_scale - offset;
      T p5_y = offset_rois[13] * spatial_scale - offset;
      T p6_x = offset_rois[10] * spatial_scale - offset;
      T p6_y = offset_rois[11] * spatial_scale - offset;
      T p7_x = offset_rois[8] * spatial_scale - offset;
      T p7_y = offset_rois[9] * spatial_scale - offset;

      T roi_width = std::max(std::abs(p0_x - p3_x), std::abs(p4_x - p7_x));
      T roi_height = std::max(std::abs(p0_y - p3_y), std::abs(p4_y - p7_y));
      if (aligned) {
        PADDLE_ENFORCE_GT(roi_width, 0,
                          platform::errors::InvalidArgument(
                              "The 'roi_width' attribute in BezierAlignOp is "
                              "invalid. The height must be greater than 0. But "
                              "received 'roi_width' = %d",
                              roi_width));
        PADDLE_ENFORCE_GT(roi_height, 0,
                          platform::errors::InvalidArgument(
                              "The 'roi_height' attribute in BezierAlignOp is "
                              "invalid. The width must be greater than 0. But "
                              "received 'roi_height' = %d",
                              roi_height));
      } else {  // for backward-compatibility only
        roi_width = std::max(roi_width, (T)1.);
        roi_height = std::max(roi_height, (T)1.);
      }
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      // When the grid is empty, output zeros == 0/1, instead of NaN.
      const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

      // we want to precalculate indices and weights shared by all channels,
      // this is the key point of optimization
      std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                       pooled_width * pooled_height);
      pre_calc_for_bilinear_interpolate(
          height, width, pooled_height, pooled_width, roi_bin_grid_h,
          roi_bin_grid_w, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x,
          p4_y, p5_x, p5_y, p6_x, p6_y, p7_x, p7_y, bin_size_h, bin_size_w,
          roi_bin_grid_h, roi_bin_grid_w, pre_calc);

      for (int c = 0; c < input_channels; c++) {
        int index_n_c = index_n + c * pooled_width * pooled_height;
        const T* offset_input =
            input + (roi_batch_ind * input_channels + c) * height * width;
        int pre_calc_index = 0;

        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            int index = index_n_c + ph * pooled_width + pw;

            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc<T> pc = pre_calc[pre_calc_index];
                output_val += pc.w1 * offset_input[pc.pos1] +
                              pc.w2 * offset_input[pc.pos2] +
                              pc.w3 * offset_input[pc.pos3] +
                              pc.w4 * offset_input[pc.pos4];

                pre_calc_index += 1;
              }
            }
            output_val /= count;

            output[index] = output_val;

          }  // for pw
        }    // for ph
      }      // for c
    }        // for n
  }
};

template <typename T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   T& w1, T& w2, T& w3, T& w4, int& x_low,
                                   int& x_high, int& y_low, int& y_high,
                                   const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename DeviceContext, typename T>
class CPUBezierAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<float>("sampling_ratio");
    auto in_dims = in->dims();
    auto aligned = ctx.Attr<bool>("aligned");

    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    if (!in_grad) {
      return;
    }
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

    int rois_batch_size;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<framework::Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      auto* rois_num_data = rois_num_t->data<int>();
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_data[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_data[n];
      }
    }  // end if has input

    in_grad->mutable_data<T>(ctx.GetPlace());
    // input_roi_grad->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, in_grad, static_cast<T>(0));
    // set_zero(dev_ctx, input_roi_grad, static_cast<T>(0));

    int output_grad_size = out_grad->numel();
    if ((!out_grad->IsInitialized()) || (output_grad_size <= 0)) {
      return;
    }

    const T* input_rois = rois->data<T>();
    const T* grad_output = out_grad->data<T>();

    T* input_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());
    // T* input_roi_grad_data = input_roi_grad->mutable_data<T>(ctx.GetPlace());

    // get stride values to ensure indexing into gradients is correct.
    auto out_stride = framework::stride(out_grad->dims());
    int n_stride = out_stride[0];
    int c_stride = out_stride[1];
    int h_stride = out_stride[2];
    int w_stride = out_stride[3];

    // backpropagate gradient per output pixel
    for (int index = 0; index < output_grad_size; ++index) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      // set roi_batch_id
      int roi_batch_ind = roi_batch_id_data[n];
      const T* offset_rois = input_rois + n * 16;
      // Do not use rounding; this implementation detail is critical
      T offset = aligned ? (T)0.5 : (T)0.0;
      T p0_x = offset_rois[0] * spatial_scale;
      T p0_y = offset_rois[1] * spatial_scale;
      T p1_x = offset_rois[2] * spatial_scale;
      T p1_y = offset_rois[3] * spatial_scale;
      T p2_x = offset_rois[4] * spatial_scale;
      T p2_y = offset_rois[5] * spatial_scale;
      T p3_x = offset_rois[6] * spatial_scale;
      T p3_y = offset_rois[7] * spatial_scale;
      T p4_x = offset_rois[14] * spatial_scale;
      T p4_y = offset_rois[15] * spatial_scale;
      T p5_x = offset_rois[12] * spatial_scale;
      T p5_y = offset_rois[13] * spatial_scale;
      T p6_x = offset_rois[10] * spatial_scale;
      T p6_y = offset_rois[11] * spatial_scale;
      T p7_x = offset_rois[8] * spatial_scale;
      T p7_y = offset_rois[9] * spatial_scale;

      // compute the coords
      const T u = pw / static_cast<T>(pooled_width);
      const T v = ph / static_cast<T>(pooled_height);
      const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
      const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
      const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
      const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
      const T x_center = x1 * v + x0 * (1. - v) - offset;
      const T y_center = y1 * v + y0 * (1. - v) - offset;

      T roi_width = std::max(std::abs(p0_x - p3_x), std::abs(p4_x - p7_x));
      T roi_height = std::max(std::abs(p0_y - p3_y), std::abs(p4_y - p7_y));
      if (aligned) {
        PADDLE_ENFORCE_GT(roi_width, 0,
                          platform::errors::InvalidArgument(
                              "The 'roi_width' attribute in BeizerAlignOp is "
                              "invalid. The height must be greater than 0. But "
                              "received 'roi_width' = %d",
                              roi_width));
        PADDLE_ENFORCE_GT(roi_height, 0,
                          platform::errors::InvalidArgument(
                              "The 'roi_height' attribute in BezierAlignOp is "
                              "invalid. The width must be greater than 0. But "
                              "received 'roi_height' = %d",
                              roi_height));
      } else {  // for backward-compatibility only
        roi_width = std::max(roi_width, (T)1.);
        roi_height = std::max(roi_height, (T)1.);
      }
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      T* offset_grad_input =
          input_grad_data + ((roi_batch_ind * channels + c) * height * width);

      int output_offset = n * n_stride + c * c_stride;
      const T* offset_grad_output = grad_output + output_offset;
      const T grad_output_this_bin =
          offset_grad_output[ph * h_stride + pw * w_stride];

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = y_center - (T)0.5 * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = x_center - (T)0.5 * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                        x_low, x_high, y_low, y_high, index);

          T g1 = grad_output_this_bin * w1 / count;
          T g2 = grad_output_this_bin * w2 / count;
          T g3 = grad_output_this_bin * w3 / count;
          T g4 = grad_output_this_bin * w4 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            // atomic add is not needed for now since it is single threaded
            add(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
            add(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
            add(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
            add(offset_grad_input + y_high * width + x_high,
                static_cast<T>(g4));
          }  // if
        }    // for x
      }      // for y
    }
  }
};

}  // namespace operators
}  // namespace paddle
