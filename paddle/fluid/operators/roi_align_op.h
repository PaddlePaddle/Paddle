/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <limits>
#include <numeric>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <class T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   const T out_grad_this_bin, const T count,
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

template <typename DeviceContext, typename T>
class CPUROIAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto in_dims = in->dims();
    auto aligned = ctx.Attr<bool>("aligned");

    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    if (!in_grad) {
      return;
    }
    Tensor roi_batch_id_list;
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
    } else {
      auto rois_lod = rois->lod().back();
      rois_batch_size = rois_lod.size() - 1;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (std::size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    }
    in_grad->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, in_grad, static_cast<T>(0));

    int output_grad_size = out_grad->numel();

    if ((!out_grad->IsInitialized()) || (output_grad_size <= 0)) {
      return;
    }

    const T* rois_data = rois->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());

    auto in_stride = phi::stride(in->dims());
    auto roi_stride = phi::stride(rois->dims());
    auto out_stride = phi::stride(out_grad->dims());

    T roi_offset = aligned ? T(0.5) : 0;
    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_idx = roi_batch_id_data[n];
      T roi_xmin = rois_data[0] * spatial_scale - roi_offset;
      T roi_ymin = rois_data[1] * spatial_scale - roi_offset;
      T roi_xmax = rois_data[2] * spatial_scale - roi_offset;
      T roi_ymax = rois_data[3] * spatial_scale - roi_offset;

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
            in_grad_data + roi_batch_idx * in_stride[0] + c * in_stride[1];
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
                bilinear_interpolate_gradient(height, width, y, x,
                                              out_grad_this_bin, count,
                                              batch_grad_data);
              }
            }
          }
        }
      }
      rois_data += roi_stride[0];
    }
  }
};
}  // namespace operators
}  // namespace paddle
