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
#include <limits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

static constexpr int kROISize = 4;

template <typename DeviceContext, typename T>
class CPUROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* mask = ctx.Output<framework::Tensor>("Mask");

    auto transformed_height = ctx.Attr<int>("transformed_height");
    auto transformed_width = ctx.Attr<int>("transformed_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int in_height = in_dims[2];
    int in_width = in_dims[3];
    int rois_num = rois->dims()[0];

    const T* input_data = in->data<T>();

    //    framework::Tensor roi_batch_id_list;
    //    roi_batch_id_list.Resize({rois_num});
    //    int* roi_batch_id_data =
    //        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());
    //
    //    auto rois_lod = rois->lod().back();
    //    int rois_batch_size = rois_lod.size() - 1;
    //    PADDLE_ENFORCE_EQ(
    //        rois_batch_size, batch_size,
    //        "The rois_batch_size and imgs batch_size must be the same.");
    //    int rois_num_with_lod = rois_lod[rois_batch_size];
    //    PADDLE_ENFORCE_EQ(rois_num, rois_num_with_lod,
    //                      "The rois_num from input and lod must be the
    //                      same.");
    //    for (int n = 0; n < rois_batch_size; ++n) {
    //      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
    //        roi_batch_id_data[i] = n;
    //      }
    //    }

    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    int64_t* mask_data = mask->mutable_data<int64_t>(ctx.GetPlace());
    const int64_t* rois_data = rois->data<int64_t>();

    for (int n = 0; n < rois_num; ++n) {
      int64_t* n_rois = rois_data + n * 8;
      int64_t roi_x[4];
      int64_t roi_y[4];
      for (int k = 0; k < 4; ++k) {
        roi_x[k] = n_rois[2 * k + 1] * spatial_scale;
        roi_y[k] = n_rois[2 * k + 2] * spatial_scale;
      }

      // Get transform matrix
      int64_t transform_matrix[9];
      get_transform_matrix(transformed_width, transformed_height, roi_x, roi_y,
                           transform_matrix);

      for (int c = 0; c < channels; ++c) {
        for (int out_h = 0; h < transformed_height; ++out_h) {
          for (int out_w = 0; w < transformed_width; ++out_w) {
            int64_t in_w, in_h;
            get_source_coords(T, out_w, out_h, in_w, in_h);
          }
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class CPUROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* argmax = ctx.Input<framework::Tensor>("Argmax");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");

    if (in_grad) {
      int rois_num = rois->dims()[0];
      framework::Tensor roi_batch_id_list;
      roi_batch_id_list.Resize({rois_num});
      int* roi_batch_id_data =
          roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }

      const int64_t* rois_data = rois->data<int64_t>();
      const T* out_grad_data = out_grad->data<T>();
      const int64_t* argmax_data = argmax->data<int64_t>();
      T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.template device_context<DeviceContext>(), in_grad,
               static_cast<T>(0));

      auto in_stride = framework::stride(in->dims());
      auto argmax_stride = framework::stride(argmax->dims());
      auto roi_stride = framework::stride(rois->dims());
      auto out_stride = framework::stride(out_grad->dims());

      int channels = in->dims()[1];

      for (int n = 0; n < rois_num; ++n) {
        int roi_batch_idx = roi_batch_id_data[n];
        T* batch_grad_data = in_grad_data + roi_batch_idx * in_stride[0];
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            for (int pw = 0; pw < pooled_width; ++pw) {
              int pool_index = ph * pooled_width + pw;
              if (argmax_data[pool_index] >= 0) {
                auto index = argmax_data[pool_index];
                batch_grad_data[index] += out_grad_data[pool_index];
              }
            }
          }
          batch_grad_data += in_stride[1];
          out_grad_data += out_stride[1];
          argmax_data += argmax_stride[1];
        }
        rois_data += roi_stride[0];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
