/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class CPUROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* argmax = ctx.Output<framework::Tensor>("Argmax");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    auto in_stride = framework::stride(in_dims);
    auto argmax_stride = framework::stride(argmax->dims());
    auto roi_stride = framework::stride(rois->dims());
    auto out_stride = framework::stride(out->dims());

    const T* input_data = in->data<T>();
    const int64_t* rois_data = rois->data<int64_t>();
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    int64_t* argmax_data = argmax->mutable_data<int64_t>(ctx.GetPlace());

    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_id = rois_data[0];
      PADDLE_ENFORCE_GE(roi_batch_id, 0);
      PADDLE_ENFORCE_LT(roi_batch_id, batch_size);
      rois_data += roi_stride[0];
    }

    rois_data = rois->data<int64_t>();
    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_id = rois_data[0];
      int roi_start_w = round(rois_data[1] * spatial_scale);
      int roi_start_h = round(rois_data[2] * spatial_scale);
      int roi_end_w = round(rois_data[3] * spatial_scale);
      int roi_end_h = round(rois_data[4] * spatial_scale);

      // Force malformed ROIs to be 1x1
      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);

      const float bin_size_h =
          static_cast<float>(roi_height) / static_cast<float>(pooled_height);
      const float bin_size_w =
          static_cast<float>(roi_width) / static_cast<float>(pooled_width);

      const T* batch_data = input_data + roi_batch_id * in_stride[0];

      for (int c = 0; c < channels; ++c) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            //  Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            int hstart =
                static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
            int wstart =
                static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
            int hend =
                static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
            int wend =
                static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

            hstart = std::min(std::max(hstart + roi_start_h, 0), height);
            hend = std::min(std::max(hend + roi_start_h, 0), height);
            wstart = std::min(std::max(wstart + roi_start_w, 0), width);
            wend = std::min(std::max(wend + roi_start_w, 0), width);

            const int pool_index = ph * pooled_width + pw;

            // Define an empty pooling region to be zero
            bool is_empty = (hend <= hstart) || (wend <= wstart);
            output_data[pool_index] =
                is_empty ? 0 : -std::numeric_limits<T>::max();
            argmax_data[pool_index] = -1;

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width + w;
                if (batch_data[index] > output_data[pool_index]) {
                  output_data[pool_index] = batch_data[index];
                  argmax_data[pool_index] = index;
                }
              }
            }
          }
        }

        batch_data += in_stride[1];
        output_data += out_stride[1];
        argmax_data += argmax_stride[1];
      }
      // Increment ROI data pointer
      rois_data += roi_stride[0];
    }
    return;
  }
};

template <typename Place, typename T>
class CPUROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* argmax = ctx.Input<framework::Tensor>("Argmax");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");

    if (in_grad) {
      const int64_t* rois_data = rois->data<int64_t>();
      const T* out_grad_data = out_grad->data<T>();
      const int64_t* argmax_data = argmax->data<int64_t>();
      T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<Place, T> set_zero;
      set_zero(ctx.device_context(), in_grad, static_cast<T>(0));

      auto in_stride = framework::stride(in->dims());
      auto argmax_stride = framework::stride(argmax->dims());
      auto roi_stride = framework::stride(rois->dims());
      auto out_stride = framework::stride(out_grad->dims());

      int rois_num = rois->dims()[0];
      int channels = in->dims()[1];

      for (int n = 0; n < rois_num; ++n) {
        int roi_batch_idx = rois_data[0];
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
