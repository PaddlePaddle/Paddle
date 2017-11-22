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
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename Place, typename T>
class CPURoiPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<Tensor>("Rois");
    auto* out = ctx.Output<Tensor>("Out");
    auto* argmax = ctx.Output<Tensor>("Argmax");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    PADDLE_ENFORCE_GT(pooled_height, 0,
                      "The pooled output height must greater than 0");
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      "The pooled output width must greater than 0");
    PADDLE_ENFORCE_GT(spatial_scale, 0,
                      "The spatial scale must greater than 0");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    auto out_dims = in_dims;
    out_dims[0] = rois_num;
    out_dims[1] = channels;
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;
    out->Resize(out_dims);
    argmax->Resize(out->dims());

    auto in_stride = framework::stride(in_dims);
    auto argmax_stride = framework::stride(argmax->dims());
    auto roi_stride = framework::stride(rois->dims());
    auto out_stride = framework::stride(out_dims);

    const T* input_data = in->data<T>();
    const int64_t* rois_data = rois->data<int64_t>();
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    int64_t* argmax_data = argmax->mutable_data<int64_t>(ctx.GetPlace());

    math::SetConstant<Place, T> set_zero;
    set_zero(ctx.device_context(), out, static_cast<T>(0));
    math::SetConstant<Place, int64_t> set_init;
    set_init(ctx.device_context(), argmax, static_cast<int64_t>(-1));

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

      const float* batch_data = input_data + roi_batch_id * in_stride[0];

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
            output_data[pool_index] = is_empty ? 0 : -__FLT_MAX__;

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
class CPURoiPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<Tensor>("Rois");
    auto* argmax = ctx.Input<Tensor>("Argmax");

    auto* out_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");

    if (x_grad) {
      int channels = in->dims()[1];
      auto in_stride = framework::stride(in->dims());
      auto roi_stride = framework::stride(rois->dims());

      const int64_t* rois_data = rois->data<int64_t>();
      int rois_num = rois->dims()[0];

      T* x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<Place, T> set_zero;
      set_zero(ctx.device_context(), x_grad, static_cast<T>(0));

      size_t roi_offset = roi_stride[0];
      size_t batch_offset = in_stride[0];
      size_t channel_offset = in_stride[1];

      const T* out_grad_data = out_grad->data<T>();
      size_t pool_channel_offset = pooled_height * pooled_width;
      const int64_t* argmax_data = argmax->data<int64_t>();

      for (size_t n = 0; n < rois_num; ++n) {
        size_t roi_batch_idx = rois_data[0];
        T* batch_grad_data = x_grad_data + batch_offset * roi_batch_idx;
        for (size_t c = 0; c < channels; ++c) {
          for (size_t ph = 0; ph < pooled_height; ++ph) {
            for (size_t pw = 0; pw < pooled_width; ++pw) {
              size_t pool_index = ph * pooled_width + pw;

              if (argmax_data[pool_index] >= 0) {
                size_t index = static_cast<size_t>(argmax_data[pool_index]);
                batch_grad_data[index] += out_grad_data[pool_index];
              }
            }
          }
          batch_grad_data += channel_offset;
          out_grad_data += pool_channel_offset;
          argmax_data += pool_channel_offset;
        }
        rois_data += roi_offset;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
