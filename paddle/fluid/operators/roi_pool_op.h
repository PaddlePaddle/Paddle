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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

static constexpr int kROISize = 4;

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
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            roi_batch_id_data[i] = n;
          }
        }
      }

      const T* rois_data = rois->data<T>();
      const T* out_grad_data = out_grad->data<T>();
      const int64_t* argmax_data = argmax->data<int64_t>();
      T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());
      phi::funcs::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.template device_context<DeviceContext>(), in_grad,
               static_cast<T>(0));

      auto in_stride = phi::stride(in->dims());
      auto argmax_stride = phi::stride(argmax->dims());
      auto roi_stride = phi::stride(rois->dims());
      auto out_stride = phi::stride(out_grad->dims());

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
