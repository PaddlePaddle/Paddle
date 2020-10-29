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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/roi_align_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPUROIAlignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];
    const T* input_data = in->data<T>();

    framework::Tensor _roi_batch_list;
    _roi_batch_list.Resize({rois_num});
    int* rois_lod = _roi_batch_list.mutable_data<int>(ctx.GetPlace());
    int rois_batch_size = 1;
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
      rois_lod[0] = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        rois_lod[n + 1] = rois_lod[n] + rois_num_data[n];
      }
    } else {
      auto _rois_lod = rois->lod().back();
      rois_batch_size = _rois_lod.size() - 1;
      for (int n = 0; n < static_cast<int>(_rois_lod.size()); ++n) {
        rois_lod[n] = _rois_lod[n];
      }
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and imgs batch_size of roi_align_xpu OP "
              "must "
              "be the same. But received rois_batch_size %d , batch_size %d",
              rois_batch_size, batch_size));
    }
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(
        rois_num, rois_num_with_lod,
        platform::errors::InvalidArgument(
            "The rois_num from input and lod of roi_align_xpu OP must be the "
            "same. But received input rois_num %d , input lod %d",
            rois_num, rois_num_with_lod));
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* rois_data = rois->data<T>();
    for (int n = 0; n < rois_batch_size; n++) {
      int cur_batch_rois_num = rois_lod[n + 1] - rois_lod[n];
      if (cur_batch_rois_num != 0) {
        int r = xpu::roi_align(
            dev_ctx.x_context(), input_data + n * channels * height * width,
            rois_data + rois_lod[n] * 4, cur_batch_rois_num, channels, height,
            width, pooled_height, pooled_width, sampling_ratio, spatial_scale,
            output_data +
                rois_lod[n] * channels * pooled_height * pooled_width);
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::External(
                "The roi_align XPU OP return wrong value[%d], please check "
                "where Baidu Kunlun Card is properly installed.",
                r));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    roi_align,
    ops::XPUROIAlignOpKernel<paddle::platform::XPUDeviceContext, float>);

#endif
