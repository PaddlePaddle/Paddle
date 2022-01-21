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
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];

    if (rois_num == 0) return;

    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto xplace = ctx.GetPlace();
    int rois_batch_size = 0;
    int* cpu_lod = nullptr;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and imgs "
              "batch_size must be the same. But received rois_batch_size = %d, "
              "batch_size = %d",
              rois_batch_size, batch_size));

      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), xplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size);
      cpu_lod = new int[rois_batch_size + 1];
      cpu_lod[0] = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        cpu_lod[i + 1] = cpu_lod[i] + rois_num_list[i];
      }
    } else {
      auto lod = rois->lod();
      PADDLE_ENFORCE_EQ(
          lod.empty(), false,
          platform::errors::InvalidArgument("Input(ROIs) in ROIAlignOp does "
                                            "not contain LoD information."));
      auto rois_lod = lod.back();
      rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and batch size "
              "of images must be the same. But received rois batch size = %d, "
              "and images batch size = %d",
              rois_batch_size, batch_size));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num, rois_num_with_lod,
          platform::errors::InvalidArgument(
              "The actual number of rois and the number of rois "
              "provided from Input(RoIsLoD) in RoIAlign must be the same."
              " But received actual number of rois is %d, and the number "
              "of rois from RoIsLoD is %d",
              rois_num, rois_num_with_lod));
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
      cpu_lod = new int[rois_batch_size + 1];
      for (int i = 0; i < rois_batch_size + 1; i++) {
        cpu_lod[i] = rois_lod[i];
      }
    }

    int* roi_id_data = nullptr;
    int r = xpu_malloc(reinterpret_cast<void**>(&roi_id_data),
                       (rois_batch_size + 1) * sizeof(int));
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External("no enough memory in xpu"));
    memory::Copy(xplace, roi_id_data, cplace, cpu_lod,
                 (rois_batch_size + 1) * sizeof(int));
    delete[] cpu_lod;
    r = xpu::roi_align<T, int>(
        dev_ctx.x_context(), in->data<T>(),
        out->mutable_data<T>(ctx.GetPlace()), rois->data<T>(), roi_id_data,
        batch_size, channels, height, width, out->dims()[0], pooled_height,
        pooled_width, spatial_scale, sampling_ratio, true);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The roi_align XPU OP return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
    xpu_free(roi_id_data);
  }
};

template <typename DeviceContext, typename T>
class XPUROIAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");

    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");

    int rois_num = rois->dims()[0];
    int channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (!in_grad) {
      return;
    }
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto xplace = ctx.GetPlace();

    int rois_batch_size = 0;
    int* cpu_lod = nullptr;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), xplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size);
      cpu_lod = new int[rois_batch_size + 1];
      cpu_lod[0] = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        cpu_lod[i + 1] = cpu_lod[i] + rois_num_list[i];
      }
    } else {
      auto rois_lod = rois->lod().back();
      rois_batch_size = rois_lod.size() - 1;
      cpu_lod = new int[rois_batch_size + 1];
      for (int i = 0; i < rois_batch_size + 1; i++) {
        cpu_lod[i] = rois_lod[i];
      }
    }
    int* roi_id_data = nullptr;
    int r = xpu_malloc(reinterpret_cast<void**>(&roi_id_data),
                       (rois_batch_size + 1) * sizeof(int));
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External("no enough memory in xpu"));
    memory::Copy(xplace, roi_id_data, cplace, cpu_lod,
                 (rois_batch_size + 1) * sizeof(int));
    in_grad->mutable_data<T>(ctx.GetPlace());

    int output_grad_size = out_grad->numel();

    delete[] cpu_lod;
    if (output_grad_size > 0) {
      r = xpu::roi_align_grad<T, int>(
          dev_ctx.x_context(), out_grad->data<T>(), in_grad->data<T>(),
          rois->data<T>(), roi_id_data, in->dims()[0], channels, height, width,
          out_grad->dims()[0], pooled_height, pooled_width, spatial_scale,
          sampling_ratio, true);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External(
              "The roi_align_grad XPU OP return wrong value[%d %s]", r,
              XPUAPIErrorMsg[r]));
    }
    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
    xpu_free(roi_id_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    roi_align,
    ops::XPUROIAlignOpKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    roi_align_grad,
    ops::XPUROIAlignGradOpKernel<paddle::platform::XPUDeviceContext, float>);

#endif
