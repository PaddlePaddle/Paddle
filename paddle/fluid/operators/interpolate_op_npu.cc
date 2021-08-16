/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/interpolate_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

inline static void CheckArgument(const framework::ExecutionContext& ctx) {
  const std::string interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  PADDLE_ENFORCE_EQ(
      align_corners, false,
      platform::errors::InvalidArgument(
          "NPU Interpolate Kernel has diff when align_corners is true."));
  PADDLE_ENFORCE_EQ(
      interp_method, "nearest",
      platform::errors::InvalidArgument(
          "NPU Interpolate Kernel only support nearest interpolotion."));
}

inline static void ExtractNCHW(const framework::DDim& dims,
                               const DataLayout& data_layout, int32_t* n,
                               int32_t* c, int32_t* h, int32_t* w) {
  *n = dims[0];
  if (data_layout == DataLayout::kNCHW) {
    *c = dims[1];
    *h = dims[2];
    *w = dims[3];
  } else {  // kNHWC
    *h = dims[1];
    *w = dims[2];
    *c = dims[3];
  }
}

static void CalcOutSize(const framework::ExecutionContext& ctx, int32_t in_h,
                        int32_t in_w, int32_t* out_h, int32_t* out_w) {
  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  *out_h = ctx.Attr<int>("out_h");
  *out_w = ctx.Attr<int>("out_w");

  auto dev_ctx = platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
  auto list_new_size_tensor = ctx.MultiInput<Tensor>("SizeTensor");

  if (list_new_size_tensor.size() > 0) {
    std::vector<int32_t> new_size_h(1);
    std::vector<int32_t> new_size_w(1);
    framework::TensorToVector(*list_new_size_tensor[0], *dev_ctx, &new_size_h);
    framework::TensorToVector(*list_new_size_tensor[1], *dev_ctx, &new_size_w);
    *out_h = new_size_h[0];
    *out_w = new_size_w[0];
  } else {
    float scale;
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    if (scale_tensor != nullptr) {
      std::vector<float> scale_data;
      framework::TensorToVector(*scale_tensor, *dev_ctx, &scale_data);
      scale = scale_data[0];
    } else {
      scale = ctx.Attr<float>("scale");
    }

    if (scale > 0) {
      *out_h = static_cast<int32_t>(in_h * scale);
      *out_w = static_cast<int32_t>(in_w * scale);
    }

    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      std::vector<int> out_size_data;
      framework::TensorToVector(*out_size, *dev_ctx, &out_size_data);
      *out_h = out_size_data[0];
      *out_w = out_size_data[1];
    }
  }

  PADDLE_ENFORCE_GT(*out_h, 0,
                    platform::errors::InvalidArgument(
                        "out_h in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
  PADDLE_ENFORCE_GT(*out_w, 0,
                    platform::errors::InvalidArgument(
                        "out_w in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
}

template <typename T>
class InterpolateNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // NOTE(Ruibiao):
    // this kernel only support nearest interpolotion for 2D images
    // the Ascend 'ResizeNearestNeighborV2' used in this kernle has diff
    // when 'align_corners' is 'true' or data type is 'double'
    CheckArgument(ctx);

    auto* input = ctx.Input<Tensor>("X");
    framework::DDim input_dims = input->dims();

    const std::string data_layout_str =
        ctx.Attr<std::string>("data_layout");  // kNCHW or kNHWC
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    int32_t n, c, h, w, out_h, out_w;
    ExtractNCHW(input_dims, data_layout, &n, &c, &h, &w);
    CalcOutSize(ctx, h, w, &out_h, &out_w);

    // the 'input' tensor may has no set (or wrong set) of the layout
    Tensor input_x(input->type());
    input_x.ShareDataWith(*input);
    input_x.set_layout(data_layout);

    auto* output = ctx.Output<Tensor>("Out");
    framework::DDim output_dims;
    if (data_layout == DataLayout::kNCHW) {
      output_dims = {n, c, out_h, out_w};
    } else {
      output_dims = {n, out_h, out_w, c};
    }
    output->set_layout(data_layout);
    output->mutable_data<T>(output_dims, ctx.GetPlace());

    NpuOpRunner npu_op_runner;
    auto npu_stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner.SetType("ResizeNearestNeighborV2")
        .AddInput(input_x)
        .AddInput(std::vector<int32_t>{out_h, out_w})
        .AddOutput(*output)
        .AddAttr("align_corners", false)
        .AddAttr("half_pixel_centers", false)
        .Run(npu_stream);
  }
};

template <typename T>
class InterpolateGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // NOTE(Ruibiao):
    // this kernel only support nearest interpolotion for 2D images
    // the Ascend 'ResizeNearestNeighborV2' used in this kernle has diff
    // when 'align_corners' is 'true' or data type is 'double'
    CheckArgument(ctx);

    auto* input = ctx.Input<Tensor>("X");
    framework::DDim input_dims = input->dims();

    const std::string data_layout_str =
        ctx.Attr<std::string>("data_layout");  // kNCHW or kNHWC
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    int32_t n, c, h, w, out_h, out_w;
    ExtractNCHW(input_dims, data_layout, &n, &c, &h, &w);
    CalcOutSize(ctx, h, w, &out_h, &out_w);

    // the 'output_grad' tensor may has no set (or wrong set) of the layout
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor output_grad_tmp(output_grad->type());
    output_grad_tmp.ShareDataWith(*output_grad);
    output_grad_tmp.set_layout(data_layout);

    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->set_layout(data_layout);
    framework::DDim input_grad_dims;
    if (data_layout == DataLayout::kNCHW) {
      input_grad_dims = {n, c, h, w};
    } else {
      input_grad_dims = {n, h, w, c};
    }
    input_grad->mutable_data<T>(input_grad_dims, ctx.GetPlace());

    NpuOpRunner npu_op_runner;
    auto npu_stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    npu_op_runner.SetType("ResizeNearestNeighborV2Grad")
        .AddInput(output_grad_tmp)
        .AddInput(std::vector<int32_t>{h, w})
        .AddOutput(*input_grad)
        .AddAttr("align_corners", false)
        .AddAttr("half_pixel_centers", false)
        .Run(npu_stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(nearest_interp, ops::InterpolateNPUKernel<float>,
                       ops::InterpolateNPUKernel<uint8_t>);
REGISTER_OP_NPU_KERNEL(nearest_interp_grad,
                       ops::InterpolateGradNPUKernel<float>);
