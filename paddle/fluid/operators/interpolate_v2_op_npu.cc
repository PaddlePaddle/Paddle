/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/interpolate_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class InterpolateV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    PADDLE_ENFORCE_EQ(input_dims.size(), 4UL,
                      platform::errors::External(
                          "NPU Interpolate Kernel only support 4-D Tensor."));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

    PADDLE_ENFORCE_EQ(
        input->layout(), data_layout,
        platform::errors::InvalidArgument(
            "Interpolate OP's input tensor layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(input->layout()), data_layout_str));
    PADDLE_ENFORCE_EQ(
        output->layout(), data_layout,
        platform::errors::InvalidArgument(
            "Interpolate OP's output tensor layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(output->layout()), data_layout_str));

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(
        align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_shape_tensor.size() > 0) {
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx =
          platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_shape_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_shape_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    } else if (ctx.HasInput("OutSize")) {
      auto out_size = ctx.Input<Tensor>("OutSize");
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_w in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_w));
          PADDLE_ENFORCE_EQ(
              scale_h > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_h in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_h));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
    }
    PADDLE_ENFORCE_GT(out_h, 0,
                      platform::errors::InvalidArgument(
                          "out_h in Attr(out_shape) of Op(interpolate) "
                          "should be greater than 0."));
    PADDLE_ENFORCE_GT(out_w, 0,
                      platform::errors::InvalidArgument(
                          "out_w in Attr(out_shape) of Op(interpolate) "
                          "should be greater than 0."));
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {n, c, out_h, out_w};
    } else {
      dim_out = {n, out_h, out_w, c};
    }
    output->mutable_data<T>(dim_out, ctx.GetPlace());

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    // To-do(qili93): need to support bilineare, try ResizeD
    if ("nearest" == interp_method) {
      runner.SetType("ResizeNearestNeighborV2")
          .AddInput(*input)
          .AddInput(std::vector<int32_t>{out_h, out_w})
          .AddOutput(*output)
          .AddAttr("align_corners", align_corners)
          .AddAttr("half_pixel_centers", false);
    }
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class InterpolateV2NPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

    PADDLE_ENFORCE_EQ(
        input->layout(), data_layout,
        platform::errors::InvalidArgument(
            "Interpolate OP's input tensor layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(input->layout()), data_layout_str));
    PADDLE_ENFORCE_EQ(output_grad->layout(), data_layout,
                      platform::errors::InvalidArgument(
                          "Interpolate OP's output_grad tensor layout should "
                          "equal to attr data_layout, but got tensor layout is "
                          "<%s>, and attr layout is <%s>",
                          framework::DataLayoutToString(output_grad->layout()),
                          data_layout_str));
    PADDLE_ENFORCE_EQ(input_grad->layout(), data_layout,
                      platform::errors::InvalidArgument(
                          "Interpolate OP's input_grad tensor layout should "
                          "equal to attr data_layout, but got tensor layout is "
                          "<%s>, and attr layout is <%s>",
                          framework::DataLayoutToString(input_grad->layout()),
                          data_layout_str));

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(
        align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx =
          platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_size_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_size_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    } else if (ctx.HasInput("OutSize")) {
      auto out_size = ctx.Input<Tensor>("OutSize");
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_w = scale_data[0];
          scale_h = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];
          PADDLE_ENFORCE_EQ(
              scale_w > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_w in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_w));
          PADDLE_ENFORCE_EQ(
              scale_h > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_h in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_h));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
    }

    framework::DDim dim_grad;
    if (data_layout == DataLayout::kNCHW) {
      dim_grad = {n, c, in_h, in_w};
    } else {
      dim_grad = {n, in_h, in_w, c};
    }

    input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    // To-do(qili93): need to support bilineare, try ResizeGradD
    if ("nearest" == interp_method) {
      runner.SetType("ResizeNearestNeighborV2Grad")
          .AddInput(*output_grad)
          .AddInput(std::vector<int32_t>{in_h, in_w})
          .AddOutput(*input_grad)
          .AddAttr("align_corners", align_corners)
          .AddAttr("half_pixel_centers", false);
    }
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2_grad,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, plat::float16>);
