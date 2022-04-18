// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;
static void CastToFP16(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream, const Tensor& in,
                       Tensor* out) {
  out->mutable_data<paddle::platform::float16>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT16)
      .Run(stream);
}

static void CastToFP32(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream, const Tensor& in,
                       Tensor* out) {
  out->mutable_data<float>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT)
      .Run(stream);
}

template <typename T>
class DepthwiseConvNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    const std::vector<int> stride = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> padding = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilation = ctx.Attr<std::vector<int>>("dilations");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";
    if (channel_last) {
      PADDLE_ENFORCE_EQ(
          output->dims()[output->dims().size() - 1],
          input->dims()[input->dims().size() - 1],
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be equal to the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[output->dims().size() - 1],
              input->dims()[input->dims().size() - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          output->dims()[1], input->dims()[1],
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be equal to the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[1], input->dims()[1]));
    }

    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&padding, &dilation, padding_algorithm,
                             in_data_dims, stride, ksize);

    std::vector<int> strides(4, 1);
    std::vector<int> dilations(4, 1);

    Tensor input_tensor, output_tensor;
    input_tensor.ShareDataWith(*input);
    output_tensor.ShareDataWith(*output);

    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_tensor.set_layout(DataLayout::kNHWC);
      strides[1] = stride[0];
      strides[2] = stride[1];
      dilations[1] = dilation[0];
      dilations[2] = dilation[1];
    } else {
      strides[2] = stride[0];
      strides[3] = stride[1];
      dilations[2] = dilation[0];
      dilations[3] = dilation[1];
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Transform filter (n, 1, h, w) --> (1, n, h, w)
    Tensor transformed_filter(filter->type());
    transformed_filter.mutable_data<T>({filter->dims()[1], filter->dims()[0],
                                        filter->dims()[2], filter->dims()[3]},
                                       ctx.device_context().GetPlace());
    std::vector<int> perm = {1, 0, 2, 3};
    const auto& runner_trans = NpuOpRunner(
        "TransposeD", {*filter}, {transformed_filter}, {{"perm", perm}});
    runner_trans.Run(stream);

    const auto& runner =
        NpuOpRunner("DepthwiseConv2D", {input_tensor, transformed_filter},
                    {output_tensor}, {{"strides", strides},
                                      {"dilations", dilations},
                                      {"pads", padding},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }
};

template <typename T>
class DepthwiseConvGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const std::vector<int> stride = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> padding = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilation = ctx.Attr<std::vector<int>>("dilations");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&padding, &dilation, padding_algorithm,
                             in_data_dims, stride, ksize);

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Transform filter (n, 1, h, w) --> (1, n, h, w)
    Tensor transformed_filter(filter->type());
    transformed_filter.mutable_data<T>({filter->dims()[1], filter->dims()[0],
                                        filter->dims()[2], filter->dims()[3]},
                                       ctx.device_context().GetPlace());
    std::vector<int> perm = {1, 0, 2, 3};
    const auto& runner_trans = NpuOpRunner(
        "TransposeD", {*filter}, {transformed_filter}, {{"perm", perm}});
    runner_trans.Run(stream);

    // construct NPU attr
    std::vector<int> strides(4, 1);
    std::vector<int> dilations(4, 1);

    Tensor input_tensor, output_grad_tensor;
    input_tensor.ShareDataWith(*input);
    output_grad_tensor.ShareDataWith(*output_grad);
    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_grad_tensor.set_layout(DataLayout::kNHWC);
      strides[1] = stride[0];
      strides[2] = stride[1];
      dilations[1] = dilation[0];
      dilations[2] = dilation[1];
    } else {
      strides[2] = stride[0];
      strides[3] = stride[1];
      dilations[2] = dilation[0];
      dilations[3] = dilation[1];
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());

      PADDLE_ENFORCE_EQ(
          (dilations[2] == 1 && dilations[3] == 1), true,
          platform::errors::InvalidArgument(
              "dilation_h and dilation_w in DepthwiseConv2DBackpropFilterD "
              "must be equal to 1, but got dilation_h %d, dilation_w %d",
              dilation[2], dilation[3]));

      NpuOpRunner runner;
      runner.SetType("DepthwiseConv2DBackpropFilterD")
          .AddInput(input_tensor)
          .AddInput(output_grad_tensor)
          .AddOutput(*filter_grad)
          .AddAttr("filter_size", phi::vectorize(transformed_filter.dims()))
          .AddAttr("strides", strides)
          .AddAttr("dilations", dilations)
          .AddAttr("pads", padding)
          .AddAttr("data_format", data_format)
          .Run(stream);
    }
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      Tensor input_grad_tensor;
      input_grad_tensor.ShareDataWith(*input_grad);
      if (channel_last) {
        input_grad_tensor.set_layout(DataLayout::kNHWC);
      }
      NpuOpRunner runner;
      runner.SetType("DepthwiseConv2DBackpropInputD")
          .AddInput(transformed_filter)
          .AddInput(output_grad_tensor)
          .AddOutput(input_grad_tensor)
          .AddAttr("input_size", phi::vectorize(input->dims()))
          .AddAttr("strides", strides)
          .AddAttr("dilations", dilations)
          .AddAttr("pads", padding)
          .AddAttr("data_format", data_format)
          .Run(stream);
    }
  }
};

template <typename T>
class NPUConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(4, 1);
    std::vector<int> dilations_vec(4, 1);

    Tensor input_tensor, output_tensor;
    input_tensor.ShareDataWith(*input);
    output_tensor.ShareDataWith(*output);
    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_tensor.set_layout(DataLayout::kNHWC);
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    const auto& runner =
        NpuOpRunner("Conv2D", {input_tensor, *filter}, {output_tensor},
                    {{"strides", strides_vec},
                     {"pads", paddings},
                     {"dilations", dilations_vec},
                     {"groups", groups},
                     {"data_format", data_format}});
    runner.Run(stream);
  }
};

template <typename T>
class NPUConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(4, 1);
    std::vector<int> dilations_vec(4, 1);

    Tensor input_tensor, output_grad_tensor;
    input_tensor.ShareDataWith(*input);
    output_grad_tensor.ShareDataWith(*output_grad);
    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_grad_tensor.set_layout(DataLayout::kNHWC);
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      std::vector<int> filter_shape_vec = phi::vectorize<int>(filter->dims());

      Tensor filter_grad_fp32(experimental::DataType::FLOAT32);
      filter_grad_fp32.Resize(filter_grad->dims());

      if (framework::TransToProtoVarType(input->dtype()) ==
          framework::proto::VarType::FP16) {
        CastToFP32(ctx, stream, *filter_grad, &filter_grad_fp32);
      } else {
        filter_grad_fp32.ShareDataWith(*filter_grad);
      }

      const auto& runner = NpuOpRunner(
          "Conv2DBackpropFilterD", {input_tensor, output_grad_tensor},
          {filter_grad_fp32}, {{"filter_size", filter_shape_vec},
                               {"strides", strides_vec},
                               {"pads", paddings},
                               {"dilations", dilations_vec},
                               {"groups", groups},
                               {"data_format", data_format}});
      runner.Run(stream);

      if (framework::TransToProtoVarType(input->dtype()) ==
          framework::proto::VarType::FP16) {
        CastToFP16(ctx, stream, filter_grad_fp32, filter_grad);
      }
    }
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      std::vector<int> input_shape_vec = phi::vectorize<int>(input->dims());

      Tensor input_grad_tensor;
      input_grad_tensor.ShareDataWith(*input_grad);
      if (channel_last) {
        input_grad_tensor.set_layout(DataLayout::kNHWC);
      }
      const auto& runner =
          NpuOpRunner("Conv2DBackpropInputD", {*filter, output_grad_tensor},
                      {input_grad_tensor}, {{"input_size", input_shape_vec},
                                            {"strides", strides_vec},
                                            {"pads", paddings},
                                            {"dilations", dilations_vec},
                                            {"groups", groups},
                                            {"data_format", data_format}});
      runner.Run(stream);
    }
  }
};

template <typename T>
class NPUConv3dKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");

    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    PADDLE_ENFORCE_EQ(data_format, "NCDHW",
                      platform::errors::Unimplemented(
                          "the data_format must be NCDHW in "
                          "the npu kernel of conv3d, but got data_format "
                          "= [%s]",
                          data_format));

    PADDLE_ENFORCE_EQ(groups, 1, platform::errors::Unimplemented(
                                     "the groups must be 1 in "
                                     "the npu kernel of conv3d, but got groups "
                                     "= [%d]",
                                     groups));

    output->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto input_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(input->dims(), dev_ctx);
    auto filter_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(filter->dims(), dev_ctx);
    auto output_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(output->dims(), dev_ctx);

    input_tensor.ShareDataWith(*input);
    filter_tensor.ShareDataWith(*filter);
    output_tensor.ShareDataWith(*output);

    input_tensor.set_layout(DataLayout::kNCDHW);
    filter_tensor.set_layout(DataLayout::kNCDHW);
    output_tensor.set_layout(DataLayout::kNCDHW);

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(5, 1);
    std::vector<int> dilations_vec(5, 1);

    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    strides_vec[4] = strides[2];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
    dilations_vec[4] = dilations[2];

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    const auto& runner =
        NpuOpRunner("Conv3D", {input_tensor, filter_tensor}, {output_tensor},
                    {{"strides", strides_vec},
                     {"pads", paddings},
                     {"dilations", dilations_vec},
                     {"groups", groups},
                     {"data_format", data_format}});
    runner.Run(stream);
  }
};

template <typename T>
class NPUConv3dGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    PADDLE_ENFORCE_EQ(data_format, "NCDHW",
                      platform::errors::Unimplemented(
                          "the data_format must be NCDHW in "
                          "the npu kernel of conv3d, but got data_format "
                          "= [%s]",
                          data_format));

    PADDLE_ENFORCE_EQ(groups, 1, platform::errors::Unimplemented(
                                     "the groups must be 1 in "
                                     "the npu kernel of conv3d, but got groups "
                                     "= [%d]",
                                     groups));

    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto input_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(input->dims(), dev_ctx);
    auto filter_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(filter->dims(), dev_ctx);
    auto output_grad_tensor = ctx.AllocateTmpTensor<T, NPUDeviceContext>(
        output_grad->dims(), dev_ctx);

    input_tensor.ShareDataWith(*input);
    filter_tensor.ShareDataWith(*filter);
    output_grad_tensor.ShareDataWith(*output_grad);

    input_tensor.set_layout(DataLayout::kNCDHW);
    filter_tensor.set_layout(DataLayout::kNCDHW);
    output_grad_tensor.set_layout(DataLayout::kNCDHW);

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(5, 1);
    std::vector<int> dilations_vec(5, 1);

    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    strides_vec[4] = strides[2];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
    dilations_vec[4] = dilations[2];

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      std::vector<int> filter_shape_vec = phi::vectorize<int>(filter->dims());

      Tensor filter_grad_tensor = ctx.AllocateTmpTensor<T, NPUDeviceContext>(
          filter_grad->dims(), dev_ctx);
      filter_grad_tensor.ShareDataWith(*filter_grad);
      filter_grad_tensor.set_layout(DataLayout::kNCDHW);

      const auto& runner = NpuOpRunner(
          "Conv3DBackpropFilterD", {input_tensor, output_grad_tensor},
          {filter_grad_tensor}, {{"filter_size", filter_shape_vec},
                                 {"strides", strides_vec},
                                 {"pads", paddings},
                                 {"dilations", dilations_vec},
                                 {"groups", groups},
                                 {"data_format", data_format}});
      runner.Run(stream);
    }

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      std::vector<int> input_shape_vec = phi::vectorize<int>(input->dims());

      Tensor input_grad_tensor = ctx.AllocateTmpTensor<T, NPUDeviceContext>(
          input_grad->dims(), dev_ctx);
      input_grad_tensor.ShareDataWith(*input_grad);
      input_grad_tensor.set_layout(DataLayout::kNCDHW);

      const auto& runner = NpuOpRunner(
          "Conv3DBackpropInputD", {filter_tensor, output_grad_tensor},
          {input_grad_tensor}, {{"input_size", input_shape_vec},
                                {"strides", strides_vec},
                                {"pads", paddings},
                                {"dilations", dilations_vec},
                                {"groups", groups},
                                {"data_format", data_format}});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(depthwise_conv2d, ops::DepthwiseConvNPUKernel<float>,
                       ops::DepthwiseConvNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(depthwise_conv2d_grad,
                       ops::DepthwiseConvGradNPUKernel<float>,
                       ops::DepthwiseConvGradNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(conv2d, ops::NPUConvOpKernel<float>,
                       ops::NPUConvOpKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(conv2d_grad, ops::NPUConvGradOpKernel<float>,
                       ops::NPUConvGradOpKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(conv3d, ops::NPUConv3dKernel<float>,
                       ops::NPUConv3dKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(conv3d_grad, ops::NPUConv3dGradKernel<float>,
                       ops::NPUConv3dGradKernel<plat::float16>);
