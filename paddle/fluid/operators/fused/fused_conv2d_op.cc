/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>

#include "paddle/fluid/operators/conv_op.h"

namespace paddle {
namespace operators {

class FusedConvOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) The input tensor of convolution operator. "
        "The format of input tensor is NCHW or NHWC, where N is batch size, "
        "C is the "
        "number of channels, H is the height of the feature, "
        "and W is the width of the feature.");
    AddInput(
        "Filter",
        "(Tensor) The filter tensor of convolution operator. "
        "The format of the filter tensor is MCHW, where M is the number of "
        "output image channels, C is the number of input image channels, "
        "H is the height of the filter, and W is the width of the filter. "
        "If the groups attribute is greater than 1, C equals the number of "
        "input image channels divided by the groups.");
    AddInput("Bias",
             "(Tensor) Bias to be added to each output of filter application."
             "The format of output tensor is X (one-dimensional) of size equal"
             "to the number of output channels. Only used with MKL-DNN.")
        .AsDispensable();
    AddInput("ResidualData",
             "(Tensor) Tensor with residual data "
             "to which convolution output will be added."
             "Used with fuse_residual_connection fusion.")
        .AsDispensable();
    AddOutput("Output",
              "(Tensor) The output tensor of convolution operator. "
              "It has same data fromat and data type as the Input.");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0, 0}), the "
                              "paddings(pad_height_top, pad_height_bottom, "
                              "pad_width_left, pad_wifth_right)  of "
                              "convolution operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "padding_algorithm",
        "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
        "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
        "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
        .SetDefault("EXPLICIT");
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. "
        "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
        "when group=2, the first half of the filters is only connected to the "
        "first half of the input channels, while the second half of the "
        "filters "
        "is only connected to the second half of the input channels.")
        .SetDefault(1);
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("NCHW");
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddAttr<std::string>("fuse_activation",
                         "(string, default \"\") Only used in mkldnn kernel")
        .SetDefault("");
    AddAttr<bool>("fuse_residual_connection",
                  "(bool, default false) Only used in mkldnn kernel. Used "
                  "whenever convolution output is as an input to residual "
                  "connection.")
        .SetDefault(false);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddAttr<bool>("use_mkldnn", "(bool, default false) Used in mkldnn kernel")
        .SetDefault(true);
    AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and Output(Output) are in NCHW or NHWC format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature.
Filters(Input) is MCHW format format. Where M is the number of output image channels, C is
the number of input image channels, H is the height of the filter, and W
is the width of the filter.
Parameters(strides, paddings, dilations) are two elements. These two elements represent
height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
  Output:
  ?
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
$$
       H_{out}= \frac{(H_{in} + pad_height_top + pad_height_bottom - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + pad_width_left + pad_width_right - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  }
};

class FusedConv2DOp : public operators::ConvOp {
 public:
  using operators::ConvOp::ConvOp;
};

// template <typename T>
// class Conv2DFusionMKLDNNKernel : public framework::OpKernel<T> {
//  public:
//   void Compute(const framework::ExecutionContext& ctx) const override {
//     VLOG(1) << "############ Conv2DFusionMKLDNNKernel ##############";
//     auto& dev_ctx =
//         ctx.template device_context<phi::OneDNNContext>();
//     dev_ctx.SetInputsName(ctx.GetOp().Inputs());

//     const auto* input = ctx.Input<phi::DenseTensor>("Input");

//     const auto* filter = ctx.Input<phi::DenseTensor>("Filter");
//     const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
//     const auto* residual_param = ctx.Input<phi::DenseTensor>("ResidualData");

//     const auto& strides = ctx.Attr<std::vector<int>>("strides");
//     const auto& paddings = ctx.Attr<std::vector<int>>("paddings");
//     const auto& padding_algorithm =
//     ctx.Attr<std::string>("padding_algorithm"); const auto& dilations =
//     ctx.Attr<std::vector<int>>("dilations"); int groups =
//     ctx.Attr<int>("groups"); const std::string& data_format =
//     ctx.Attr<std::string>("data_format");

//     bool is_BFLOAT16 = ctx.Attr<std::string>("mkldnn_data_type") ==
//     "bfloat16";; const std::string& fuse_activation =
//     ctx.Attr<std::string>("fuse_activation"); bool fuse_residual_conn =
//     ctx.Attr<bool>("fuse_residual_connection"); bool force_fp32_output =
//     ctx.Attr<bool>("force_fp32_output");

//     auto* output = ctx.Output<phi::DenseTensor>("Output");

//     phi::ConvOnednn<T>(dev_ctx, input, filter, bias, residual_param, strides,
//     paddings, padding_algorithm, dilations, groups, data_format, true,
//                             is_BFLOAT16, fuse_activation, fuse_residual_conn,
//                             force_fp32_output, output);
//   }
// };

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

// fused_conv2d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv2d,
    ops::ConvOp,
    ops::FusedConvOpMaker,
    ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

// fused_conv3d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv3d,
    ops::ConvOp,
    ops::FusedConvOpMaker,
    ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
