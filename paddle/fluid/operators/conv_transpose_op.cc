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

#include "paddle/fluid/operators/conv_transpose_op.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using DataLayout = framework::DataLayout;

framework::OpKernelType ConvTransposeOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
  bool use_cudnn =
      ctx.HasAttr("use_cudnn") ? ctx.Attr<bool>("use_cudnn") : false;
  use_cudnn &= platform::is_gpu_place(ctx.GetPlace());
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(ctx.GetPlace())) {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
    if (use_cudnn) {
      library_ = framework::LibraryType::kCUDNN;
    }
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      this->CanMKLDNNBeUsed(ctx, data_type)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(data_type, ctx.GetPlace(), layout_, library_);
}

framework::OpKernelType ConvTransposeOp::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((var_name == "Input") &&
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = framework::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(
      expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}

void Conv2DTransposeOpMaker::Make() {
  AddInput("Input",
           "(Tensor) The input tensor of convolution transpose operator. "
           "The format of input tensor is NCHW or NHWC. Where N is batch size, "
           "C is the number of input channels, H is the height of the feature, "
           "and W is the width of the feature.");
  AddInput(
      "Filter",
      "(Tensor) The filter tensor of convolution transpose operator. "
      "The format of the filter tensor is MCHW, where M is the number of "
      "input feature channels, C is the number of "
      "output feature channels,"
      "H is the height of the filter, and W is the width of the filter. "
      "We enforce groups number == 1 in the convolution transpose scenario.");
  AddInput("Bias",
           "(Tensor) Bias to be added to each output of filter application."
           "The format of output tensor is X (one-dimensional) of size equal"
           "to the number of output channels. Only used with MKL-DNN.")
      .AsDispensable()
      .AsExtra();
  AddOutput("Output",
            "(Tensor) The output tensor of convolution transpose operator. "
            "The format of output tensor is the same as input tensor.");
  AddAttr<std::vector<int>>("output_padding",
                            "(vector<int> default: []), Additional size added "
                            "to one side of each dimension in the output "
                            "shape")
      .SetDefault({});
  AddAttr<std::vector<int>>("output_size",
                            "(vector<int> default: []), the "
                            "size of the output tensor")
      .SetDefault({})
      .SupportTensor();
  AddAttr<int>("groups",
               "(int default:1), the groups number of the convolution "
               "transpose operator. ")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1}), the "
                            "dilations(h_dilation, w_dilation) of convolution "
                            "transpose operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>(
      "strides",
      "(vector<int> default:{1, 1}), the strides(h_stride, w_stride) of "
      "convolution transpose operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int> default:{0, 0}), the paddings(h_pad, w_pad) of convolution "
      "transpose operator.")
      .SetDefault({0, 0});
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Specify that the data format of the input and output data is "
      "channel_first or channel_last.")
      .SetDefault("NCHW");
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddComment(R"DOC(
Convolution2D Transpose Operator.

The convolution transpose operation calculates the output based on the input, filter
and dilations, strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and output(Output) are in NCHW or NHWC format. Where N is batchsize, C is the
number of channels, H is the height of the feature, and W is the width of the feature.
Filter(Input) is in MCHW format. Where M is the number of input feature channels,
C is the number of output feature channels, H is the height of the filter,
and W is the width of the filter.
Parameters(strides, paddings) are two elements. These two elements represent height
and width, respectively.
The input(X) size and output(Out) size may be different.

For an example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{in}, C_{out}, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
  $$
       H_{out} = (H_{in} - 1) * strides[0] - pad_height_top - pad_height_bottom  + dilations[0] * (H_f - 1) + 1 \\
       W_{out} = (W_{in} - 1) * strides[1] - pad_width_left  - pad_width_right + dilations[1] * (W_f - 1) + 1
  $$
)DOC");
}

void Conv3DTransposeOpMaker::Make() {
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution transpose operator."
      "The format of input tensor is NCDHW or NDHWC. Where N is batch "
      "size, C is the number of channels, D is the depth of the feature, "
      "H is the height of the feature, and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution transpose operator."
           "The format of the filter tensor is MCDHW, where M is the number of "
           "input feature channels, C is the number of "
           "output feature channels, D "
           "is the depth of the filter, H is the height of the filter, and "
           "W is the width of the filter."
           "We enforce groups number == 1 and padding == 0 in "
           "the convolution3d transpose scenario.");
  AddOutput("Output",
            "(Tensor) The output tensor of convolution transpose operator."
            "The format of output tensor is the same as input tensor."
            "Where N is batch size, C is "
            "the number of channels, D is the depth of the feature, H is the "
            "height of the feature, and W is the width of the feature.");
  AddAttr<std::vector<int>>("output_padding",
                            "(vector<int> default: []), Additional size added "
                            "to one side of each dimension in the output "
                            "shape")
      .SetDefault({});
  AddAttr<std::vector<int>>("output_size",
                            "(vector<int> default: []), the "
                            "size of the output tensor")
      .SetDefault({});
  AddAttr<std::vector<int>>(
      "dilations",
      "(vector<int> default:{1, 1, 1}), the "
      "dilations(d_dilation,h_dilation, w_dilation) of convolution "
      "transpose operator.")
      .SetDefault({1, 1, 1});
  AddAttr<std::vector<int>>("strides",
                            "(vector<int> default:{1, 1, 1}), the "
                            "strides{d_stride, h_stride, w_stride} of "
                            "convolution transpose operator.")
      .SetDefault({1, 1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(vector<int> default:{0, 0, 0}), paddings(d_pad, "
                            "h_pad, w_pad) of convolution transpose operator.")
      .SetDefault({0, 0, 0});
  AddAttr<int>("groups",
               "(int default:1), the groups number of the convolution3d "
               "transpose operator. ")
      .SetDefault(1);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Specify that the data format of the input and output data is "
      "channel_first or channel_last.")
      .SetDefault("NCHW");
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddComment(R"DOC(
Convolution3D Transpose Operator.

The convolution transpose operation calculates the output based on the input, filter
and dilations, strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and output(Output) are in NCDHW or NDHWC format. Where N is batch size, C is the
number of channels, D is the depth of the feature, H is the height of the feature,
and W is the width of the feature.
Filter(Input) is in MCDHW format. Where M is the number of input feature channels,
C is the number of output feature channels, D is the depth of the filter,H is the
height of the filter, and W is the width of the filter.
Parameters(strides, paddings) are three elements. These three elements represent
depth, height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, D_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{in}, C_{out}, D_f, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, D_{out}, H_{out}, W_{out})$
  Where
  $$
       D_{out} = (D_{in} - 1) * strides[0] - pad_depth_front - pad_depth_back + dilations[0] * (D_f - 1) + 1 \\
       H_{out} = (H_{in} - 1) * strides[1] - pad_height_top  - pad_height_bottom + dilations[1] * (H_f - 1) + 1 \\
       W_{out} = (W_{in} - 1) * strides[2] - pad_width_left - pad_width_right + dilations[2] * (W_f - 1) + 1
  $$
)DOC");
}

framework::OpKernelType ConvTransposeOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  bool use_cudnn =
      ctx.HasAttr("use_cudnn") ? ctx.Attr<bool>("use_cudnn") : false;
  use_cudnn &= platform::is_gpu_place(ctx.GetPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(ctx.GetPlace())) {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
  }
#endif
  framework::LibraryType library_;
  if (use_cudnn) {
    library_ = framework::LibraryType::kCUDNN;
  } else {
    library_ = framework::LibraryType::kPlain;
  }

  framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.GetPlace(),
      layout_,
      library_);
}

template <typename T>
class ConvTransposeGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
    if (this->HasInput("Bias")) {
      op->SetInput("Bias", this->Input("Bias"));
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    op->SetAttrMap(this->Attrs());
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 */
template <typename T>
class ConvTransposeDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    // I, W, dO, ddI, ddW
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput("DOutput", this->Input(framework::GradVarName("Output")));
    op->SetInput("DDInput", this->OutputGrad(framework::GradVarName("Input")));
    op->SetInput("DDFilter",
                 this->OutputGrad(framework::GradVarName("Filter")));

    // ddO, dI, dW
    // Unlike grad op, double grad op does not use name@GRAD@GRAD
    // as key of ops' inputs and outputs.
    auto ddx = this->OutputGrad(framework::GradVarName("Input"));
    auto ddw = this->OutputGrad(framework::GradVarName("Filter"));

    op->SetOutput("DDOutput",
                  ddx.empty()
                      ? this->EmptyInputGrad()
                      : this->InputGrad(framework::GradVarName("Output")));
    op->SetOutput(
        "DFilter",
        ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Filter"));
    op->SetOutput(
        "DInput",
        ddw.empty() ? this->EmptyInputGrad() : this->InputGrad("Input"));

    op->SetAttrMap(this->Attrs());
  }
};

framework::OpKernelType ConvTransposeOpDoubleGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  bool use_cudnn =
      ctx.HasAttr("use_cudnn") ? ctx.Attr<bool>("use_cudnn") : false;
  use_cudnn &= platform::is_gpu_place(ctx.GetPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(ctx.GetPlace())) {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
  }
#endif
  framework::LibraryType library_;
  if (use_cudnn) {
    library_ = framework::LibraryType::kCUDNN;
  } else {
    library_ = framework::LibraryType::kPlain;
  }

  framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.GetPlace(),
      layout_,
      library_);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

// conv2d_transpose
DECLARE_INFER_SHAPE_FUNCTOR(conv2d_transpose,
                            Conv2dTranposeInferShapeFunctor,
                            PD_INFER_META(phi::Conv2dTransposeInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(conv2d_transpose_grad,
                            Conv2dTranposeGradInferShapeFunctor,
                            PD_INFER_META(phi::Conv2dTransposeGradInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(
    conv2d_transpose_grad_grad,
    Conv2dTranposeDoubleGradInferShapeFunctor,
    PD_INFER_META(phi::Conv2dTransposeDoubleGradInferMeta));

REGISTER_OPERATOR(conv2d_transpose,
                  ops::ConvTransposeOp,
                  ops::Conv2DTransposeOpMaker,
                  ops::ConvTransposeGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConvTransposeGradOpMaker<paddle::imperative::OpBase>,
                  Conv2dTranposeInferShapeFunctor);
REGISTER_OPERATOR(conv2d_transpose_grad,
                  ops::ConvTransposeOpGrad,
                  ops::ConvTransposeDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::ConvTransposeDoubleGradMaker<paddle::imperative::OpBase>,
                  Conv2dTranposeGradInferShapeFunctor);
REGISTER_OPERATOR(conv2d_transpose_grad_grad,
                  ops::ConvTransposeOpDoubleGrad,
                  Conv2dTranposeDoubleGradInferShapeFunctor);

// conv3d_transpose
DECLARE_INFER_SHAPE_FUNCTOR(conv3d_transpose,
                            Conv3dTranposeInferShapeFunctor,
                            PD_INFER_META(phi::ConvTransposeInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(conv3d_transpose_grad,
                            Conv3dTranposeGradInferShapeFunctor,
                            PD_INFER_META(phi::ConvTransposeGradInferMeta));

REGISTER_OPERATOR(conv3d_transpose,
                  ops::ConvTransposeOp,
                  ops::Conv3DTransposeOpMaker,
                  ops::ConvTransposeGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConvTransposeGradOpMaker<paddle::imperative::OpBase>,
                  Conv3dTranposeInferShapeFunctor);
REGISTER_OPERATOR(conv3d_transpose_grad,
                  ops::ConvTransposeOpGrad,
                  Conv3dTranposeGradInferShapeFunctor);

// depthwise conv2d_transpose
DECLARE_INFER_SHAPE_FUNCTOR(depthwise_conv2d_transpose,
                            DepthWiseConv2dTranposeInferShapeFunctor,
                            PD_INFER_META(phi::Conv2dTransposeInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(depthwise_conv2d_transpose_grad,
                            DepthWiseConv2dTranposeGradInferShapeFunctor,
                            PD_INFER_META(phi::Conv2dTransposeGradInferMeta));

REGISTER_OPERATOR(depthwise_conv2d_transpose,
                  ops::ConvTransposeOp,
                  ops::Conv2DTransposeOpMaker,
                  ops::ConvTransposeGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConvTransposeGradOpMaker<paddle::imperative::OpBase>,
                  DepthWiseConv2dTranposeInferShapeFunctor);
REGISTER_OPERATOR(depthwise_conv2d_transpose_grad,
                  ops::ConvTransposeOpGrad,
                  DepthWiseConv2dTranposeGradInferShapeFunctor);

REGISTER_OP_VERSION(conv_transpose)
    .AddCheckpoint(
        R"ROC(
      Upgrade convtranspose add a new attribute [output_padding].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "output_padding",
            "In order to add additional size to one side of each dimension "
            "in the output",
            std::vector<int>{}));

REGISTER_OP_VERSION(conv2d_transpose)
    .AddCheckpoint(
        R"ROC(
      Upgrade conv2d transpose to add a new attribute [output_padding].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "output_padding",
            "In order to add additional size to one side of each dimension "
            "in the output",
            std::vector<int>{}))
    .AddCheckpoint(
        R"ROC(
      Upgrade conv2d transpose to add a new attributes [force_fp32_output, mkldnn_data_type].
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("force_fp32_output",
                     "Force BF16 kernel output FP32, only used in MKL-DNN BF16",
                     false)
            .NewAttr("mkldnn_data_type",
                     "Data type of mkldnn kernel",
                     "float32"));

REGISTER_OP_VERSION(conv3d_transpose)
    .AddCheckpoint(
        R"ROC(
      Upgrade conv3d transpose to add a new attribute [output_padding].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "output_padding",
            "In order to add additional size to one side of each dimension "
            "in the output",
            std::vector<int>{}));

REGISTER_OP_VERSION(depthwise_conv2d_transpose)
    .AddCheckpoint(
        R"ROC(
      Upgrade depthwise conv2d transpose to add a new attribute [output_padding].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "output_padding",
            "In order to add additional size to one side of each dimension "
            "in the output",
            std::vector<int>{}));
