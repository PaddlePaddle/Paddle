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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {

class ResNetBasicBlockOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // Check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Filter1"), "Input", "Filter1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale1"), "Input", "Scale1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias1"), "Input", "Bias1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Mean1"), "Input", "Mean1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Var1"), "Input", "Var1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Filter2"), "Input", "Filter2", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale2"), "Input", "Scale2", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias2"), "Input", "Bias2", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Mean2"), "Input", "Mean2", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Var2"), "Input", "Var2", "ResNetBasicBlockOp");

    bool has_shortcut = ctx->Attrs().Get<bool>("has_shortcut");
    if (has_shortcut) {
      OP_INOUT_CHECK(
          ctx->HasInput("Filter3"), "Input", "Filter3", "ResNetBasicBlockOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Scale3"), "Input", "Scale3", "ResNetBasicBlockOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Bias3"), "Input", "Bias3", "ResNetBasicBlockOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Mean3"), "Input", "Mean3", "ResNetBasicBlockOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Var3"), "Input", "Var3", "ResNetBasicBlockOp");
    }

    // Check output
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Conv1"), "Output", "Conv1", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedMean1"),
                   "Output",
                   "SavedMean1",
                   "ResNetBasicBlockOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedInvstd1"),
                   "Output",
                   "SavedInvstd1",
                   "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Mean1Out"), "Output", "Mean1Out", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Var1Out"), "Output", "Var1Out", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Conv2"), "Output", "Conv2", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedMean2"),
                   "Output",
                   "SavedMean2",
                   "ResNetBasicBlockOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedInvstd2"),
                   "Output",
                   "SavedInvstd2",
                   "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Mean2Out"), "Output", "Mean2Out", "ResNetBasicBlockOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Var2Out"), "Output", "Var2Out", "ResNetBasicBlockOp");
    if (has_shortcut) {
      OP_INOUT_CHECK(
          ctx->HasOutput("Conv3"), "Output", "Conv3", "ResNetBasicBlockOp");
      OP_INOUT_CHECK(ctx->HasOutput("SavedMean3"),
                     "Output",
                     "SavedMean3",
                     "ResNetBasicBlockOp");
      OP_INOUT_CHECK(ctx->HasOutput("SavedInvstd3"),
                     "Output",
                     "SavedInvstd3",
                     "ResNetBasicBlockOp");
      OP_INOUT_CHECK(ctx->HasOutput("Mean3Out"),
                     "Output",
                     "Mean3Out",
                     "ResNetBasicBlockOp");
      OP_INOUT_CHECK(
          ctx->HasOutput("Var3Out"), "Output", "Var3Out", "ResNetBasicBlockOp");
    }

    // make sure Mean/RunningMean and Var/RunningVar share memory
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean1")[0],
                      ctx->Outputs("Mean1Out")[0],
                      platform::errors::InvalidArgument(
                          "Mean1 and Mean1Out should share the same memory"));
    PADDLE_ENFORCE_EQ(ctx->Inputs("Var1")[0],
                      ctx->Outputs("Var1Out")[0],
                      platform::errors::InvalidArgument(
                          "Var1 and Var1Out should share the same memory"));
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean2")[0],
                      ctx->Outputs("Mean2Out")[0],
                      platform::errors::InvalidArgument(
                          "Mean2 and Mean2Out should share the same memory"));
    PADDLE_ENFORCE_EQ(ctx->Inputs("Var2")[0],
                      ctx->Outputs("Var2Out")[0],
                      platform::errors::InvalidArgument(
                          "Var2 and Var2Out should share the same memory"));

    if (has_shortcut) {
      PADDLE_ENFORCE_EQ(ctx->Inputs("Mean3")[0],
                        ctx->Outputs("Mean3Out")[0],
                        platform::errors::InvalidArgument(
                            "Mean3 and Mean3Out should share the same memory"));
      PADDLE_ENFORCE_EQ(ctx->Inputs("Var3")[0],
                        ctx->Outputs("Var3Out")[0],
                        platform::errors::InvalidArgument(
                            "Var3 and Var3Out should share the same memory"));
    }

    // Check dims of inputs
    auto data_format = ctx->Attrs().Get<std::string>("data_format");
    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        platform::errors::InvalidArgument("The data format must equal to NCHW. "
                                          "But received: the data format "
                                          "= [%s]",
                                          data_format));
    int stride1 = ctx->Attrs().Get<int>("stride1");
    int stride2 = ctx->Attrs().Get<int>("stride2");
    int padding1 = ctx->Attrs().Get<int>("padding1");
    int padding2 = ctx->Attrs().Get<int>("padding2");

    const auto x1_dims = ctx->GetInputDim("X");
    const auto w1_dims = ctx->GetInputDim("Filter1");
    const auto bn1_param_dims = ctx->GetInputDim("Scale1");
    PADDLE_ENFORCE_EQ(
        x1_dims.size(),
        4,
        platform::errors::InvalidArgument("The dimensions of input "
                                          "must equal to 4."
                                          "But received: the shape of input "
                                          "= [%s], the dimension of input = "
                                          "[%d]",
                                          x1_dims,
                                          x1_dims.size()));

    // Calculate the dims of output1
    int batch = x1_dims[0];
    int output1_channel = w1_dims[0];
    int filter1_size = w1_dims[2];
    int out1_h = (x1_dims[2] + padding1 * 2 - filter1_size) / stride1 + 1;
    int out1_w = (x1_dims[3] + padding1 * 2 - filter1_size) / stride1 + 1;
    std::vector<int> out1_shape = {batch, output1_channel, out1_h, out1_w};

    const auto w2_dims = ctx->GetInputDim("Filter2");
    const auto bn2_param_dims = ctx->GetInputDim("Scale2");
    int output2_channel = w2_dims[0];
    int filter2_size = w2_dims[2];
    int out2_h = (out1_h + padding2 * 2 - filter2_size) / stride2 + 1;
    int out2_w = (out1_w + padding2 * 2 - filter2_size) / stride2 + 1;
    std::vector<int> out2_shape = {batch, output2_channel, out2_h, out2_w};

    auto y_dims = phi::make_ddim(out2_shape);
    auto conv1_dims = phi::make_ddim(out1_shape);
    ctx->SetOutputDim("Y", y_dims);
    ctx->SetOutputDim("Conv1", conv1_dims);
    ctx->SetOutputDim("SavedMean1", bn1_param_dims);
    ctx->SetOutputDim("SavedInvstd1", bn1_param_dims);
    ctx->SetOutputDim("Mean1Out", bn1_param_dims);
    ctx->SetOutputDim("Var1Out", bn1_param_dims);
    ctx->SetOutputDim("Conv2", y_dims);
    ctx->SetOutputDim("Conv2Input", conv1_dims);
    ctx->SetOutputDim("SavedMean2", bn2_param_dims);
    ctx->SetOutputDim("SavedInvstd2", bn2_param_dims);
    ctx->SetOutputDim("Mean2Out", bn2_param_dims);
    ctx->SetOutputDim("Var2Out", bn2_param_dims);
    if (has_shortcut) {
      ctx->SetOutputDim("Conv3", y_dims);
      ctx->SetOutputDim("SavedMean3", bn2_param_dims);
      ctx->SetOutputDim("SavedInvstd3", bn2_param_dims);
      ctx->SetOutputDim("Mean3Out", bn2_param_dims);
      ctx->SetOutputDim("Var3Out", bn2_param_dims);
    }

    bool find_max = ctx->Attrs().Get<bool>("find_conv_input_max");
    if (find_max) {
      auto max_dims = phi::make_ddim({6});
      ctx->SetOutputDim("MaxInput1", max_dims);
      ctx->SetOutputDim("MaxFilter1", max_dims);
      ctx->SetOutputDim("MaxInput2", max_dims);
      ctx->SetOutputDim("MaxFilter2", max_dims);
      if (has_shortcut) {
        ctx->SetOutputDim("MaxInput3", max_dims);
        ctx->SetOutputDim("MaxFilter3", max_dims);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    // By default, the type of the scale, bias, mean,
    // and var tensors should be float when input tensor's dtype is float16.
    auto bn_param_type = framework::proto::VarType::FP32;
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Scale1")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Bias1")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Scale2")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Bias2")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));

    framework::LibraryType library = framework::LibraryType::kPlain;
    phi::DataLayout layout = phi::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        input_data_type, ctx.GetPlace(), layout, library);
  }
};

class ResNetBasicBlockOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    //  has_shortcut = True:       else:
    //          X                         X
    //        /                         /
    //      |       |                 |       |
    //    CONV1     |               CONV1     |
    //      |       |                 |       |
    //     BN1      |                BN1      |
    //      |       |                 |       |
    //    RELU1     |               RELU1     |
    //      |       |                 |       |
    //    CONV2   CONV3             CONV2     |
    //      |       |                 |       |
    //     BN2     BN3               BN2      |
    //      \       /                 \       /
    //         ADD                       ADD
    //          |                         |
    //         RELU                      RELU
    //          |                         |
    //          Y                         Y
    AddInput("X", "Input tensor of conv 1");
    AddInput("Filter1", "Filter tensor of conv 1");
    AddInput("Scale1", "Scale tensor of bn 1");
    AddInput("Bias1", "Bias tensor of bn 1");
    AddInput("Mean1", "Mean tensor of bn 1");
    AddInput("Var1", "Variance tensor of bn 1");
    AddInput("Filter2", "Filter tensor of conv 2");
    AddInput("Scale2", "Scale tensor of bn 2");
    AddInput("Bias2", "Bias tensor of bn 2");
    AddInput("Mean2", "Mean tensor of bn 2");
    AddInput("Var2", "Variance tensor of bn 2");
    AddInput("Filter3", "Filter tensor of conv 3").AsDispensable();
    AddInput("Scale3", "Scale tensor of bn 3").AsDispensable();
    AddInput("Bias3", "Bias tensor of bn 3").AsDispensable();
    AddInput("Mean3", "Mean tensor of bn 3").AsDispensable();
    AddInput("Var3", "Variance tensor of bn 3").AsDispensable();
    AddOutput("Y", "The result of ssd resnet unit");
    AddOutput("Conv1", "The result of conv 1");
    AddOutput("SavedMean1", "Mean of input 1 after conv 1");
    AddOutput("SavedInvstd1", "Invstd of input 1 after conv 1");
    AddOutput("Mean1Out", "Shared memory with Mean1");
    AddOutput("Var1Out", "Shared memory with Var1");
    AddOutput("Conv2", "The result of conv 2");
    AddOutput("Conv2Input", "Conv2 input data");
    AddOutput("SavedMean2", "Mean of input 2 after conv 2");
    AddOutput("SavedInvstd2", "Invstd of input 2 after conv 2");
    AddOutput("Mean2Out", "Shared memory with Mean2");
    AddOutput("Var2Out", "Shared memory with Var2");
    AddOutput("Conv3", "The result of conv 3").AsDispensable();
    AddOutput("SavedMean3", "Mean of input 3 after conv 3").AsDispensable();
    AddOutput("SavedInvstd3", "Invstd of input 3 after conv 3").AsDispensable();
    AddOutput("Mean3Out", "Shared memory with Mean3").AsDispensable();
    AddOutput("Var3Out", "Shared memory with Var3").AsDispensable();
    AddOutput("MaxInput1", "The max value of conv1 input tensor")
        .AsDispensable();
    AddOutput("MaxFilter1", "The max value of conv1 filter tensor")
        .AsDispensable();
    AddOutput("MaxInput2", "The max value of conv2 input tensor")
        .AsDispensable();
    AddOutput("MaxFilter2", "The max value of conv2 filter tensor")
        .AsDispensable();
    AddOutput("MaxInput3", "The max value of conv3 input tensor")
        .AsDispensable();
    AddOutput("MaxFilter3", "The max value of conv3 filter tensor")
        .AsDispensable();
    AddAttr<int>("stride1", "Stride of conv1").SetDefault(1);
    AddAttr<int>("stride2", "Stride of conv2").SetDefault(1);
    AddAttr<int>("stride3", "Stride of conv3").SetDefault(1);
    AddAttr<int>("padding1", "Padding of conv1").SetDefault(0);
    AddAttr<int>("padding2", "Padding of conv2").SetDefault(0);
    AddAttr<int>("padding3", "Padding of conv3").SetDefault(0);
    AddAttr<int>("dilation1", "Dilation of conv1").SetDefault(1);
    AddAttr<int>("dilation2", "Dilation of conv2").SetDefault(1);
    AddAttr<int>("dilation3", "Dilation of conv3").SetDefault(1);
    AddAttr<int>("group", "Group of all the 3 conv").SetDefault(1);
    AddAttr<float>("momentum", "Momentum of all the 3 bn").SetDefault(0.9);
    AddAttr<float>("epsilon", "Epsilon of all the 3 bn").SetDefault(1e-5);
    AddAttr<std::string>("data_format", "").SetDefault("NCHW");
    AddAttr<bool>("has_shortcut", "").SetDefault(false);
    AddAttr<bool>("use_global_stats", "").SetDefault(false);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>(
        "trainable_statistics",
        "(bool, default false) Whether to calculate mean and variance "
        "in test mode. If setting true in test mode, mean and variace "
        "will be calculated by current batch statistics.")
        .SetDefault(false);
    AddAttr<std::string>("act_type", "The activation type to be fused.")
        .SetDefault("relu");
    AddAttr<bool>("find_conv_input_max",
                  "(bool, default true) Whether to calculate max value of conv "
                  "input tensor.")
        .SetDefault(true);
    AddComment(R"DOC(
Fusion op of the basic unit of ssd resnet block.
** This is only use for XPU, if has problems, concat zhangyikun02@baidu.com **
)DOC");
  }
};

template <typename T>
class ResNetBasicBlockGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("resnet_basic_block_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Filter1", this->Input("Filter1"));
    op->SetInput("Conv1", this->Output("Conv1"));
    op->SetInput("Scale1", this->Input("Scale1"));
    op->SetInput("Bias1", this->Input("Bias1"));
    op->SetInput("SavedMean1", this->Output("SavedMean1"));
    op->SetInput("SavedInvstd1", this->Output("SavedInvstd1"));
    op->SetInput("Filter2", this->Input("Filter2"));
    op->SetInput("Conv2", this->Output("Conv2"));
    op->SetInput("Conv2Input", this->Output("Conv2Input"));
    op->SetInput("Scale2", this->Input("Scale2"));
    op->SetInput("Bias2", this->Input("Bias2"));
    op->SetInput("SavedMean2", this->Output("SavedMean2"));
    op->SetInput("SavedInvstd2", this->Output("SavedInvstd2"));
    op->SetInput("Filter3", this->Input("Filter3"));
    op->SetInput("Conv3", this->Output("Conv3"));
    op->SetInput("Scale3", this->Input("Scale3"));
    op->SetInput("Bias3", this->Input("Bias3"));
    op->SetInput("SavedMean3", this->Output("SavedMean3"));
    op->SetInput("SavedInvstd3", this->Output("SavedInvstd3"));
    op->SetInput("MaxInput1", this->Output("MaxInput1"));
    op->SetInput("MaxFilter1", this->Output("MaxFilter1"));
    op->SetInput("MaxInput2", this->Output("MaxInput2"));
    op->SetInput("MaxFilter2", this->Output("MaxFilter2"));
    op->SetInput("MaxInput3", this->Output("MaxInput3"));
    op->SetInput("MaxFilter3", this->Output("MaxFilter3"));
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Filter1"),
                  this->InputGrad("Filter1"));
    op->SetOutput(framework::GradVarName("Scale1"), this->InputGrad("Scale1"));
    op->SetOutput(framework::GradVarName("Bias1"), this->InputGrad("Bias1"));
    op->SetOutput(framework::GradVarName("Filter2"),
                  this->InputGrad("Filter2"));
    op->SetOutput(framework::GradVarName("Scale2"), this->InputGrad("Scale2"));
    op->SetOutput(framework::GradVarName("Bias2"), this->InputGrad("Bias2"));
    op->SetOutput(framework::GradVarName("Filter3"),
                  this->InputGrad("Filter3"));
    op->SetOutput(framework::GradVarName("Scale3"), this->InputGrad("Scale3"));
    op->SetOutput(framework::GradVarName("Bias3"), this->InputGrad("Bias3"));
  }
};

class ResNetBasicBlockOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

class ResNetBasicBlockGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Filter1"), "Input", "Filter1", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Conv1"), "Input", "Conv1", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale1"), "Input", "Scale1", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias1"), "Input", "Bias1", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedMean1"),
                   "Input",
                   "SavedMean1",
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedInvstd1"),
                   "Input",
                   "SavedInvstd1",
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Filter2"), "Input", "Filter2", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Conv2"), "Input", "Conv2", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale2"), "Input", "Scale2", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias2"), "Input", "Bias2", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedMean2"),
                   "Input",
                   "SavedMean2",
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedInvstd2"),
                   "Input",
                   "SavedInvstd2",
                   "ResNetBasicBlockGradOp");
    bool has_shortcut = ctx->Attrs().Get<bool>("has_shortcut");
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("Filter3"),
                     "Input",
                     "Filter3",
                     "ResNetBasicBlockGradOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Scale3"), "Input", "Scale3", "ResNetBasicBlockGradOp");
      OP_INOUT_CHECK(
          ctx->HasInput("Bias3"), "Input", "Bias3", "ResNetBasicBlockGradOp");
    }
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "ResNetBasicBlockGradOp");

    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Filter1")),
                   "Output",
                   framework::GradVarName("Filter1"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Scale1")),
                   "Output",
                   framework::GradVarName("Scale1"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Bias1")),
                   "Output",
                   framework::GradVarName("Bias1"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Filter2")),
                   "Output",
                   framework::GradVarName("Filter2"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Scale2")),
                   "Output",
                   framework::GradVarName("Scale2"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Bias2")),
                   "Output",
                   framework::GradVarName("Bias2"),
                   "ResNetBasicBlockGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "ResNetBasicBlockGradOp");
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Filter3")),
                     "Output",
                     framework::GradVarName("Filter3"),
                     "ResNetBasicBlockGradOp");
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Scale3")),
                     "Output",
                     framework::GradVarName("Scale3"),
                     "ResNetBasicBlockGradOp");
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Bias3")),
                     "Output",
                     framework::GradVarName("Bias3"),
                     "ResNetBasicBlockGradOp");
    }

    const auto x1_dims = ctx->GetInputDim("X");
    const auto filter1_x_dims = ctx->GetInputDim("Filter1");
    const auto param1_dims = ctx->GetInputDim("Scale1");
    const auto filter2_x_dims = ctx->GetInputDim("Filter2");
    const auto param2_dims = ctx->GetInputDim("Scale2");
    ctx->SetOutputDim(framework::GradVarName("X"), x1_dims);
    ctx->SetOutputDim(framework::GradVarName("Filter1"), filter1_x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale1"), param1_dims);
    ctx->SetOutputDim(framework::GradVarName("Bias1"), param1_dims);
    ctx->SetOutputDim(framework::GradVarName("Filter2"), filter2_x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale2"), param2_dims);
    ctx->SetOutputDim(framework::GradVarName("Bias2"), param2_dims);
    if (has_shortcut) {
      const auto filter_z_dims = ctx->GetInputDim("Filter3");
      ctx->SetOutputDim(framework::GradVarName("Filter3"), filter_z_dims);
      ctx->SetOutputDim(framework::GradVarName("Scale3"), param2_dims);
      ctx->SetOutputDim(framework::GradVarName("Bias3"), param2_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar(framework::GradVarName("Y")),
        platform::errors::NotFound(
            "Can not find Y@GRAD in the execution context."));

    framework::LibraryType library = framework::LibraryType::kPlain;
    phi::DataLayout layout = phi::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.GetPlace(),
        layout,
        library);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(resnet_basic_block,
                  ops::ResNetBasicBlockOp,
                  ops::ResNetBasicBlockOpMaker,
                  ops::ResNetBasicBlockOpInferVarType,
                  ops::ResNetBasicBlockGradOpMaker<paddle::framework::OpDesc>,
                  ops::ResNetBasicBlockGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(resnet_basic_block_grad, ops::ResNetBasicBlockGradOp);
