/*Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/shuffle_channel_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class ShuffleChannelOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ShuffleChannelOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ShuffleChannelOp");

    auto input_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        input_dims.size(),
        4,
        platform::errors::InvalidArgument("The layout of input is NCHW."));

    ctx->SetOutputDim("Out", input_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ShuffleChannelOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), "
             "the input feature data of ShuffleChannelOp, the layout is NCHW.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output of "
              "ShuffleChannelOp. The layout is NCHW.");
    AddAttr<int>("group", "the number of groups.")
        .SetDefault(1)
        .AddCustomChecker([](const int& group) {
          PADDLE_ENFORCE_GE(group,
                            1,
                            platform::errors::InvalidArgument(
                                "group should be larger than 0."));
        });
    AddComment(R"DOC(
		Shuffle Channel operator
		This opearator shuffles the channels of input x.
		It  divide the input channels in each group into several subgroups,
		and obtain a new order by selecting element from every subgroup one by one.

		Shuffle channel operation makes it possible to build more powerful structures
		with multiple group convolutional layers.
		please get more information from the following paper:
		https://arxiv.org/pdf/1707.01083.pdf
        )DOC");
  }
};

class ShuffleChannelGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto input_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(
        input_dims.size(),
        4,
        platform::errors::InvalidArgument("The layout of input is NCHW."));

    ctx->SetOutputDim(framework::GradVarName("X"), input_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ShuffleChannelGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("shuffle_channel_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shuffle_channel,
                  ops::ShuffleChannelOp,
                  ops::ShuffleChannelOpMaker,
                  ops::ShuffleChannelGradMaker<paddle::framework::OpDesc>,
                  ops::ShuffleChannelGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(shuffle_channel_grad, ops::ShuffleChannelGradOp);

REGISTER_OP_CPU_KERNEL(shuffle_channel,
                       ops::ShuffleChannelOpKernel<phi::CPUContext, float>,
                       ops::ShuffleChannelOpKernel<phi::CPUContext, double>);

REGISTER_OP_CPU_KERNEL(
    shuffle_channel_grad,
    ops::ShuffleChannelGradOpKernel<phi::CPUContext, float>,
    ops::ShuffleChannelGradOpKernel<phi::CPUContext, double>);
