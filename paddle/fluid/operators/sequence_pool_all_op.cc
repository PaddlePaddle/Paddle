//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sequence_pool_all_op.h"
#include <string>

namespace paddle {
namespace operators {

class SequencePoolAllOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Input(X) of SequencePoolAllOp can not be null"));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Output(Out) of SequencePoolAllOp can not be null"));

    auto inputs_dims = ctx->GetInputsDim("X");
    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(inputs_num, 0,
                      platform::errors::InvalidArgument(
                          "ShapeError: Input tensors count should > 0. But "
                          "recevied inputs' length is 0."));
    if (inputs_num == 1) {
      VLOG(3) << "Warning: sequence_pool_all op has only one input";
    }

    int64_t input_len = -1;
    for (size_t i = 0; i < inputs_num; ++i) {
      PADDLE_ENFORCE_EQ(inputs_dims[i].size(), 2,
                        platform::errors::InvalidArgument(
                            "Only suppert two dimensions input now."));
      if (i == 0) {
        input_len = inputs_dims[0][1];
      } else {
        PADDLE_ENFORCE_EQ(inputs_dims[i][1], input_len,
                          platform::errors::InvalidArgument(
                              "The input len of all inputs must be same"));
      }
    }
    ctx->SetOutputsDim("Out", inputs_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<framework::Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = 1;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(flag, 1,
                      platform::errors::InvalidArgument(
                          "All Inputs of SequencePoolAll OP are Empty!"));
    return framework::OpKernelType(input_data_type, ctx.device_context());
  }
};

class SequencePoolAllOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of SequencePoolAll").AsDuplicable();
    AddOutput("Out", "Output result tensors.").AsDuplicable();
    AddAttr<std::string>(
        "pooltype",
        "(string, default 'SUM') the pooling pooltype of SequencePoolAllOp.")
        .SetDefault("SUM")
        .InEnum({"SUM"});
    AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
    AddComment(R"DOC(
SequencePoolAll Operator.

This Op can calculate sequence_pool of many variables(LoDTensor)  
Notice: It currently supports GPU device and SUM pooltype.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

class SequencePoolAllGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));
    auto in_names = ctx->Inputs(in_x);
    auto out_names = ctx->Outputs(out_x_g_n);
    PADDLE_ENFORCE_EQ(
        in_names.size(), out_names.size(),
        platform::errors::InvalidArgument(
            "The number of arguments in %s[%d] and %s[%d] is not equal.", in_x,
            in_names.size(), out_x_g_n, out_names.size()));
    for (size_t i = 0; i < in_names.size(); ++i) {
      if (out_names[i] != framework::kEmptyVarName) {
        ctx->ShareLoD(in_x, out_x_g_n, i, i);
      }
    }
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
class SequencePoolAllGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_pool_all_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    SequencePoolAllGradOpNoNeedBufferVarsInference, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pool_all, ops::SequencePoolAllOp,
                  ops::SequencePoolAllOpMaker,
                  ops::SequencePoolAllGradOpMaker<paddle::framework::OpDesc>,
                  ops::SequencePoolAllGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(sequence_pool_all_grad, ops::SequencePoolAllGradOp,
                  ops::SequencePoolAllGradOpNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    sequence_pool_all,
    ops::SequencePoolAllKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequencePoolAllKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(sequence_pool_all_grad,
                       ops::SequencePoolAllGradOpCPUKernel<float>,
                       ops::SequencePoolAllGradOpCPUKernel<double>);
