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

    auto all_dims = ctx->GetInputsDim("X");
    // const size_t var_nums = all_dims.size();
    // std::vector<framework::DDim> outs_dims;
    // outs_dims.resize(var_nums);
    // for (size_t i = 0; i < var_nums; ++i) {
    //   const auto input_dim = all_dims[i];
    //   outs_dims[i] = input_dim;
    // }
    ctx->SetOutputsDim("Out", all_dims);
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

This operator is used to perform lookups on the PSLib
then concatenated into a dense tensor.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pool_all, ops::SequencePoolAllOp,
                  ops::SequencePoolAllOpMaker);

REGISTER_OP_CPU_KERNEL(
    sequence_pool_all,
    ops::SequencePoolAllKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequencePoolAllKernel<paddle::platform::CPUDeviceContext, double>);
