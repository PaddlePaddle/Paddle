/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/partial_concat_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class PartialConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "Inputs(X) of Partial ConcatOp should not be empty.");

    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of Partial ConcatOp should not be null.");

    auto inputs_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE_EQ(inputs_dims[0].size(), 2,
                      "Only supports 2-D array with batch size in the 1st "
                      "dimension and data in the 2nd.");

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(inputs_num, 0,
                      "ShapeError: Input tensors count should > 0. But "
                      "recevied inputs' length is 0.");
    if (inputs_num == 1) {
      VLOG(3) << "Warning: concat op have only one input, may waste memory";
    }

    int start_index = ComputeStartIndex(
        static_cast<int64_t>(ctx->Attrs().Get<int>("start_index")),
        inputs_dims[0][1]);
    int partial_len = ctx->Attrs().Get<int>("length");
    if (partial_len < 0) {
      partial_len = inputs_dims[0][1] - start_index;
    }

    ctx->SetOutputDim("Out", {inputs_dims[0][0],
                              static_cast<int64_t>(partial_len * inputs_num)});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto *input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW("All Inputs of Concat OP are Empty!");
    }
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class PartialConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<int>("start_index",
                 "The start index of each instance for concatenation.")
        .SetDefault(0);
    AddAttr<int>("length",
                 "The length of each instance for concatenation."
                 " Negative values for all elements after start_index")
        .SetDefault(-1);
    AddComment(R"DOC(
Partial Concat Operator.
Partial Concatenate the input tensors along the 2nd dimension.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6],[7,8]]
  start_index = 1
  length = 1
  Output = [[2,6],
            [4,8]]
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(partial_concat, ops::PartialConcatOp,
                             ops::PartialConcatOpMaker);
REGISTER_OP_CPU_KERNEL(
    partial_concat,
    ops::PartialConcatKernel<paddle::platform::CPUDeviceContext, double>,
    ops::PartialConcatKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PartialConcatKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::PartialConcatKernel<paddle::platform::CPUDeviceContext, int>);
