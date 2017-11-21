/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/concat_op.h"
#include <vector>

namespace paddle {
namespace operators {
using framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "Inputs(X) of ConcatOp should be empty.")
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ConcatOp should not be null.");

    auto ins = ctx->GetInputsDim("X");
    size_t axis = static_cast<size_t>(ctx->Attrs().Get<int>("axis"));
    const size_t n = ins.size();

    PADDLE_ENFORCE_GT(n, 1, "Input tensors count should > 1.");

    auto out_dims = ins[0];
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == axis) {
          out_dims[axis] += ins[i][j];
          continue;
        }
        PADDLE_ENFORCE_EQ(out_dims[j], ins[i][j],
                          "Input tensors should have the same "
                          "elements except the specify axis.")
      }
    }
    ctx->SetOutputDim("Out", out_dims);
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConcatOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated.")
        .SetDefault(0);
    AddComment(R"DOC(
Concat Operator.

Concatenate the input tensors along dimension axis.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6]]
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  ConcatOpGrad(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(concat, ops::ConcatOp, ops::ConcatOpMaker, concat_grad,
            ops::ConcatOpGrad)
REGISTER_OP_CPU_KERNEL(concat,
                       ops::ConcatKernel<paddle::platform::CPUPlace, float>)
REGISTER_OP_CPU_KERNEL(concat_grad,
                       ops::ConcatGradKernel<paddle::platform::CPUPlace, float>)
