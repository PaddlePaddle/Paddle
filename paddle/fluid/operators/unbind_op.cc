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

#include "paddle/fluid/operators/unbind_op.h"
#include <string>

namespace paddle {
namespace operators {
using framework::Tensor;

class UnbindOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of UnbindOp is not found."));
    PADDLE_ENFORCE_GE(
        ctx->Outputs("Out").size(), 1UL,
        platform::errors::NotFound("Outputs(Out) of UnbindOp is not found."));
    auto in_dims = ctx->GetInputDim("X");
    auto outs_names = ctx->Outputs("Out");
    int axis = ctx->Attrs().Get<int>("axis");
    const size_t outs_number = outs_names.size();
    auto out_dims = UnbindOutsDims(in_dims, axis);
    std::vector<framework::DDim> outs_dims(outs_number, out_dims);
    ctx->SetOutputsDim("Out", outs_dims);
    for (size_t i = 0; i < outs_number; ++i) {
      ctx->ShareLoD("X", "Out", 0, i);
    }
  }
};

class UnbindOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddOutput("Out", "(Tensor) Output tensors of the unbind operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Unbind operator

Remove a tensor dimension.

Example:
  Input = [[1,2],
           [3,4],
           [5,6]]
  axis = 0
  Output[0] = [1,2]
  Output[1] = [3,4]
  Output[2] = [5,6]

    )DOC");
    AddAttr<int>("axis",
                 "(int, default 0) "
                 "dimension to remove.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(unbind, ops::UnbindOp, ops::UnbindOpMaker,
                  ops::UnbindGradMaker<paddle::framework::OpDesc>,
                  ops::UnbindGradMaker<paddle::imperative::OpBase>);
namespace plat = paddle::platform;
REGISTER_OP_CPU_KERNEL(
    unbind, ops::UnbindOpKernel<plat::CPUDeviceContext, double>,
    ops::UnbindOpKernel<plat::CPUDeviceContext, float>,
    ops::UnbindOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::UnbindOpKernel<plat::CPUDeviceContext, int>,
    ops::UnbindOpKernel<plat::CPUDeviceContext, plat::float16>,
    ops::UnbindOpKernel<plat::CPUDeviceContext, plat::bfloat16>);
