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

#include "paddle/operators/gather_op.h"
#include "paddle/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherOp : public framework::OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 2, "");
    PADDLE_ENFORCE(ctx.OutputSize() == 1, "");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(0),
                            "Inputs of GatherOp must all be set");
    int batch_size = ctx.Input<Tensor>(1)->dims()[0];
    PADDLE_ENFORCE(batch_size > 0);
    paddle::framework::DDim output_dims(ctx.Input<Tensor>(0)->dims());
    output_dims[0] = batch_size;
    ctx.Output<Tensor>(0)->Resize(output_dims);
  }
};

class GatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GatherOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddOutput("Y", "The output of add op");
    AddComment(R"DOC(
Gather Operator by selecting from the first axis, 

Y = X[Index]
)DOC");
  }
};

class GatherGradOp : public framework::OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    ctx.Output<Tensor>("X" + framework::kGradVarSuffix)
        ->Resize(ctx.Input<Tensor>("X")->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(gather, ops::GatherOp, ops::GatherOpMaker);
REGISTER_OP_CPU_KERNEL(gather,
                       ops::GatherOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_GRADIENT_OP(gather, gather_grad, ops::GatherGradOp);
REGISTER_OP_CPU_KERNEL(
    gather_grad,
    ops::GatherGradientOpKernel<paddle::platform::CPUPlace, float>);
