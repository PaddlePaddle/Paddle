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

#include "paddle/operators/clip_op.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;

class ClipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of ClipOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of ClipOp should not be null.");
    auto x_dims = ctx.Input<LoDTensor>("X")->dims();
    auto max = Attr<float>("max");
    auto min = Attr<float>("min");
    PADDLE_ENFORCE_LT(min, max, "max should be greater than min.");
    ctx.Output<LoDTensor>("Out")->Resize(x_dims);
  }
};

template <typename AttrType>
class ClipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ClipOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor)The input of clip op."
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddOutput("Out", "(Tensor)The output of clip op with shape as input(X)");
    AddAttr<AttrType>(
        "min", "(float)Minimum value, under which element is replaced by min.");
    AddAttr<AttrType>(
        "max", "(float)Maximum value, above which element is replaced by max");
    AddComment(R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'.
)DOC");
  }
};

class ClipOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<LoDTensor>("X")->dims();
    auto *x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(clip, ops::ClipOp, ops::ClipOpMaker<float>, clip_grad,
            ops::ClipOpGrad);
REGISTER_OP_CPU_KERNEL(clip,
                       ops::ClipKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(clip_grad, ops::ClipGradKernel<float>);
