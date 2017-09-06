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

#include "paddle/operators/crop_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class CropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dim0 = ctx.Input<Tensor>("X")->dims();
    auto Y = ctx.Input<Tensor>("Y");
    if (Y == nullptr) {
      auto shape = GetAttr<std::vector<int>>("shape");
      PADDLE_ENFORCE_EQ(
          shape.size(), dim0.size(),
          "Shape size should be equal to dimention size of input tensor.");
      ctx.Output<Tensor>("Out")->Resize(paddle::framework::make_ddim(shape));
    } else {
      ctx.Output<Tensor>("Out")->Resize(Y->dims());
    }
  }
};

class CropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CropOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of crop op");
    AddInput("Y", "The input used as reference for cropping. ");
    AddOutput("Out", "The output of crop op.");
    AddComment(R"DOC(
Crop Operator.
)DOC");
    AddAttr<std::vector<int>>("offsets", "The offsets for cropping.");
    AddAttr<std::vector<int>>("shape", "The shape for cropping.");
  }
};

class CropOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(crop, ops::CropOp, ops::CropOpMaker, crop_grad, ops::CropOpGrad);
REGISTER_OP_CPU_KERNEL(crop,
                       ops::CropKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(crop_grad,
                       ops::CropGradKernel<paddle::platform::CPUPlace, float>);
