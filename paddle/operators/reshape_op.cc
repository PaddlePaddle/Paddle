
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

#include "paddle/operators/reshape_op.h"

namespace paddle {
namespace operators {

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto shape = ctx.Attr<std::vector<int>>("shape");
    PADDLE_ENFORCE_EQ((unsigned)shape.size(), in->dims().size(),
                      "The dimension of Input(X) mismatches with Attr(shape).");
    size_t shape_size = 1;
    for (auto dim : shape) {
      shape_size *= dim;
    }
    size_t in_size = framework::product(in->dims());
    PADDLE_ENFORCE_EQ(shape_size, in_size,
                      "The size of Input(X) mismatches with Attr(shape).");
    ctx.Output<framework::Tensor>("Out")->Resize(in->dims());
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReshapeOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of reshape operator.");
    AddOutput("Out", "The output tensor of reshape operator.");
    AddAttr<std::vector<int>>("shape", "Target shape of reshape operator.");
    AddComment(R"DOC(Reshape operator

Reshape Input(X) into the shape specified by Attr(shape).
)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dims = ctx.Input<framework::Tensor>("X")->dims();
    auto *d_in = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_in->Resize(dims);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(reshape, ops::ReshapeOp, ops::ReshapeOpMaker, reshape_grad,
            ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL(reshape,
                       ops::ReshapeKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    reshape_grad, ops::ReshapeGradKernel<paddle::platform::CPUPlace, float>);
