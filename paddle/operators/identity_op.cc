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

#include "paddle/operators/identity_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class IdentityOp : public framework::OperatorWithKernel {
 public:
  IdentityOp(const std::string &type, const VarNameMap &inputs,
             const VarNameMap &outputs, const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());
  }
};

class IdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IdentityOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of identity operator.").NotInGradient();
    AddOutput("Out", "The output tensor of identity operator.").NotInGradient();
    AddComment(R"DOC(Identity operator

The equation is: Out = X
)DOC");
  }
};

// Identity Op's gradient is identity op, too.
// Grad(Out=identity_op(X)) => Grad(X) = identity_op(Grad(Out))
class IdentityGradOp : public NetOp {
 public:
  IdentityGradOp(const std::string &type, const VarNameMap &inputs,
                 const VarNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    AddOp(framework::OpRegistry::CreateOp(
        "identity", {{"X", {Input(framework::GradVarName("Out"))}}},
        {{"Out", {Output(framework::GradVarName("X"))}}}, {}));
    CompleteAddOp(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(identity, ops::IdentityOp, ops::IdentityOpMaker, identity_grad,
            ops::IdentityGradOp);
REGISTER_OP_CPU_KERNEL(identity, ops::IdentityKernel<float>);
