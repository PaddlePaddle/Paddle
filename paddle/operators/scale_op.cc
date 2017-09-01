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

#include "paddle/operators/scale_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class ScaleOp : public framework::OperatorWithKernel {
 public:
  ScaleOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto *in = ctx.Input<framework::LODTensor>("X");
    auto *out = ctx.Output<framework::LODTensor>("Out");
    out->Resize(in->dims());
  }
};

template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of scale operator.").NotInGradient();
    AddOutput("Out", "The output tensor of scale operator.").NotInGradient();
    AddComment(R"DOC(Scale operator

The equation is: Out = scale*X
)DOC");
    AddAttr<AttrType>("scale", "scale of scale operator.").SetDefault(1.0);
  }
};

// Identity Op's gradient is identity op, too.
// Grad(Out=scale(X)) => Grad(X) = scale(Grad(Out))
template <typename AttrType>
class ScaleGradOp : public NetOp {
 public:
  ScaleGradOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    AppendOp(framework::OpRegistry::CreateOp(
        "scale", {{"X", {Input(framework::GradVarName("Out"))}}},
        {{"Out", {Output(framework::GradVarName("X"))}}},
        {{"scale", GetAttr<AttrType>("scale")}}));
    CompleteAddOp(false);
  }
};

// identity is a alias of scale op. This is also a example for creating a alias
// operator.
template <typename AttrType>
class IdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IdentityOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "input tensor of identity op");
    AddOutput("Out", "output tensor of identity op");
    AddComment("identity operator. Just a alias of scale op which scale = 1.0");
  }
};

template <typename AttrType>
class IdentityOp : public NetOp {
 public:
  IdentityOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    AppendOp(framework::OpRegistry::CreateOp(
        "scale", {{"X", {Input("X")}}}, {{"Out", {Output("Out")}}},
        {{"scale", static_cast<AttrType>(1)}}));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(scale, ops::ScaleOp, ops::ScaleOpMaker<float>, scale_grad,
            ops::ScaleGradOp<float>);
REGISTER_OP_CPU_KERNEL(scale,
                       ops::ScaleKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_WITHOUT_GRADIENT(identity, ops::IdentityOp<float>,
                             ops::IdentityOpMaker<float>);
