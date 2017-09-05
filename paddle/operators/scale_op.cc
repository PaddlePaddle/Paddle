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
#include "paddle/operators/unary_operator.h"

namespace paddle {
namespace operators {
template <typename AttrType>
class ScaleOpInfo : public UnaryOpInformation {
 public:
  std::string Name() const override { return "Scale"; }
  std::string Comment() const override {
    return R"DOC(Scale operator

The equation is: Out = scale*X
)DOC";
  }
  void AddAttrs(framework::OpProtoAndCheckerMaker *maker) const override {
    maker->AddAttr<AttrType>("scale", "scale of scale operator.")
        .SetDefault(1.0);
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

class IdentityOpInfo : public operators::UnaryOpInformation {
 public:
  std::string Name() const override { return "identity"; }
  std::string Comment() const override {
    return "identity operator. Just a alias of scale op which scale = 1.0";
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

REGISTER_OP(scale, ops::UnaryOp, ops::UnaryOpMaker<ops::ScaleOpInfo<float>>,
            scale_grad, ops::ScaleGradOp<float>);
REGISTER_OP_CPU_KERNEL(scale,
                       ops::ScaleKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_WITHOUT_GRADIENT(identity, ops::IdentityOp<float>,
                             ops::UnaryOpMaker<ops::IdentityOpInfo>);
