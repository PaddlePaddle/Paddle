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

#include "paddle/operators/net_op.h"
#include "paddle/operators/scale_op.h"

namespace paddle {
namespace operators {

// The identity operator is an alias of the scale operator. This is also an
// example for creating an alias for an existing operator.
template <typename AttrType>
class IdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IdentityOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of identity operator.");
    AddOutput("Y", "The output tensor of identity operator.");
    AddComment(R"DOC(
The identity operator is an alias of the scale operator
with the attribute scale fixed to 1.0.
)DOC");
  }
};

template <typename AttrType>
class IdentityOp : public NetOp {
 public:
  IdentityOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    PADDLE_ENFORCE_NE(Input("X"), framework::kEmptyVarName,
                      "Input(X) of IdentityOp should not be null.");
    PADDLE_ENFORCE_NE(Output("Y"), framework::kEmptyVarName,
                      "Output(Y) of IdentityOp should not be null.");

    AppendOp(framework::OpRegistry::CreateOp(
        "scale", {{"X", {Input("X")}}}, {{"Out", {Output("Y")}}},
        {{"scale", static_cast<AttrType>(1)}}));
    CompleteAddOp(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(identity, ops::IdentityOp<float>,
                             ops::IdentityOpMaker<float>);
