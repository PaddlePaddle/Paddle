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

// identity is a alias of scale op. This is also a example for creating a alias
// operator.
template <typename AttrType>
class IdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IdentityOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "input tensor of identity op");
    AddOutput("Y", "output tensor of identity op");
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
