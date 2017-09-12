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

#include "paddle/operators/cond_op.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class CondOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CondOpProtoAndCheckerMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Cond", "The condition, which is a bool vector");
    AddInput("Xs", "Inputs of Subnets").AsDuplicable();
    AddOutput("Outs", "Outputs of Cond_Op after merge").AsDuplicable();

    AddOutput("SubScopes", "sub scopes for true and false branches");
    AddOutput("IndexTensors", "Index Tensors contains indices for true/false");

    AddComment(R"DOC(
Sample dependent Cond Operator:
The equation is: Out[i] = subnet_t[i], if Cond[i] == true
Out[i] = subnet_t[i], if Cond[i] == false
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(cond_op, paddle::operators::CondOp,
                             paddle::operators::CondOpProtoAndCheckerMaker);
