/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/scalar_switch_case_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class ScalarSwitchCaseOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScalarSwitchCaseOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The conditional variable of this operator.").AsDuplicable();
    AddOutput("Scope",
              "(std::vector<Scope*>) The step scope of conditional block. To "
              "unify the conditional block, rnn and while op, the type of "
              "scope is std::vector<Scope*>");
    AddAttr<std::vector<framework::BlockDesc *>>(
        "sub_blocks",
        "The step block of conditional "
        "block operator, the length should be the same as X");
    AddComment(R"DOC(Conditional block operator

Run the sub-block if X is not empty. Params is the other inputs and Out is the
outputs of the sub-block.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle