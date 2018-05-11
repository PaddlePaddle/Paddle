/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;

static constexpr char kBlock[] = "sub_block";
static constexpr char kX[] = "X";

class GoOp : public framework::OperatorBase {
 public:
  GoOp(const std::string &type, const framework::VariableNameMap &inputs,
       const framework::VariableNameMap &outputs,
       const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void ExecuteOnThread(framework::Executor *executor,
                       framework::BlockDesc *block,
                       framework::Scope *scope) const {
    framework::ProgramDesc *program = block->Program();
    executor->Run(*program, scope, block->ID(), false /*create_local_scope*/);
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    /*
     * Determine the global scope. Create a new child scope.
     * Within the child scope, add all the local variables relevant
     * to that scope.
     *
     * Now go through all the inputs to the op to ensure that
     * all of them are in the newly created scope. This is important
     * to ensure that they don't get destroyed when the parent scope
     * is deleted.
     * */

    // TODO(varunarora): Consider moving this root scope lookup to scope.h.
    const framework::Scope *root_scope = &scope;
    const framework::Scope *parent_scope = root_scope->parent();

    while (parent_scope != nullptr) {
      root_scope = parent_scope;
      parent_scope = parent_scope->parent();
    }

    framework::BlockDesc *block = Attr<framework::BlockDesc *>(kBlock);
    framework::Executor executor(dev_place);
    framework::Scope &new_scope = root_scope->NewScope();

    for (auto &var : block->AllVars()) {
      new_scope.Var(var->Name());
    }

    auto &inputs = Inputs(kX);
    for (size_t i = 0; i < inputs.size(); i++) {
      PADDLE_ENFORCE_NOT_NULL(new_scope.FindVar(inputs.at(i)),
                              "All variables used in the go block "
                              "should be created in the global scope");
    }

    // Now execute the go op with the newly created scope.
    std::thread go_thread([dev_place, block, &new_scope, this]() {
      framework::Executor executor(dev_place);
      ExecuteOnThread(&executor, block, &new_scope);
    });
    go_thread.detach();
  }
};

class GoOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "A set of variables, which are required by operators inside the "
             "block of Go Op.")
        .AsDuplicable();
    AddAttr<framework::BlockDesc *>(kBlock, "The block inside GoOp");
    AddComment(R"DOC(
)DOC");
  }
};

// TODO(thuan): Look into Gradient Operator for GO_OP

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(go, paddle::operators::GoOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::GoOpMaker);
