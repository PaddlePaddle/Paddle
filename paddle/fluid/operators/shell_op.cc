/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

// flag use to determine whether to skip shell op.
constexpr char SKIP_SHELLOP_FLAGS[] = "kSkipShellOpFlag";
class ShellOp : public framework::OperatorBase {
 public:
  ShellOp(const std::string& type, const framework::VariableNameMap& inputs,
          const framework::VariableNameMap& outputs,
          const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    // if you want to skip shell op, please set kSkipShellOpFlag variable become true.
    framework::Variable* skip_flag_var = scope.FindVar(SKIP_SHELLOP_FLAGS);
    if (skip_flag_var != null && skip_flag_var->Get<bool>()) {
      VLOG(10) << "skip shell op flag is true";
      return;
    }
    std::string cmd_format = Attr<std::string>("cmd_format");
    std::vector<std::string> cmd_params =
        Attr<std::vector<std::string>>("cmd_params");
    std::string cmd = cmd_format;
    for (size_t i = 0; i < cmd_params.size(); i++) {
      framework::Variable* cmd_params_var = scope.FindVar(cmd_params[i]);
      if (cmd_params_var != nullptr) {
        auto* pv = cmd_params_var->GetMutable<std::string>();
        if (*pv.length() == 0)
          PADDLE_THROW("%s variable is empty, it's needed by shell op.
		Please assign a value to the variable first.", cmd_params[i])
        cmd.replace(cmd.find("{}"), 2, *pv);
      } else {
        PADDLE_THROW("%s variable doesn't exist, it's needed by shell op",
                     cmd_params[i]);
      }
    }
    VLOG(4) << "shell op: " << cmd;
    framework::shell_execute("ls /");
  }
};

class ShellOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddComment(R"DOC(
		Shell operator
        This operator will execute cmds.
        )DOC");
    AddAttr<std::string>("cmd_format",
                         "(string ,default'')"
                         "indicate the command format with placeholder {}");
    AddAttr<std::vector<std::string>>(
        "cmd_params",
        "(vector<string>, default vector<string>())"
        "indicate the name of needed parameters")
        .SetDefault({});
  }
};

class ShellOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(shell, ops::ShellOp, paddle::framework::EmptyGradOpMaker,
                  ops::ShellOpMaker, ops::ShellOpShapeInference);
