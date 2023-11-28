/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/gcu/common/op_common.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device/gcu/compiler/single_op_compiler.h"
#include "paddle/fluid/platform/device/gcu/executor/single_op_executor.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
namespace paddle {

using LoDTensor = phi::DenseTensor;
using paddle::framework::BlockDesc;
using paddle::framework::ExecutionContext;
using paddle::framework::ProgramDesc;
using paddle::framework::Scope;
using paddle::framework::Variable;
using paddle::framework::ir::Graph;
using paddle::platform::gcu::SingleOpGcuExecutor;
using paddle::platform::gcu::SingleOpGcuExecutorManager;
using GraphPtr = const paddle::framework::ir::Graph *;
using GcuRunTimeInfo = platform::gcu::runtime::GcuRunTimeInfo;

namespace operators {
namespace gcu {

const char *const use_gcu_cache_executor = "USE_GCU_CACHE_EXECUTOR";

// all input vars should be LoDTensor & is initialized
void CheckInputVarStatus(const Variable &var, const std::string &var_name) {
  PADDLE_ENFORCE_EQ(
      var.IsType<LoDTensor>(),
      true,
      platform::errors::InvalidArgument(
          "The input variable %s of "
          "RunProgram(Grad)Op holds "
          "wrong type. Expect type is LoDTensor, but receive type is %s.",
          var_name,
          platform::demangle(framework::ToTypeName(var.Type()))));
  PADDLE_ENFORCE_EQ(
      var.Get<LoDTensor>().IsInitialized(),
      true,
      platform::errors::InvalidArgument("The tensor in input variable %s of "
                                        "RunProgram(Grad)Op "
                                        "is not initialized.",
                                        var_name));
}

void CheckOutputVarStatus(const Variable &src_var,
                          const Variable &dst_var,
                          const std::string &var_name) {
  if (dst_var.IsType<LoDTensor>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<LoDTensor>(),
        true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is LoDTensor, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<LoDTensor>().IsInitialized(),
                      true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          var_name));
  } else if (dst_var.IsType<phi::SelectedRows>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<phi::SelectedRows>(),
        true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is SelectedRows, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<phi::SelectedRows>().value().IsInitialized(),
                      true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          var_name));

  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type LoDTensor or SelectedRows, "
        "but received variable %s's type is %s",
        var_name,
        platform::demangle(framework::ToTypeName(dst_var.Type()))));
  }
}

void VariableShare(const Variable &src_var, Variable *dst_var) {
  // The previous check ensures that the variable type can only be LoDTensor or
  // SelectedRows.
  if (src_var.IsType<LoDTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<LoDTensor>();
    lod_tensor->ShareDataWith(src_var.Get<LoDTensor>());
    lod_tensor->set_lod(src_var.Get<LoDTensor>().lod());
  } else if (src_var.IsType<phi::SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<phi::SelectedRows>();
    selected_rows->mutable_value()->ShareDataWith(
        src_var.Get<phi::SelectedRows>().value());
    selected_rows->set_rows(src_var.Get<phi::SelectedRows>().rows());
    selected_rows->set_height(src_var.Get<phi::SelectedRows>().height());
  }
}

void ShareVarsIntoScope(const std::vector<VarNameValuePair> &vars,
                        Scope *scope) {
  for (auto &var : vars) {
    if (var.first == "Fake_var" || var.first == framework::kEmptyVarName) {
      continue;
    }
    auto *scope_var = scope->Var(var.first);
    CheckInputVarStatus(*var.second, var.first);
    VariableShare(*var.second, scope_var);
  }
}

void ShareVarsFromScope(const std::vector<VarNameValuePair> &vars,
                        const std::vector<std::string> var_names,
                        const BlockDesc &global_block,
                        Scope *scope) {
  for (auto &var : vars) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't findthem in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    if (var.first == framework::kEmptyVarName || var.first == "Fake_var" ||
        !global_block.HasVar(var.first)) {
      VLOG(2) << "find variable name is " << var.first << ", skip it!";
      continue;
    }
    if (std::find(var_names.begin(), var_names.end(), var.first) ==
        var_names.end()) {
      VLOG(2) << "variable " << var.first << " in not in var_names ";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *scope_var = scope->FindVar(var.first);
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        platform::errors::NotFound("The output variable %s is not in "
                                   "RunProgram(Grad)Op'"
                                   "s internal scope.",
                                   var.first));
    CheckOutputVarStatus(*scope_var, *var.second, var.first);
    VariableShare(*scope_var, var.second);
  }
}

}  // namespace gcu
}  // namespace operators
}  // namespace paddle
