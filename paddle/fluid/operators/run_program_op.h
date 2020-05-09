/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using BlockDesc = framework::BlockDesc;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

namespace details {

// all input vars should be LoDTensor & is initialized
static void CheckInputVarStatus(const Variable &var,
                                const std::string &var_name) {
  PADDLE_ENFORCE_EQ(
      var.IsType<LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "The input variable %s of "
          "RunProgram(Grad)Op(StaticModelRunner) holds "
          "wrong type. Expect type is LoDTensor, but receive type is %s.",
          var_name, platform::demangle(framework::ToTypeName(var.Type()))));
  PADDLE_ENFORCE_EQ(
      var.Get<LoDTensor>().IsInitialized(), true,
      platform::errors::InvalidArgument("The tensor in input variable %s of "
                                        "RunProgram(Grad)Op(StaticModelRunner) "
                                        "is not initialized.",
                                        var_name));
}

static void CheckOutputVarStatus(const Variable &src_var,
                                 const Variable &dst_var,
                                 const std::string &var_name) {
  if (dst_var.IsType<LoDTensor>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op(StaticModelRunner)'s internal scope holds "
            "wrong type. Expect type is LoDTensor, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<LoDTensor>().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op(StaticModelRunner)'s internal "
                          "scope is not initialized.",
                          var_name));
  } else if (dst_var.IsType<SelectedRows>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<SelectedRows>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op(StaticModelRunner)'s internal scope holds "
            "wrong type. Expect type is SelectedRows, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<SelectedRows>().value().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op(StaticModelRunner)'s "
                          "internal scope is not initialized.",
                          var_name));

  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op(StaticModelRunner) only support output "
        "variable of type LoDTensor or SelectedRows, "
        "but received variable %s's type is %s",
        var_name, platform::demangle(framework::ToTypeName(dst_var.Type()))));
  }
}

static void VariableShare(const Variable &src_var, Variable *dst_var) {
  // The previous check ensures that the variable type can only be LoDTensor or
  // SelectedRows.
  if (src_var.IsType<LoDTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<LoDTensor>();
    lod_tensor->ShareDataWith(src_var.Get<LoDTensor>());
    lod_tensor->set_lod(src_var.Get<LoDTensor>().lod());
  } else if (src_var.IsType<SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<SelectedRows>();
    selected_rows->mutable_value()->ShareDataWith(
        src_var.Get<SelectedRows>().value());
    selected_rows->set_rows(src_var.Get<SelectedRows>().rows());
    selected_rows->set_height(src_var.Get<SelectedRows>().height());
  }
}

static void ShareVarsIntoScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    auto *var = scope->Var(var_names[i]);
    CheckInputVarStatus(*vars[i], var_names[i]);
    VariableShare(*vars[i], var);
  }
}

static void ShareVarsFromScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == framework::kEmptyVarName) {
      VLOG(2) << "find variable name is " << framework::kEmptyVarName
              << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(var_names[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("The output variable %s is not in "
                                        "RunProgram(Grad)Op(StaticModelRunner)'"
                                        "s internal scope.",
                                        var_names[i]));
    CheckOutputVarStatus(*var, *vars[i], var_names[i]);
    VariableShare(*var, vars[i]);
  }
}

static void AppendSkipDeletionVars(
    std::vector<std::string> *all_vars,
    const std::vector<std::string> &append_vars) {
  for (auto &var : append_vars) {
    all_vars->emplace_back(var);
  }
}

}  // namespace details

template <typename DeviceContext, typename T>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramOpKernel Compute";
    // Step 1. prepare inputs, outputs, attrs
    auto &input_vars = ctx.MultiInputVar("X");
    auto &param_vars = ctx.MultiInputVar("Params");
    auto output_vars = ctx.MultiOutputVar("Out");

    auto input_var_names = ctx.InputNames("X");
    auto param_names = ctx.InputNames("Params");
    auto output_var_names = ctx.OutputNames("Out");

    auto *block = ctx.Attr<BlockDesc *>("global_block");
    auto *program = block->Program();
    auto start_op_index = ctx.Attr<int64_t>("start_op_index");
    auto end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto is_test = ctx.Attr<bool>("is_test");

    // NOTE(chenweihang): In order not to add new variable type, use vector
    // here. Originally, here can use scope directly.
    auto *out_scope_vec = ctx.Output<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(
        out_scope_vec->size(), 1,
        platform::errors::InvalidArgument(
            "The OutScope of RunProgramGradOp should only hold one scope."));

    // Step 2. prepare executor and init persistable variables
    framework::Executor exe(ctx.GetPlace());

    // skip delete vars
    std::vector<std::string> skip_vars;
    details::AppendSkipDeletionVars(&skip_vars, output_var_names);
    VLOG(2) << "Prepare to skip " << skip_vars.size()
            << " var(s): " << string::join_strings(skip_vars, ' ');

    auto exe_ctx = exe.Prepare(*program, 0, skip_vars);

    framework::Scope &scope = *(out_scope_vec->front());
    // share input_vars & parameters into scope
    details::ShareVarsIntoScope(input_vars, input_var_names, &scope);
    details::ShareVarsIntoScope(param_vars, param_names, &scope);

    // Step 3. run ops
    exe.RunPartialPreparedContext(exe_ctx.get(), &scope, start_op_index,
                                  end_op_index, /*create_local_scope=*/false,
                                  /*create_vars=*/true, /*keep_kids=*/!is_test);

    // Step 4. Get Output
    details::ShareVarsFromScope(output_vars, output_var_names, &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());
  }
};

template <typename DeviceContext, typename T>
class RunProgramGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramGradOpKernel Compute";
    // Step 1. prepare inputs and outputs
    auto &output_grad_vars = ctx.MultiInputVar(framework::GradVarName("Out"));
    auto input_grad_vars = ctx.MultiOutputVar(framework::GradVarName("X"));
    auto param_grad_vars = ctx.MultiOutputVar(framework::GradVarName("Params"));

    // if all output vars are set to stop_gradient, grad op no need to executed
    if (input_grad_vars.empty() && param_grad_vars.empty()) return;

    auto output_grad_var_names = ctx.InputNames(framework::GradVarName("Out"));
    // NOTE: after PR22939 [Add double grad] merged, the grad op maker's
    //   SetOutput will set to None if the input var stop_gradient=True,
    //   it will cause an NotFound error when ctx.OutputNames() is called
    std::vector<std::string> input_grad_var_names;
    std::vector<std::string> param_grad_names;
    if (!input_grad_vars.empty()) {
      input_grad_var_names = ctx.OutputNames(framework::GradVarName("X"));
    }
    if (!param_grad_vars.empty()) {
      param_grad_names = ctx.OutputNames(framework::GradVarName("Params"));
    }

    auto *block = ctx.Attr<BlockDesc *>("global_block");
    auto *program = block->Program();

    auto orig_end_op_index = ctx.Attr<int64_t>("end_op_index");
    // NOTE: skip `shape` and `fill_constant` op created by
    // fluid.backward.gradients,
    // one forward output will generate one `shape` and `fill_constant`
    int64_t start_op_index = orig_end_op_index + (output_grad_vars.size() * 2);
    int64_t end_op_index = block->OpSize();

    auto *out_scope_vec = ctx.Input<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(
        out_scope_vec->size(), 1,
        platform::errors::InvalidArgument(
            "The OutScope of RunProgramGradOp should only hold one scope."));

    // Step 2. prepare executor and scope
    framework::Executor exe(ctx.GetPlace());

    // skip delete vars
    std::vector<std::string> skip_vars;
    details::AppendSkipDeletionVars(&skip_vars, input_grad_var_names);
    details::AppendSkipDeletionVars(&skip_vars, param_grad_names);
    VLOG(2) << "Prepare to skip " << skip_vars.size()
            << " var(s): " << string::join_strings(skip_vars, ' ');

    auto exe_ctx = exe.Prepare(*program, 0, skip_vars);

    auto &scope = *(out_scope_vec->front());
    details::ShareVarsIntoScope(output_grad_vars, output_grad_var_names,
                                &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    // Step 3. run ops
    exe.RunPartialPreparedContext(exe_ctx.get(), &scope, start_op_index,
                                  end_op_index, /*create_local_scope=*/false,
                                  /*create_vars=*/true, /*keep_kids=*/false);

    // Step 4. get outputs
    details::ShareVarsFromScope(input_grad_vars, input_grad_var_names, &scope);
    details::ShareVarsFromScope(param_grad_vars, param_grad_names, &scope);
  }
};

}  // namespace operators
}  // namespace paddle
