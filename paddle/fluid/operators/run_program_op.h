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
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using ProgramDesc = framework::ProgramDesc;
using BlockDesc = framework::BlockDesc;
using LoDTensor = framework::LoDTensor;

using FeedFetchList = framework::FeedFetchList;

namespace {  // NOLINT
static std::string GetSkipEagerDeletionVarsDebugString(
    const std::vector<std::string> &vars) {
  std::string str = "Skip " + std::to_string(vars.size()) +
                    " var(s) in eager deletion mode: ";
  for (auto &var : vars) {
    str.append(var);
    str.push_back(' ');
  }
  return str;
}
}  // NOLINT

template <typename DeviceContext, typename T>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramOpKernel Compute";
    // Step 1. prepare inputs, outputs, attrs
    auto &input_vars = ctx.MultiInputVar("X");
    auto &param_vars = ctx.MultiInputVar("Params");
    auto output_vars = ctx.MultiOutputVar("Out");

    auto &input_var_names =
        ctx.Attr<std::vector<std::string>>("input_var_names");
    auto &param_names = ctx.Attr<std::vector<std::string>>("param_names");
    auto &output_var_names =
        ctx.Attr<std::vector<std::string>>("output_var_names");

    auto *block = ctx.Attr<BlockDesc *>("fwd_block");
    auto *fwd_program = block->Program();

    // NOTE: In order not to add new variable type, use vector here.
    // Originally, here can use scope directly.
    auto *out_scope_vec = ctx.Output<StepScopeVar>("OutScope");

    // TODO(chenweihang): check input output size
    PADDLE_ENFORCE_EQ(out_scope_vec->size(), 0,
                      "The StepScope should be empty.");

    // Step 2. prepare executor and init persistable variables
    framework::Executor exe(ctx.GetPlace());
    // framework::Scope scope;
    out_scope_vec->emplace_back(new framework::Scope());
    framework::Scope &scope = *(out_scope_vec->front());

    // share input_vars to scope
    // auto *feed_var = scope.Var("feed");
    // auto &feed_inputs = *(feed_var->GetMutable<FeedFetchList>());
    // feed_inputs.resize(input_vars.size());
    for (size_t i = 0; i < input_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_vars[i]->IsType<LoDTensor>(), true);
      PADDLE_ENFORCE_EQ(input_vars[i]->Get<LoDTensor>().IsInitialized(), true);
      auto *var = scope.Var(input_var_names[i]);
      var->GetMutable<LoDTensor>()->ShareDataWith(
          input_vars[i]->Get<LoDTensor>());
      VLOG(3) << "Create Variable " << param_names[i]
              << " global, which pointer is " << var;
      PADDLE_ENFORCE_EQ(var->Get<LoDTensor>().IsInitialized(), true);
    }

    // share paramters to scope
    for (size_t i = 0; i < param_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(param_vars[i]->IsType<LoDTensor>(), true);
      PADDLE_ENFORCE_EQ(param_vars[i]->Get<LoDTensor>().IsInitialized(), true);
      auto *var = scope.Var(param_names[i]);
      var->GetMutable<LoDTensor>()->ShareDataWith(
          param_vars[i]->Get<LoDTensor>());
      VLOG(3) << "Create Variable " << param_names[i]
              << " global, which pointer is " << var;
      PADDLE_ENFORCE_EQ(var->Get<LoDTensor>().IsInitialized(), true);
    }

    // Step 3. run ops
    exe.Run(*fwd_program, &scope, 0, false, true, {}, true);

    // find outputs
    // auto *fetch_var = scope.FindVar("fetch");
    // PADDLE_ENFORCE_NOT_NULL(fetch_var);
    // PADDLE_ENFORCE(fetch_var->IsType<FeedFetchList>(),
    //               "Only %s can be invoked by GetFetchVariable",
    //               typeid(FeedFetchList).name());
    // auto& fetch_outputs = *(fetch_var->GetMutable<FeedFetchList>());
    for (size_t i = 0; i < output_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(output_vars[i]->IsType<LoDTensor>(), true);
      auto *var = scope.FindVar(output_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(var);
      PADDLE_ENFORCE_EQ(var->IsType<LoDTensor>(), true);
      PADDLE_ENFORCE_EQ(var->Get<LoDTensor>().IsInitialized(), true);
      // TODO(chenweihang): MKLDNN
      // TensorCopySync(fetch_outputs[i], ctx.GetPlace(),
      //     output_vars[i]->GetMutable<LoDTensor>());
      output_vars[i]->GetMutable<LoDTensor>()->ShareDataWith(
          var->Get<LoDTensor>());
      PADDLE_ENFORCE_EQ(output_vars[i]->Get<LoDTensor>().IsInitialized(), true);
    }

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
    // TODO(chenweihang): whether X is stop_gradient?
    // auto &input_grad_vars = ctx.MultiOutputVar(framework::GradVarName("X"));
    auto param_grad_vars = ctx.MultiOutputVar(framework::GradVarName("Params"));

    std::vector<std::string> output_grad_var_names;
    std::vector<std::string> param_grad_names;
    auto &output_var_names =
        ctx.Attr<std::vector<std::string>>("output_var_names");
    auto &params_names = ctx.Attr<std::vector<std::string>>("param_names");
    std::transform(output_var_names.begin(), output_var_names.end(),
                   std::back_inserter(output_grad_var_names),
                   [this](const std::string &name) {
                     return framework::GradVarName(name);
                   });
    std::transform(params_names.begin(), params_names.end(),
                   std::back_inserter(param_grad_names),
                   [this](const std::string &name) {
                     return framework::GradVarName(name);
                   });

    std::stringstream ss;
    ss << "output_grad_vars size: " << output_grad_vars.size() << "\n";
    for (size_t i = 0; i < output_grad_vars.size(); ++i) {
      ss << output_grad_vars[i] << " init: ";
      ss << output_grad_vars[i]->Get<LoDTensor>().IsInitialized() << "\n";
    }
    VLOG(3) << ss.str();

    auto *block = ctx.Attr<BlockDesc *>("bwd_block");
    auto *bwd_program = block->Program();

    auto *scope_vec = ctx.Input<StepScopeVar>("OutScope");
    auto &scope = *(scope_vec->front());

    // skip delete vars, out@grad & params@grad
    std::vector<std::string> skip_vars;
    std::copy(output_grad_var_names.begin(), output_grad_var_names.end(),
              std::back_inserter(skip_vars));
    std::copy(param_grad_names.begin(), param_grad_names.end(),
              std::back_inserter(skip_vars));
    VLOG(2) << GetSkipEagerDeletionVarsDebugString(skip_vars);

    // Step 2. prepare executor and scope
    framework::Executor exe(ctx.GetPlace());

    VLOG(3) << framework::GenScopeTreeDebugInfo(scope_vec->front());

    // auto *feed_grad_var = scope.Var(framework::GradVarName("feed"));
    // auto &feed_grad_inputs = *(feed_grad_var->GetMutable<FeedFetchList>());
    // VLOG(3) << "Create Variable " << "feed@GRAD"
    //             << " global, which pointer is " << feed_grad_var;
    // feed_grad_inputs.resize(output_grad_vars.size());
    for (size_t i = 0; i < output_grad_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(output_grad_vars[i]->IsType<LoDTensor>(), true);
      PADDLE_ENFORCE_EQ(output_grad_vars[i]->Get<LoDTensor>().IsInitialized(),
                        true);
      auto *var = scope.Var(output_grad_var_names[i]);
      var->GetMutable<LoDTensor>()->ShareDataWith(
          output_grad_vars[i]->Get<LoDTensor>());
      PADDLE_ENFORCE_EQ(var->Get<LoDTensor>().IsInitialized(), true);
    }

    // Step 3. run ops
    exe.Run(*bwd_program, &scope, 0, false, true, skip_vars);

    // find outputs
    // auto *fetch_grad_var = scope.FindVar(framework::GradVarName("fetch"));
    // PADDLE_ENFORCE_NOT_NULL(fetch_grad_var);
    // PADDLE_ENFORCE(fetch_grad_var->IsType<FeedFetchList>(),
    //               "Only %s can be invoked by GetFetchVariable",
    //               typeid(FeedFetchList).name());
    // auto& fetch_grad_outputs =
    // *(fetch_grad_var->GetMutable<FeedFetchList>());
    // for (size_t i = 0; i < input_grad_vars.size(); ++i) {
    //   PADDLE_ENFORCE_EQ(input_grad_vars[i]->IsType<LoDTensor>(), true);
    //   input_grad_vars[i]->GetMutable<LoDTensor>()->ShareDataWith(fetch_grad_outputs[i]);
    // }
    for (size_t i = 0; i < param_grad_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(param_grad_vars[i]->IsType<LoDTensor>(), true);
      auto *var = scope.FindVar(param_grad_names[i]);
      PADDLE_ENFORCE_NOT_NULL(var);
      PADDLE_ENFORCE_EQ(var->IsType<LoDTensor>(), true);
      PADDLE_ENFORCE_EQ(var->Get<LoDTensor>().IsInitialized(), true);
      // TODO(chenweihang): MKLDNN
      TensorCopySync(var->Get<LoDTensor>(), ctx.GetPlace(),
                     param_grad_vars[i]->GetMutable<LoDTensor>());
      // param_grad_vars[i]->GetMutable<LoDTensor>()->ShareDataWith(var->Get<LoDTensor>());
      PADDLE_ENFORCE_EQ(param_grad_vars[i]->Get<LoDTensor>().IsInitialized(),
                        true);
    }

    std::stringstream ss2;
    ss2 << "param_grad_vars size: " << param_grad_vars.size() << "\n";
    for (size_t i = 0; i < param_grad_vars.size(); ++i) {
      ss2 << param_grad_vars[i] << "\n";
      ss2 << param_grad_vars[i]->Get<LoDTensor>() << "\n";
    }
    VLOG(3) << ss2.str();

    // Step 4. clear
    // scope_vec->clear();
  }
};

}  // namespace operators
}  // namespace paddle
