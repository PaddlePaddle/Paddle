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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/cuda_graph_with_in_out.h"
#endif

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using BlockDesc = framework::BlockDesc;
using ProgramDesc = framework::ProgramDesc;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using SelectedRows = phi::SelectedRows;

namespace details {

// all input vars should be LoDTensor & is initialized
static void CheckInputVarStatus(const Variable &var,
                                const std::string &var_name) {
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

static void CheckOutputVarStatus(const Variable &src_var,
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

static void VariableShare(const Variable &src_var, Variable *dst_var) {
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

static void ShareVarsIntoScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == "Fake_var") {
      continue;
    }
    auto *var = scope->Var(var_names[i]);
    CheckInputVarStatus(*vars[i], var_names[i]);
    VariableShare(*vars[i], var);
  }
}

static void ShareVarsFromScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               const BlockDesc &global_block,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't findthem in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    if (var_names[i] == framework::kEmptyVarName ||
        var_names[i] == "Fake_var" || !global_block.HasVar(var_names[i])) {
      VLOG(2) << "find variable name is " << var_names[i] << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(var_names[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound("The output variable %s is not in "
                                   "RunProgram(Grad)Op'"
                                   "s internal scope.",
                                   var_names[i]));
    CheckOutputVarStatus(*var, *vars[i], var_names[i]);
    VariableShare(*var, vars[i]);
  }
}

#ifdef PADDLE_WITH_CUDA
static cudaStreamCaptureMode StringToCUDAGraphCaptureMode(
    const std::string &mode) {
  if (mode == "global") {
    return cudaStreamCaptureModeGlobal;
  } else if (mode == "thread_local") {
    return cudaStreamCaptureModeThreadLocal;
  } else if (mode == "relaxed") {
    return cudaStreamCaptureModeRelaxed;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported CUDA Graph capture mode %s", mode));
  }
}
#endif

}  // namespace details

template <typename DeviceContext, typename T>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto &capture_mode = ctx.Attr<std::string>("cuda_graph_capture_mode");
    auto is_test = ctx.Attr<bool>("is_test");
    if (capture_mode.empty()) {
      ComputeImpl(ctx, is_test, false);
      return;
    }

#ifdef PADDLE_WITH_CUDA
    auto mode = details::StringToCUDAGraphCaptureMode(capture_mode);
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
    auto *graph_var = ctx.OutputVar("CUDAGraph");
    PADDLE_ENFORCE_NOT_NULL(
        graph_var,
        phi::errors::InvalidArgument("Output(CUDAGraph) must exist when "
                                     "cuda_graph_capture_mode is valid."));
    using GraphVecType = std::vector<std::unique_ptr<CUDAGraphWithInOuts>>;
    auto &inner_graphs = *(graph_var->GetMutable<GraphVecType>());
    inner_graphs.resize(std::max<size_t>(3, inner_graphs.size()));
    size_t graph_idx = is_test ? 0 : 1;
    if (inner_graphs[graph_idx].get() == nullptr) {
      int64_t pool_id;
      if (inner_graphs[1 - graph_idx].get() != nullptr) {
        pool_id = inner_graphs[1 - graph_idx]->PoolID();
      } else {
        pool_id = ctx.Attr<int64_t>("cuda_graph_pool_id");
      }

      framework::PEAndGraphPair pe_and_graph;
      auto callable = [this, is_test, &pe_and_graph](
                          const framework::ExecutionContext &exe_ctx) {
        pe_and_graph = ComputeImpl(exe_ctx, is_test, true);
      };
      inner_graphs[graph_idx] = CaptureCUDAGraph(
          callable, ctx, {"X"}, {"Out", "DOut"}, mode, pool_id);
      VLOG(10) << "Capture Forward CUDA Graph";
    } else {
      VLOG(10) << "Run Forward CUDA Graph directly";
      ExecuteCUDAGraph(
          ctx, {"X"}, {"Out", "DOut"}, inner_graphs[graph_idx].get());
    }
#else
    PADDLE_THROW(
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
#endif
  }

 private:
  framework::PEAndGraphPair ComputeImpl(const framework::ExecutionContext &ctx,
                                        bool is_test,
                                        bool use_cuda_graph) const {
    VLOG(2) << "RunProgramOpKernel Compute";
    framework::PEAndGraphPair pe_and_graph;
    // Step 1. prepare inputs, outputs, attrs
    auto &input_vars = ctx.MultiInputVar("X");
    auto &param_vars = ctx.MultiInputVar("Params");
    auto output_vars = ctx.MultiOutputVar("Out");
    auto dout_vars = ctx.MultiOutputVar("DOut");

    auto input_var_names = ctx.InputNames("X");
    auto output_var_names = ctx.OutputNames("Out");
    std::vector<std::string> dout_var_names;
    if (!dout_vars.empty()) {
      // DOut is a dispensable out, only get the names when it exists.
      // Otherwise, it will throw a NotFound error.
      dout_var_names = ctx.OutputNames("DOut");
    }

    // current program may not hold parameters
    std::vector<std::string> param_names;
    if (!param_vars.empty()) {
      param_names = ctx.InputNames("Params");
    }

    auto start_op_index = ctx.Attr<int64_t>("start_op_index");
    auto end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto program_id = ctx.Attr<int64_t>("program_id");

    // NOTE(chenweihang): In order not to add new variable type, use vector
    // here. Originally, here can use scope directly.
    auto *out_scope_vec = ctx.Output<StepScopeVar>("OutScope");
    std::unique_ptr<framework::Scope> inner_scope{nullptr};
    if (out_scope_vec->size() == 0) {
      // For cuda graph under static mode usage.
      // For static mode, we cannot set value of a tensor before any run,
      // the OutScope variable passed to the op actually contains nothing.
      // Just create a tmp scope to run the program.
      PADDLE_ENFORCE_EQ(
          use_cuda_graph,
          true,
          platform::errors::InvalidArgument(
              "If not provide OutScope then must run under cuda graph mode."));
      inner_scope = std::make_unique<framework::Scope>();
    } else {
      PADDLE_ENFORCE_EQ(
          out_scope_vec->size(),
          1,
          platform::errors::InvalidArgument(
              "The OutScope of RunProgramGradOp should only hold one scope."));
    }

    // Step 2. prepare executor and init persistable variables

    // NOTE(Aurelius84): While training some models, forward can be called many
    // times and then apply backpropagation all at once, such as Reinforcement
    // Learning. Tensor data in multi-step training should be saved into single
    // scope separately. Otherwise, the gradients can be miscalculated because
    // always using the Tensor data of the last step in forward.
    framework::Scope *global_inner_scope =
        out_scope_vec->size() == 0 ? inner_scope.get() : out_scope_vec->front();
    VLOG(2) << "The number of sub scopes before forward: "
            << global_inner_scope->kids().size();
    framework::Scope &scope = global_inner_scope->NewScope();

    // share input_vars & parameters into scope
    details::ShareVarsIntoScope(input_vars, input_var_names, &scope);
    details::ShareVarsIntoScope(param_vars, param_names, &scope);

    auto *global_block = ctx.Attr<BlockDesc *>("global_block");

    if (end_op_index > start_op_index) {
      auto *program = global_block->Program();
      bool is_new_created;
      if (use_cuda_graph) {
        pe_and_graph = framework::CreateFixOrderExecutorInfo(
            *program, ctx.GetPlace(), start_op_index, end_op_index, &scope);
        is_new_created = true;
      } else {
        auto cache_info = framework::GetExecutorInfoFromCache(*program,
                                                              ctx.GetPlace(),
                                                              start_op_index,
                                                              end_op_index,
                                                              /*is_grad=*/false,
                                                              program_id,
                                                              &scope);
        pe_and_graph.first = cache_info.first;
        is_new_created = cache_info.second;
      }

      auto &parallel_executor = pe_and_graph.first;

      // all out_vars are skip_eager_var
      std::vector<std::string> tmp_vars;
      auto &skip_eager_delete_vars =
          use_cuda_graph
              ? tmp_vars
              : framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
                    program_id, false);
      if (is_new_created) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_var_names);
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      output_var_names.begin(),
                                      output_var_names.end());
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      dout_var_names.begin(),
                                      dout_var_names.end());
        framework::details::ParseSafeEagerDeletionSkipVars(
            *program, end_op_index, output_var_names, &skip_eager_delete_vars);
      }

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
    }
    // Step 4. Get Output
    details::ShareVarsFromScope(
        output_vars, output_var_names, *global_block, &scope);
    details::ShareVarsFromScope(
        dout_vars, dout_var_names, *global_block, &scope);

    // Debug info: scope info when run end
    framework::Scope *target_scope{nullptr};
    if (out_scope_vec->size() == 0) {
      target_scope = inner_scope.get();
    } else {
      target_scope = out_scope_vec->front();
    }
    VLOG(3) << framework::GenScopeTreeDebugInfo(target_scope);
    // Step 5. Drop all children scopes while testing.
    if (is_test) {
      target_scope->DropKids();
    }
    VLOG(2) << "The number of sub scopes after forward: "
            << target_scope->kids().size();
#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) platform::DontClearMKLDNNCache(ctx.GetPlace());
#endif
    return pe_and_graph;
  }
};

template <typename DeviceContext, typename T>
class RunProgramGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto &capture_mode = ctx.Attr<std::string>("cuda_graph_capture_mode");
    if (capture_mode.empty()) {
      ComputeImpl(ctx, false);
      return;
    }

#ifdef PADDLE_WITH_CUDA
    auto mode = details::StringToCUDAGraphCaptureMode(capture_mode);
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
    auto *graph_var =
        const_cast<framework::Variable *>(ctx.InputVar("CUDAGraph"));
    PADDLE_ENFORCE_NOT_NULL(
        graph_var,
        phi::errors::InvalidArgument("Output(CUDAGraph) must exist when "
                                     "cuda_graph_capture_mode is valid."));
    auto &inner_graphs = *(
        graph_var
            ->GetMutable<std::vector<std::unique_ptr<CUDAGraphWithInOuts>>>());
    const size_t graph_idx = 2;
    if (inner_graphs[graph_idx].get() == nullptr) {
      framework::PEAndGraphPair pe_and_graph;
      auto callable =
          [this, &pe_and_graph](const framework::ExecutionContext &exe_ctx) {
            pe_and_graph = ComputeImpl(exe_ctx, true);
          };
      int64_t pool_id = inner_graphs[0].get() != nullptr
                            ? inner_graphs[0]->PoolID()
                            : inner_graphs[1]->PoolID();
      inner_graphs[graph_idx] =
          CaptureCUDAGraph(callable,
                           ctx,
                           {framework::GradVarName("Out")},
                           {framework::GradVarName("X")},
                           mode,
                           pool_id);
      VLOG(10) << "Capture Backward CUDA Graph";
    } else {
      ExecuteCUDAGraph(ctx,
                       {framework::GradVarName("Out")},
                       {framework::GradVarName("X")},
                       inner_graphs[graph_idx].get());
      VLOG(10) << "Run Backward CUDA Graph directly";
    }
#else
    PADDLE_THROW(
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
#endif
  }

 private:
  framework::PEAndGraphPair ComputeImpl(const framework::ExecutionContext &ctx,
                                        bool use_cuda_graph) const {
    VLOG(2) << "RunProgramGradOpKernel Compute";
    framework::PEAndGraphPair pe_and_graph;
    // Step 1. prepare inputs and outputs
    auto &output_grad_vars = ctx.MultiInputVar(framework::GradVarName("Out"));
    auto input_grad_vars = ctx.MultiOutputVar(framework::GradVarName("X"));
    auto param_grad_vars = ctx.MultiOutputVar(framework::GradVarName("Params"));

    // if all output vars are set to stop_gradient, grad op no need to executed
    if (input_grad_vars.empty() && param_grad_vars.empty()) {
      return pe_and_graph;
    }

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
    auto orig_end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto program_id = ctx.Attr<int64_t>("program_id");
    // NOTE: skip `shape` and `fill_constant` op created by
    // fluid.backward.gradients, one forward output will generate one `shape`
    // and `fill_constant`
    int64_t start_op_index = orig_end_op_index + (output_grad_vars.size() * 2);
    int64_t end_op_index = block->OpSize();

    auto *out_scope_vec = ctx.Input<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(
        out_scope_vec->size(),
        1,
        platform::errors::InvalidArgument(
            "The OutScope of RunProgramGradOp should only hold one scope."));

    framework::Scope *global_inner_scope = out_scope_vec->front();
    auto sub_scope_num = global_inner_scope->kids().size();
    VLOG(2) << "The number of sub scopes before backward: " << sub_scope_num;
    PADDLE_ENFORCE_GT(sub_scope_num,
                      0,
                      platform::errors::InvalidArgument(
                          "The OutScope of RunProgramGradOp should hold at "
                          "least one sub scope."));

    auto &scope = *(global_inner_scope->kids().front());
    auto *global_block = ctx.Attr<BlockDesc *>("global_block");

    if (end_op_index > start_op_index) {
      // Step 2. prepare executor and scope
      auto *program = global_block->Program();
      bool is_new_created;
      if (use_cuda_graph) {
        pe_and_graph = framework::CreateFixOrderExecutorInfo(
            *program, ctx.GetPlace(), start_op_index, end_op_index, &scope);
        is_new_created = true;
      } else {
        auto cache_info = framework::GetExecutorInfoFromCache(*program,
                                                              ctx.GetPlace(),
                                                              start_op_index,
                                                              end_op_index,
                                                              /*is_grad*/ true,
                                                              program_id,
                                                              &scope);
        pe_and_graph.first = cache_info.first;
        is_new_created = cache_info.second;
      }

      auto &parallel_executor = pe_and_graph.first;
      std::vector<std::string> tmp_vars;
      auto &skip_eager_delete_vars =
          use_cuda_graph
              ? tmp_vars
              : framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
                    program_id, true);
      if (is_new_created) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0,
                                           output_grad_var_names);

        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      input_grad_var_names.begin(),
                                      input_grad_var_names.end());
        framework::details::AppendSkipDeletionVars(param_grad_names,
                                                   &skip_eager_delete_vars);
      }

      details::ShareVarsIntoScope(
          output_grad_vars, output_grad_var_names, &scope);
      // Debug info: scope info when run end
      VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(
          /*skip_eager_delete_vars=*/skip_eager_delete_vars);
    }

    // Step 4. get outputs
    details::ShareVarsFromScope(
        input_grad_vars, input_grad_var_names, *global_block, &scope);
    details::ShareVarsFromScope(
        param_grad_vars, param_grad_names, *global_block, &scope);

    // Step5. drop current scope
    global_inner_scope->DeleteScope(&scope);
    VLOG(2) << "The number of sub scopes after backward: "
            << global_inner_scope->kids().size();
    return pe_and_graph;
  }
};

}  // namespace operators
}  // namespace paddle
