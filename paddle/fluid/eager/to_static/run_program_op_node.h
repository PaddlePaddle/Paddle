// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace details {
using Tensor = paddle::experimental::Tensor;

static std::vector<Tensor> DereferenceTensors(
    const std::vector<Tensor *> &tensor_ptr) {
  std::vector<Tensor> res;
  for (auto *t : tensor_ptr) {
    res.emplace_back(*t);
  }
  return res;
}

static std::vector<std::string> GetTensorsName(const std::vector<Tensor> &ins) {
  std::vector<std::string> in_names;
  for (auto &in_t : ins) {
    in_names.emplace_back(in_t.name());
  }
  return in_names;
}

static std::vector<std::string> GetTensorsName(
    const std::vector<Tensor *> &ins) {
  std::vector<std::string> in_names;
  for (auto *in_t : ins) {
    in_names.emplace_back(in_t->name());
  }
  return in_names;
}

static void CheckInputVarStatus(const Tensor &tensor) {
  PADDLE_ENFORCE_EQ(tensor.defined() && tensor.is_dense_tensor(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The input tensor %s of "
                        "RunProgram(Grad)Op holds "
                        "wrong type. Expect type is DenseTensor.",
                        tensor.name()));

  PADDLE_ENFORCE_EQ(
      static_cast<phi::DenseTensor *>(tensor.impl().get())->IsInitialized(),
      true,
      paddle::platform::errors::InvalidArgument(
          "The tensor in input tensor %s of "
          "RunProgram(Grad)Op "
          "is not initialized.",
          tensor.name()));
}

static void CheckOutputVarStatus(const paddle::framework::Variable &src_var,
                                 const Tensor &dst_tensor) {
  auto name = dst_tensor.name();
  PADDLE_ENFORCE_EQ(dst_tensor.defined(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "dst_tensor `%s` shall be defined.", name));

  if (dst_tensor.is_dense_tensor()) {
    auto &src_tensor = src_var.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::DenseTensor::classof(&src_tensor),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is DenseTensor",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.IsInitialized(),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          name));
  } else if (dst_tensor.is_selected_rows()) {
    auto &src_tensor = src_var.Get<phi::SelectedRows>();
    PADDLE_ENFORCE_EQ(phi::SelectedRows::classof(&src_tensor),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensodfr %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is SelectedRows",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.initialized(),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          name));

  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type LoDTensor or SelectedRows",
        name));
  }
}

static void ShareTensorsIntoScope(const std::vector<Tensor> &tensors,
                                  paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto name = tensors[i].name();
    if (name == "Fake_var") {
      continue;
    }
    auto *var = scope->Var(name);
    CheckInputVarStatus(tensors[i]);
    // share tensor
    auto tensor_base = tensors[i].impl();
    if (phi::DenseTensor::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor_base);
      *dst_tensor = *t;
    } else if (phi::SelectedRows::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::SelectedRows>();
      auto t = std::dynamic_pointer_cast<phi::SelectedRows>(tensor_base);
      *dst_tensor = *t;
    }
  }
}

static void ShareTensorsFromScope(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't find them in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    auto &name = tensors[i]->name();
    if (name == paddle::framework::kEmptyVarName || name == "Fake_var" ||
        !global_block.HasVar(name)) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        paddle::platform::errors::NotFound("The output tensor %s is not in "
                                           "RunProgram(Grad)Op'"
                                           "s internal scope.",
                                           name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      VLOG(2) << "share " << name << " from scope";
      *dst_tensor = src_tensor;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    }
  }
}

static void ShareTensorsFromScopeWithPartialBlock(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &forward_global_block,
    const paddle::framework::BlockDesc &backward_global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &name = tensors[i]->name();
    if (name == paddle::framework::kEmptyVarName || name == "Fake_var" ||
        (!forward_global_block.HasVar(name) &&
         !backward_global_block.HasVar(name))) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        paddle::platform::errors::NotFound("The output tensor %s is not in "
                                           "RunProgram(Grad)Op'"
                                           "s internal scope.",
                                           name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      VLOG(2) << "share " << name << " from scope";
      *dst_tensor = src_tensor;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    }
  }
}

static void BuildScopeByBlock(
    const paddle::framework::InterpreterCore &interpreter_core,
    const paddle::framework::BlockDesc &block,
    paddle::framework::Scope *scope) {
  for (auto &var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    if (var_name == paddle::framework::kEmptyVarName) {
      continue;
    }
    if (!scope->FindLocalVar(var_name)) {
      auto *ptr = scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(2) << "Initialize Block Variable " << var_name;
    }
  }
  auto &data_transfer_added_vars =
      interpreter_core.GetVariableScope()->DataTransferAddedVars();
  for (size_t i = 0; i < data_transfer_added_vars.size(); i++) {
    auto *ptr = scope->Var(data_transfer_added_vars[i].first);
    InitializeVariable(ptr,
                       static_cast<paddle::framework::proto::VarType::Type>(
                           data_transfer_added_vars[i].second));
    VLOG(2) << "Initialize Transfer Added Variable "
            << data_transfer_added_vars[i].first;
  }
}

static void GcScope(paddle::framework::Scope *scope) {
  std::deque<std::shared_ptr<paddle::memory::Allocation>> *garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();

  for (auto &var : scope->LocalVars()) {
    if (var != nullptr) {
      if (var->IsType<paddle::framework::LoDTensor>()) {
        garbages->emplace_back(var->GetMutable<paddle::framework::LoDTensor>()
                                   ->MoveMemoryHolder());
      }
      if (var->IsType<phi::SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
                                   ->mutable_value()
                                   ->MoveMemoryHolder());
      }
      if (var->IsType<paddle::framework::LoDTensorArray>()) {
        auto *lod_tensor_arr =
            var->GetMutable<paddle::framework::LoDTensorArray>();
        for (auto &t : *lod_tensor_arr) {
          garbages->emplace_back(t.MoveMemoryHolder());
        }
        lod_tensor_arr->clear();
      }
    }
  }
  delete garbages;  // free mem
}

}  // namespace details

inline void RunProgramAPI(
    const std::vector<paddle::experimental::Tensor> &x,
    const std::vector<paddle::experimental::Tensor> &params,
    std::vector<paddle::experimental::Tensor *> &out,     // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    std::vector<paddle::experimental::Tensor *> &dout,    // NOLINT
    const paddle::framework::AttributeMap &attrs) {
  VLOG(2) << "RunProgramOpKernel Compute";
  // In the original run_program OP, the default value of the is_test
  // attribute is false, we should check if there is is_test parameter
  // in attrs
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  auto program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));
  auto place = egr::Controller::Instance().GetExpectedPlace();

  // NOTE(chenweihang): In order not to add new variable type, use vector
  // here. Originally, here can use scope directly.
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      paddle::platform::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  bool use_interpretorcore =
      PADDLE_GET_CONST(bool, attrs.at("use_interpretorcore"));

  if (use_interpretorcore) {
    VLOG(0) << "RunProgramOp use interpretercore to execute program.";

    paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

    auto input_names = details::GetTensorsName(x);
    auto output_names = details::GetTensorsName(out);
    auto dout_names = details::GetTensorsName(dout);

    auto *forward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
    auto *backward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc *, attrs.at("backward_global_block"));
    auto *forward_program = forward_global_block->Program();
    auto *backward_program = backward_global_block->Program();

    auto &interpretercore_info_cache =
        paddle::framework::InterpreterCoreInfoCache::Instance();

    if (!interpretercore_info_cache.Has(program_id, /*is_grad=*/false)) {
      VLOG(2) << "No interpretercore cahce, so create a new interpretercore";
      // Step 1. share input_vars & parameters into scope
      details::ShareTensorsIntoScope(x, global_inner_scope);
      details::ShareTensorsIntoScope(params, global_inner_scope);
      // Step 2. create new interpretercore
      auto interpreter_core =
          paddle::framework::CreateInterpreterCoreInfoToCache(
              *forward_program,
              place,
              /*is_grad=*/false,
              program_id,
              global_inner_scope);
      // Step 3. get all eager gc vars
      std::set<std::string> skip_eager_delete_vars =
          paddle::framework::details::ParseSafeEagerDeletionSkipVarsSet(
              *backward_program);
      // all out_vars are skip_eager_var
      skip_eager_delete_vars.insert(output_names.begin(), output_names.end());
      skip_eager_delete_vars.insert(dout_names.begin(), dout_names.end());
      // update interpretercore skip_gc_var
      interpreter_core->SetSkipGcVars(skip_eager_delete_vars);
      interpretercore_info_cache.UpdateSkipEagerDeleteVars(
          program_id, false, skip_eager_delete_vars);
      VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
      // Step 4. interpretercore run
      if (forward_global_block->OpSize() > 0) {
        interpreter_core->Run({});
      }
      // Step 5. Get Output
      details::ShareTensorsFromScopeWithPartialBlock(out,
                                                     *forward_global_block,
                                                     *backward_global_block,
                                                     global_inner_scope);
      details::ShareTensorsFromScopeWithPartialBlock(dout,
                                                     *forward_global_block,
                                                     *backward_global_block,
                                                     global_inner_scope);
    } else {
      VLOG(2) << "Get interpretercore cahce by program:" << program_id;
      // Step 1. get cache interpretercore
      auto &cached_value =
          interpretercore_info_cache.GetMutable(program_id, /*is_grad=*/false);
      auto &interpreter_core = cached_value.core_;
      // Step 2. update scope for cache interpretercore
      details::ShareTensorsIntoScope(x, global_inner_scope);
      details::ShareTensorsIntoScope(params, global_inner_scope);
      if (interpreter_core->GetVariableScope()->GetMutableScope() !=
          global_inner_scope) {
        details::BuildScopeByBlock(
            *interpreter_core.get(), *forward_global_block, global_inner_scope);
        interpreter_core->reset_scope(global_inner_scope);
      }
      // Step 3. interpretercore run
      if (forward_global_block->OpSize() > 0) {
        interpreter_core->Run({});
      }
      // Step 4. Get Output
      details::ShareTensorsFromScopeWithPartialBlock(out,
                                                     *forward_global_block,
                                                     *backward_global_block,
                                                     global_inner_scope);
      details::ShareTensorsFromScopeWithPartialBlock(dout,
                                                     *forward_global_block,
                                                     *backward_global_block,
                                                     global_inner_scope);
    }
    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    if (is_test) {
      VLOG(4) << "is test, set this scope can reused";
      global_inner_scope->SetCanReuesd(true);
      details::GcScope(global_inner_scope);
    } else {
      VLOG(4) << "not test, set this scope can not reused";
      global_inner_scope->SetCanReuesd(false);
    }
#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
#endif
  } else {
    VLOG(2) << "RunProgramOp execute with parallel_executor.";

    // Step 2. prepare executor and init persistable variables
    // NOTE(Aurelius84): While training some models, forward can be called many
    // times and then apply backpropagation all at once, such as Reinforcement
    // Learning. Tensor data in multi-step training should be saved into single
    // scope separately. Otherwise, the gradients can be miscalculated because
    // always using the Tensor data of the last step in forward.
    paddle::framework::Scope *global_inner_scope = out_scope_vec->front();
    VLOG(2) << "The number of sub scopes before forward: "
            << out_scope_vec->front()->kids().size();
    paddle::framework::Scope &scope = global_inner_scope->NewScope();

    // share input_vars & parameters into scope
    details::ShareTensorsIntoScope(x, &scope);
    details::ShareTensorsIntoScope(params, &scope);

    const auto &place = egr::Controller::Instance().GetExpectedPlace();

    auto *global_block = PADDLE_GET_CONST(paddle::framework::BlockDesc *,
                                          attrs.at("global_block"));
    auto start_op_index = PADDLE_GET_CONST(int64_t, attrs.at("start_op_index"));
    auto end_op_index = PADDLE_GET_CONST(int64_t, attrs.at("end_op_index"));

    if (end_op_index > start_op_index) {
      auto input_names = details::GetTensorsName(x);
      auto output_names = details::GetTensorsName(out);
      auto dout_names = details::GetTensorsName(dout);
      auto *program = global_block->Program();

      auto cache_info =
          paddle::framework::GetExecutorInfoFromCache(*program,
                                                      place,
                                                      start_op_index,
                                                      end_op_index,
                                                      /*is_grad=*/false,
                                                      program_id,
                                                      &scope);
      auto &parallel_executor = cache_info.first;
      // all out_vars are skip_eager_var
      auto &skip_eager_delete_vars =
          paddle::framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, false);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_names);
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      output_names.begin(),
                                      output_names.end());
        skip_eager_delete_vars.insert(
            skip_eager_delete_vars.end(), dout_names.begin(), dout_names.end());
        paddle::framework::details::ParseSafeEagerDeletionSkipVars(
            *program, end_op_index, output_names, &skip_eager_delete_vars);
      }

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
    }
    // Step 4. Get Output
    details::ShareTensorsFromScope(out, *global_block, &scope);
    details::ShareTensorsFromScope(dout, *global_block, &scope);

    // Debug info: scope info when run end
    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());
    // Step 5. Drop all children scopes while testing.
    if (is_test) {
      out_scope_vec->front()->DropKids();
    }
    VLOG(2) << "The number of sub scopes after forward: "
            << out_scope_vec->front()->kids().size();
#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
#endif
  }
}

inline void RunProgramGradAPI(
    const std::vector<paddle::experimental::Tensor> &x,
    const std::vector<paddle::experimental::Tensor> &params,
    const std::vector<paddle::experimental::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::experimental::Tensor *> &x_grad,      // NOLINT
    std::vector<paddle::experimental::Tensor *> &params_grad  // NOLINT
) {
  // if all output vars are set to stop_gradient, grad op no need to executed
  if (x_grad.empty() && params_grad.empty()) return;

  bool use_interpretorcore =
      PADDLE_GET_CONST(bool, attrs.at("use_interpretorcore"));
  auto program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));

  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      paddle::platform::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  auto place = egr::Controller::Instance().GetExpectedPlace();

  if (use_interpretorcore) {
    VLOG(0) << "RunProgramGradOp use interpretercore to execute program.";

    paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

    auto *forward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
    auto *backward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc *, attrs.at("backward_global_block"));
    auto *backward_program = backward_global_block->Program();

    auto out_grad_names = details::GetTensorsName(out_grad);

    std::vector<std::string> x_grad_names;
    std::vector<std::string> param_grad_names;
    if (!x_grad.empty()) {
      x_grad_names = details::GetTensorsName(x_grad);
    }
    if (!params_grad.empty()) {
      param_grad_names = details::GetTensorsName(params_grad);
    }

    auto &interpretercore_info_cache =
        paddle::framework::InterpreterCoreInfoCache::Instance();
    if (!interpretercore_info_cache.Has(program_id, /*is_grad=*/true)) {
      VLOG(2) << "No interpretercore cahce, so create a new interpretercore";
      details::ShareTensorsIntoScope(out_grad, global_inner_scope);
      auto interpreter_core =
          paddle::framework::CreateInterpreterCoreInfoToCache(
              *backward_program,
              place,
              /*is_grad=*/true,
              program_id,
              global_inner_scope);

      // get all eager gc vars
      std::set<std::string> skip_eager_delete_vars;
      // all out_vars are skip_eager_var
      skip_eager_delete_vars.insert(x_grad_names.begin(), x_grad_names.end());
      // initialize skip gc vars by forward_program and backward_program
      paddle::framework::details::AppendSkipDeletionVars(
          param_grad_names, &skip_eager_delete_vars);
      interpreter_core->SetSkipGcVars(skip_eager_delete_vars);
      interpretercore_info_cache.UpdateSkipEagerDeleteVars(
          program_id, /*is_grad=*/true, skip_eager_delete_vars);
      VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
      if (backward_global_block->OpSize() > 0) {
        // Debug info: scope info when run end
        VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(
            out_scope_vec->front());
        interpreter_core->Run({});
      }
    } else {
      VLOG(2) << "Get interpretercore cahce by program:" << program_id;
      auto &cached_value =
          interpretercore_info_cache.GetMutable(program_id, /*is_grad=*/true);
      auto &interpreter_core = cached_value.core_;
      // update scope
      details::ShareTensorsIntoScope(out_grad, global_inner_scope);
      if (interpreter_core->GetVariableScope()->GetMutableScope() !=
          global_inner_scope) {
        details::BuildScopeByBlock(*interpreter_core.get(),
                                   *backward_global_block,
                                   global_inner_scope);
        interpreter_core->reset_scope(global_inner_scope);
      }

      if (backward_global_block->OpSize() > 0) {
        // Debug info: scope info when run end
        VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(
            out_scope_vec->front());
        interpreter_core->Run({});
      }
    }
    // Step 4. get outputs
    details::ShareTensorsFromScopeWithPartialBlock(x_grad,
                                                   *forward_global_block,
                                                   *backward_global_block,
                                                   global_inner_scope);
    details::ShareTensorsFromScopeWithPartialBlock(params_grad,
                                                   *forward_global_block,
                                                   *backward_global_block,
                                                   global_inner_scope);
    VLOG(4) << "after backward gc all vars";
    global_inner_scope->SetCanReuesd(true);
    details::GcScope(global_inner_scope);
  } else {
    VLOG(2) << "RunProgramGradOp use pe to execute program.";

    paddle::framework::Scope *global_inner_scope = out_scope_vec->front();
    auto sub_scope_num = global_inner_scope->kids().size();
    VLOG(2) << "The number of sub scopes before backward: " << sub_scope_num;
    PADDLE_ENFORCE_GT(sub_scope_num,
                      0,
                      paddle::platform::errors::InvalidArgument(
                          "The OutScope of RunProgramGradOp should hold at "
                          "least one sub scope."));

    auto &scope = *(global_inner_scope->kids().front());

    auto *global_block = PADDLE_GET_CONST(paddle::framework::BlockDesc *,
                                          attrs.at("global_block"));
    auto orig_end_op_index =
        PADDLE_GET_CONST(int64_t, attrs.at("end_op_index"));

    // NOTE: skip `shape` and `fill_constant` op created by
    // fluid.backward.gradients, one forward output will generate one `shape`
    // and `fill_constant`
    int64_t start_op_index = orig_end_op_index + (out_grad.size() * 2);
    int64_t end_op_index = global_block->OpSize();

    if (end_op_index > start_op_index) {
      auto out_grad_names = details::GetTensorsName(out_grad);
      // NOTE: after PR22939 [Add double grad] merged, the grad op maker's
      //   SetOutput will set to None if the input var stop_gradient=True,
      //   it will cause an NotFound error when ctx.OutputNames() is called
      std::vector<std::string> x_grad_names;
      std::vector<std::string> param_grad_names;
      if (!x_grad.empty()) {
        x_grad_names = details::GetTensorsName(x_grad);
      }
      if (!params_grad.empty()) {
        param_grad_names = details::GetTensorsName(params_grad);
      }

      // Step 2. prepare executor and scope
      auto *program = global_block->Program();
      auto cache_info =
          paddle::framework::GetExecutorInfoFromCache(*program,
                                                      place,
                                                      start_op_index,
                                                      end_op_index,
                                                      /*is_grad*/ true,
                                                      program_id,
                                                      &scope);
      auto &parallel_executor = cache_info.first;

      auto &skip_eager_delete_vars =
          paddle::framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, true);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, out_grad_names);

        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      x_grad_names.begin(),
                                      x_grad_names.end());
        paddle::framework::details::AppendSkipDeletionVars(
            param_grad_names, &skip_eager_delete_vars);
      }

      details::ShareTensorsIntoScope(out_grad, &scope);
      // Debug info: scope info when run end
      VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(
          out_scope_vec->front());

      // Step 3. run ops
      parallel_executor->RunWithoutFetch(
          /*skip_eager_delete_vars=*/skip_eager_delete_vars);
    }

    // Step 4. get outputs
    details::ShareTensorsFromScope(x_grad, *global_block, &scope);
    details::ShareTensorsFromScope(params_grad, *global_block, &scope);

    // Step5. drop current scope
    global_inner_scope->DeleteScope(&scope);
    VLOG(2) << "The number of sub scopes after backward: "
            << global_inner_scope->kids().size();
  }
}

class GradNodeRunProgram : public egr::GradNodeBase {
 public:
  GradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}

  ~GradNodeRunProgram() override = default;
  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph,
             bool is_new_grad) override {
    VLOG(3) << "Running Eager Backward Node: GradNodeRunProgram";
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>
        hooked_grads = GradNodeRunProgram::ApplyGradientHooks(grads);
    PADDLE_ENFORCE_EQ(hooked_grads.size(),
                      1,
                      paddle::platform::errors::InvalidArgument(
                          "The hooked_grads.size() of RunProgramGradOp should "
                          "be equal to 1."));

    egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                       this->InputMeta()[0]);
    VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
    std::vector<paddle::experimental::Tensor> x_grad;
    std::vector<paddle::experimental::Tensor> params_grad;
    ConstructXGradTensors(x_, &x_grad);
    ConstructParamGradTensors(params_, &params_grad);
    std::vector<paddle::experimental::Tensor *> x_grad_ptr;
    std::vector<paddle::experimental::Tensor *> params_grad_ptr;
    for (auto &i : x_grad) {
      x_grad_ptr.emplace_back(&i);
    }
    for (auto &i : params_grad) {
      if (i.defined()) {
        params_grad_ptr.emplace_back(&i);
      }
    }

    PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                      fwd_out_names_.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The hooked_grads[0].size() and "
                          "fwd_out_names_.size() should be equal."));
    for (size_t i = 0; i < fwd_out_names_.size(); ++i) {
      hooked_grads[0][i].set_name(fwd_out_names_[i] + "@GRAD");
    }
    RunProgramGradAPI(x_,
                      params_,
                      hooked_grads[0],
                      step_scope_,
                      attrs_,
                      x_grad_ptr,
                      params_grad_ptr);
    VLOG(3) << "End Eager Backward Node: GradNodeRunProgram";
    return {x_grad, params_grad};
  }

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &attrs) {
    attrs_ = attrs;
  }

  void SetFwdX(const std::vector<paddle::experimental::Tensor> &tensors) {
    x_ = tensors;
  }

  void SetFwdParams(const std::vector<paddle::experimental::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

  void SetFwdOutNames(std::vector<std::string> out_names) {
    fwd_out_names_ = out_names;
  }

 protected:
  void ConstructXGradTensors(
      const std::vector<paddle::experimental::Tensor> &x,
      std::vector<paddle::experimental::Tensor> *x_grad) {
    // TODO(dev): Need an elegant way to determine inforamtion of grad_tensor,
    // such as: name, tensor type(DenseTensor or SelectedRows).
    for (auto &t : x) {
      if (t.is_dense_tensor()) {
        x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (t.is_selected_rows()) {
        x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      x_grad->back().set_name(t.name() + "@GRAD");
    }
  }

  void ConstructParamGradTensors(
      const std::vector<paddle::experimental::Tensor> &param,
      std::vector<paddle::experimental::Tensor> *param_grad) {
    for (auto &t : param) {
      auto t_grad = egr::EagerUtils::unsafe_autograd_meta(t)->Grad();
      // In eager mode, the number of param_grad should be the same as
      // param, so here an empty Tensor is added for the param with
      // stop_gradient=True
      if (!t_grad.defined()) {
        param_grad->emplace_back();
      } else if (t_grad.is_dense_tensor()) {
        param_grad->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (t_grad.is_selected_rows()) {
        param_grad->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      param_grad->back().set_name(t.name() + "@GRAD");
    }
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GradNodeRunProgram>(new GradNodeRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::experimental::Tensor> x_;
  std::vector<paddle::experimental::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  std::vector<std::string> fwd_out_names_;

  // Attribute Map
  paddle::framework::AttributeMap attrs_;
};
