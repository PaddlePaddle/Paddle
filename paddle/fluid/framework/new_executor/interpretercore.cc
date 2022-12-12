// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpretercore.h"

#include <unordered_set>

#include "gflags/gflags.h"

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_functor.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/phi/backends/device_manager.h"

PADDLE_DEFINE_EXPORTED_bool(new_executor_use_inplace,
                            false,
                            "Use inplace in new executor");
PADDLE_DEFINE_EXPORTED_bool(new_executor_use_local_scope,
                            true,
                            "Use local_scope in new executor(especially used "
                            "in UT), can turn off for better performance");
PADDLE_DEFINE_EXPORTED_bool(control_flow_use_new_executor,
                            false,
                            "Use new executor in control flow op");

DECLARE_bool(check_nan_inf);
DECLARE_bool(benchmark);

constexpr const char* kExceptionCaught = "ExceptionCaught";
constexpr const char* kTaskCompletion = "TaskCompletion";

namespace paddle {
namespace framework {

inline void SetDeviceId(const platform::Place& place) {
  // TODO(zhiqiu): reduce the cost
  if (platform::is_gpu_place(place)) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
    PADDLE_THROW(platform::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with CUDA support.",
        place));
#else
    auto dev_id = place.device;
    platform::SetDeviceId(dev_id);
#endif
  } else if (platform::is_xpu_place(place)) {
#ifndef PADDLE_WITH_XPU
    PADDLE_THROW(platform::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with XPU support.",
        place));
#else
    auto dev_id = place.device;
    platform::SetXPUDeviceId(dev_id);
#endif
  } else if (platform::is_npu_place(place)) {
#ifndef PADDLE_WITH_ASCEND_CL
    PADDLE_THROW(platform::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with NPU support.",
        place));
#else
    auto dev_id = place.device;
    platform::SetNPUDeviceId(dev_id);
#endif
  } else if (platform::is_custom_place(place)) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    PADDLE_THROW(platform::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with CustomDevice support.",
        place));
#else
    phi::DeviceManager::SetDevice(place);
#endif
  }
}

// TODO(Ruibiao): Pass skip_gc_vars, used_for_jit, and other config messages by
// constructing an interpreter::ExecutionConfig
InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const BlockDesc& block,
                                 const std::set<std::string>& skip_gc_vars,
                                 framework::Scope* scope,
                                 bool used_for_jit,
                                 bool used_for_control_flow_op)
    : place_(place),
      block_(block),
      execution_config_(place, block.OpSize()),
      stream_analyzer_(place),
      var_scope_(scope) {
  VLOG(4) << "InterpreterCore(): " << this << " on " << place_;

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  execution_config_.used_for_jit = used_for_jit;
  execution_config_.used_for_control_flow_op = used_for_control_flow_op;
  execution_config_.create_local_scope = !used_for_jit &&
                                         FLAGS_new_executor_use_local_scope &&
                                         !used_for_control_flow_op;
  execution_config_.skip_gc_vars = skip_gc_vars;
  execution_config_.Log(/*log_level=*/8);

  if (execution_config_.create_local_scope) {
    auto local_scope = &var_scope_.GetMutableScope()->NewScope();
    local_scope_ = local_scope;
  }
  var_scope_.SetLocalScope(local_scope_);
}

InterpreterCore::~InterpreterCore() {
  // cancle gc's thread
  gc_.reset(nullptr);
  async_work_queue_.reset();
  VLOG(4) << "~InterpreterCore(): " << this << " on " << place_;

#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

interpreter::CostInfo InterpreterCore::DryRun(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors) {
  SetDeviceId(place_);

  Prepare(feed_names, feed_tensors, true);
  interpreter::CostInfo cost_info;
  {
    interpreter::ProfilerGuard(place_, &cost_info);

    // For the program that only run once, it is no need to
    // create work_queue, so the async_work_queue_ is created
    // until the second step run.
    async_work_queue_ = GetWorkQueue();

    // lazy initialization of gc, do not create gc is the program only run once
    if (!gc_) {
      gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_);
    }

    ExecuteInstructionList(vec_instruction_);
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  return cost_info;
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors) {
  SetDeviceId(place_);

#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  bool is_build = is_build_;
  Prepare(feed_names, feed_tensors, is_build);

  if (is_build) {
    // For the program that only run once, it is no need to
    // create work_queue, so the async_work_queue_ is created
    // until the second step run.
    async_work_queue_ = GetWorkQueue();

    // lazy initialization of gc, do not create gc is the program only run once
    if (!gc_) {
      gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_);
    }

    if (execution_config_.used_for_jit && (sync_op_num_ == 0)) {
      VLOG(4) << "Tracing Instruction List";
      TraceInstructionList(vec_instruction_);
    } else {
      ExecuteInstructionList(vec_instruction_);
    }
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(place_)) {
      platform::DeviceContextPool::Instance().Get(place_)->Wait();
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (platform::is_custom_place(place_)) {
      platform::DeviceContextPool::Instance().Get(place_)->Wait();
    }
#endif
  }
  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  auto* fetch_var = local_scope_->FindVar(interpreter::kFetchVarName);
  if (fetch_var) {
    return std::move(*fetch_var->GetMutable<framework::FetchList>());
  } else {
    return {};
  }
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names, bool need_fetch) {
  SetDeviceId(place_);

#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  if (!is_build_) {
    LOG_FIRST_N(INFO, 1) << "New Executor is Running.";
    paddle::framework::interpreter::BuildVariableScope(
        block_, &var_scope_, HasLocalScope());

    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::BuildOpFuncList(
        place_,
        block_,
        execution_config_.skip_gc_vars,
        &op_func_nodes,
        &var_scope_,
        execution_config_,
        HasLocalScope());
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
    is_build_ = true;
    UpdateSyncOpNum();
  } else {
    // For the program that only run once, it is no need to
    // create work_queue, so the async_work_queue_ is created
    // until the second step run.
    async_work_queue_ = GetWorkQueue();

    // lazy initialization of gc, do not create gc is the program only run once
    if (!gc_) {
      gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_);
    }

    if (execution_config_.used_for_jit && (sync_op_num_ == 0)) {
      VLOG(4) << "Tracing Instruction List";
      TraceInstructionList(vec_instruction_);
    } else {
      ExecuteInstructionList(vec_instruction_);
    }
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(place_)) {
      platform::DeviceContextPool::Instance().Get(place_)->Wait();
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (platform::is_custom_place(place_)) {
      platform::DeviceContextPool::Instance().Get(place_)->Wait();
    }
#endif
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  Scope* inner_scope =
      HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
  auto* fetch_var = inner_scope->FindVar(interpreter::kFetchVarName);
  if (fetch_var && need_fetch) {
    return std::move(*fetch_var->GetMutable<framework::FetchList>());
  } else {
    return {};
  }
}

void InterpreterCore::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  copy_program_ = prog;
}

void InterpreterCore::SetSkipGcVars(const std::set<std::string>& skip_gc_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.skip_gc_vars.empty(),
      true,
      platform::errors::PreconditionNotMet(
          "execution_config_.skip_gc_vars can only be initialized once, now "
          "execution_config_.skip_gc_vars is "
          "not empty, do not call SetSkipGcVars method repeatedly."));
  execution_config_.skip_gc_vars = skip_gc_vars;
}

void InterpreterCore::SetJitInputVars(
    const std::set<std::string>& jit_input_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.jit_input_vars.empty(),
      true,
      platform::errors::PreconditionNotMet(
          "execution_config_.jit_input_vars can only be initialized once, now "
          "execution_config_.jit_input_vars is "
          "not empty, do not call SetJitInputVars method repeatedly."));
  execution_config_.jit_input_vars = jit_input_vars;
}

const std::set<std::string>& InterpreterCore::JitInputVars() const {
  return execution_config_.jit_input_vars;
}

const VariableScope* InterpreterCore::GetVariableScope() const {
  return &var_scope_;
}

void InterpreterCore::reset_scope(Scope* new_scope) {
  var_scope_.SetScope(new_scope);
  auto& var_list = var_scope_.MutableVarList();
  for (size_t i = 0; i < var_list.size(); i++) {
    const auto& var_name = var_scope_.GetNameById(i);
    var_list[i] = new_scope->FindVar(var_name);
  }
  // The index should assured valid, cause the InterpreterCore may not be fully
  // built, but was still cached and used. For example, see unit test
  // `test_assert.py`, it may exit before `InterpreterCore::Convert`, but still
  // was cached and used by later tests.
  for (size_t i = 0; i < std::min(refs_.size(), var_list.size()); i++) {
    refs_[i]->ResetVariable(var_list[i]);
  }

  for (size_t i = 0; i < vec_instruction_.size(); i++) {
    BuildAndCacheInstructionCtx(&vec_instruction_[i]);
  }
}

void InterpreterCore::ShareWorkQueueFrom(std::shared_ptr<InterpreterCore> src) {
  async_work_queue_ = src->GetWorkQueue();
  VLOG(8) << "Share AsyncWorkQueue from InterpreterCore(" << src.get()
          << ") to InterpreterCore(" << this << ")";
}

bool InterpreterCore::BuildInplaceCheckVarIsOnlyInput(
    const std::vector<std::vector<size_t>>& input_var2op, size_t var_index) {
  if (!var_scope_.VarDesc(var_index)) {
    return input_var2op.at(var_index).size() == 1;
  } else {
    int is_input_cnt = 0;
    for (auto inst_id : input_var2op.at(var_index)) {
      OpInOutInfo info;
      info.Build(vec_instruction_.at(inst_id).OpBase());
      if (info.IsInArgBufferNeeded(var_scope_.VarDesc(var_index)->Name())) {
        is_input_cnt++;
      }
    }
    return is_input_cnt == 1;
  }
}

std::shared_ptr<interpreter::AsyncWorkQueue> InterpreterCore::GetWorkQueue() {
  if (async_work_queue_ == nullptr) {
    async_work_queue_ = std::make_shared<interpreter::AsyncWorkQueue>(
        execution_config_.host_num_threads,
        execution_config_.deivce_num_threads,
        &main_thread_blocker_);
  }
  return async_work_queue_;
}

void InterpreterCore::BuildAndCacheInstructionCtx(Instruction* instr_node) {
  Scope* inner_scope =
      HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
  VariableValueMap ins_map;
  for (auto& var_name_item : instr_node->Inputs()) {
    std::vector<Variable*> input_vars;

    input_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      input_vars.emplace_back(inner_scope->FindVar(var_scope_.GetNameById(id)));
    }
    ins_map.emplace(var_name_item.first, std::move(input_vars));
  }

  VariableValueMap outs_map;
  for (auto& var_name_item : instr_node->Outputs()) {
    std::vector<Variable*> out_vars;

    out_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      out_vars.emplace_back(inner_scope->FindVar(var_scope_.GetNameById(id)));
    }
    outs_map.emplace(var_name_item.first, std::move(out_vars));
  }

  // set runtime_ctx and infershape_ctx_
  if (instr_node->OpBase()->Type() == "cinn_launch") {  // OP use scope in
                                                        // kernel
    Scope* local_scope = HasLocalScope() ? var_scope_.GetMutableLocalScope()
                                         : var_scope_.GetMutableScope();
    instr_node->ResetContextWithScope(ins_map, outs_map, *local_scope);
  } else {
    instr_node->ResetContext(ins_map, outs_map);
  }
}

void InterpreterCore::BuildInplace() {
  // NOTE(Ruibiao): coalesce_tensor_op outputs a FusedOutput phi::DenseTensor
  // and a list of Output Tensors which are sliced from the FusedOutput. These
  // outputs sholud not be the outvar of the in-place var-pair since memory
  // reuse between FusedOutput and Output Tensors is assumed. For the following
  // example:
  // fused_var, var1, var2, var3 = coalesce_tensor(var1, var2, var3)
  // var1 = sum(var4, var5)
  // ...
  //
  // After running coalesce_tensor_op, var1 is assumed to share the buffer
  // slices from fused_var. However, if sum_op is in-place, then var1 would
  // re-share the buffer with var4 instead of fused_var.
  std::set<std::string> skip_inplace_outvars;
  for (Instruction& instr : vec_instruction_) {
    OperatorBase* op = instr.OpBase();
    if (op->Type() == kCoalesceTensor) {
      const std::vector<std::string>& outputs =
          op->OutputVars(/*has_intermediate=*/false);
      skip_inplace_outvars.insert(outputs.begin(), outputs.end());
    }
  }

  Scope* local_scope = HasLocalScope() ? var_scope_.GetMutableLocalScope()
                                       : var_scope_.GetMutableScope();
  std::vector<std::vector<size_t>> input_var2op(var_scope_.VarSize());
  for (Instruction& instr : vec_instruction_) {
    for (auto& item : instr.Inputs()) {
      for (int var_id : item.second) {
        if (var_id != kEmptyVarIndex) {
          input_var2op.at(var_id).push_back(instr.Id());
        }
      }
    }
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    auto& instr = vec_instruction_[i];
    auto* op_base = instr.OpBase();
    if (!op_base->Info().infer_inplace_) {
      continue;
    }

    auto in_to_outs = op_base->Info().infer_inplace_(
        platform::is_gpu_place(instr.DeviceContext().GetPlace()));

    auto& inputs = instr.Inputs();
    auto& outputs = instr.Outputs();
    for (auto& pair : in_to_outs) {
      auto iter = inputs.find(pair.first);
      if (iter != inputs.end() && !iter->second.empty()) {
        auto in_var_desc = var_scope_.VarDesc(iter->second[0]);
        if (in_var_desc && in_var_desc->Persistable()) {
          continue;
        }
        if (var_scope_.GetVarSikpInplace(iter->second[0])) {
          continue;
        }
        if (BuildInplaceCheckVarIsOnlyInput(input_var2op, iter->second[0])) {
          auto iterout = outputs.find(pair.second);
          if (iterout != outputs.end() && !iterout->second.empty()) {
            const std::string& invar_name =
                var_scope_.GetNameById(iter->second[0]);
            const std::string& outvar_name =
                var_scope_.GetNameById(iterout->second[0]);
            auto invar = local_scope->FindVar(invar_name);
            auto outvar = local_scope->FindVar(outvar_name);

            if (invar && outvar && invar->IsType<phi::DenseTensor>() &&
                outvar->IsType<phi::DenseTensor>() &&
                skip_inplace_outvars.find(outvar_name) ==
                    skip_inplace_outvars.end()) {
              instr.AddInplace(invar, outvar);
              VLOG(3) << "inplace " << op_base->Type() << " " << invar_name
                      << " -> " << outvar_name;
            }
          }
        }
      }
    }
  }
}

void InterpreterCore::BuildOperatorDependences() {
  // analysis the dependences between ops, add next_instr_list to each instr,
  // and set the dependecy_count_
  size_t instr_num = vec_instruction_.size();
  dependecy_count_.resize(instr_num);
  auto downstream_map = dependency_builder_.Build(vec_instruction_);

  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    Instruction& cur_instr = vec_instruction_[instr_id];
    const std::set<size_t>& next_instr_ids = downstream_map[instr_id];

    if (cur_instr.KernelType() == OpFuncType::kGpuAsync) {
      for (size_t next_instr_id : next_instr_ids) {
        if (vec_instruction_[next_instr_id].KernelType() ==
            OpFuncType::kGpuAsync) {
          cur_instr.AddNextInstrInSameThread(next_instr_id);
        } else {
          cur_instr.AddNextInstrInDifferentThread(next_instr_id);
        }
      }
    } else {
      bool has_instr_in_same_thread = false;
      for (size_t next_instr_id : next_instr_ids) {
        if (!has_instr_in_same_thread &&
            vec_instruction_[next_instr_id].KernelType() !=
                OpFuncType::kGpuAsync) {
          cur_instr.AddNextInstrInSameThread(next_instr_id);
          has_instr_in_same_thread = true;
        } else {
          cur_instr.AddNextInstrInDifferentThread(next_instr_id);
        }
      }
    }

    for (size_t next_instr_id : next_instr_ids) {
      ++dependecy_count_[next_instr_id];
    }
  }
}

// At the end of each step, the holder of phi::DenseTensor in LoDTensorArray is
// null. Clear these Tensors and leave LoDTensorArray empty, otherwise an
// exception will occur in the next step
void InterpreterCore::ClearLoDTensorArrayInLocalScope() {
  auto vars = local_scope_->LocalVars();
  for (auto var : vars) {
    if (var->IsType<LoDTensorArray>()) {
      auto* lod_tensor_arr = var->GetMutable<LoDTensorArray>();
      lod_tensor_arr->clear();
    }
  }
}

void InterpreterCore::Convert(
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  auto& vec_meta_info = var_scope_.MutableVecMetaInfo();
  auto nodes = *op_func_nodes;
  auto op_nums = nodes.size();
  vec_instruction_.reserve(op_nums);
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    auto& op_func_node = nodes[op_idx];
    auto* dev_ctx_ = stream_analyzer_.ParseDeviceContext(op_func_node);
    Priority priority =
        interpreter::IsCommunicationOp(op_func_node.operator_base_->Type())
            ? Priority::kLowest
            : Priority::kNormal;
    vec_instruction_.emplace_back(
        op_idx, std::move(op_func_node), *dev_ctx_, priority);
  }

  BuildOperatorDependences();

  // NOTE(Ruibiao): For cross-step stream synchronization, an event may be
  // recorded in the first step and waited in the second step. So, in the first
  // step, the WaitEvent may be called without RecordEvent. Considering that
  // before the first call to RecordEvent, an Event represents an empty set of
  // work and WaitEvent always return succeed immediately, we omit the
  // prelude-record for the first step here.
  stream_analyzer_.ConstructEvents(&vec_instruction_);

  // add event for the input var of jit program, since there are async copied
  // from gpu_pinned place to gpu place on compute stream.
  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      auto& inst = vec_instruction_[i];
      if (inst.OpBase()->Type() == interpreter::kMemcpyD2H &&
          platform::is_gpu_place(place_)) {
        for (auto& item : inst.Inputs()) {
          for (auto var_id : item.second) {
            auto name = var_scope_.GetNameById(var_id);
            if (JitInputVars().count(name)) {
              auto device_event = std::make_shared<platform::DeviceEvent>(
                  place_, platform::GenerateDeviceEventFlag());
              VLOG(4) << "Add input event for input: " << name << " of "
                      << inst.OpBase()->Type();
              inst.AddEventToWait(
                  i, device_event, stream_analyzer_.GetWaiterType(inst));
            }
          }
        }
      }
    }
  }

  // calculate last_live_ops_
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    Instruction& instr = vec_instruction_[op_idx];
    OpInOutInfo info;
    info.Build(instr.OpBase());

    std::set<size_t> gc_check_vars;

    const std::map<std::string, std::vector<int>>& ins = instr.Inputs();
    const std::map<std::string, std::vector<int>>& outs = instr.Outputs();
    std::multimap<std::string, std::vector<int>> ins_and_outs{ins.begin(),
                                                              ins.end()};
    ins_and_outs.insert(outs.begin(), outs.end());

    for (auto& item : ins_and_outs) {
      for (auto id : item.second) {
        if (id == kEmptyVarIndex) {
          continue;
        }
        auto* var_desc = var_scope_.VarDesc(id);
        // skip no_need_buffer input vars
        if (var_desc && ins.count(item.first) &&
            !info.IsInArgBufferNeeded(var_desc->Name())) {
          continue;
        }
        // skip when this var is not in block and not a data_transferred var,
        // which means this var is managed by other block
        const auto& var_name = var_scope_.GetNameById(id);
        bool not_owned = !block_.HasVar(var_name);
        const auto& transferred_vars = var_scope_.DataTransferAddedVars();
        bool not_transferred =
            std::all_of(transferred_vars.begin(),
                        transferred_vars.end(),
                        [&](const std::pair<std::string, int>& elem) {
                          return elem.first != var_name;
                        });
        if (not_owned && not_transferred) {
          VLOG(10) << "[gc_check_inputs] skip gc: " << var_name;
          continue;
        }
        gc_check_vars.insert(id);
      }
    }

    for (auto var_id : gc_check_vars) {
      Scope* inner_scope =
          HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
      paddle::framework::Variable* var =
          inner_scope->FindVar(var_scope_.GetNameById(var_id));
      if (var->IsType<phi::DenseTensor>() || var->IsType<phi::SelectedRows>() ||
          var->IsType<LoDTensorArray>()) {
        last_live_ops_[var_id].insert(op_idx);
      } else {
        VLOG(4) << "not clear " << var_scope_.GetNameById(var_id) << " after "
                << instr.OpBase()->Type() << " because its type is "
                << framework::ToTypeName(var->Type());
      }
    }
  }

  // clear the last_live_ops list for all vars in skip_gc_vars
  for (const std::string& skip_gc_var : execution_config_.skip_gc_vars) {
    int var_id = var_scope_.GetIdByName(skip_gc_var);
    if (var_id != -1) {
      last_live_ops_[var_id].clear();
      VLOG(8) << "Skip gc for var: " << skip_gc_var;
    }
  }

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  // For example,
  // b = op1(a)
  // c = op2(a, b)
  // in this case, a is the input of op1 and op2, we only need to check
  // a after op2, because op2 always uses a after op1.
  for (size_t i = 0; i < last_live_ops_.size(); ++i) {
    std::set<size_t> minumum_last_live_ops;
    for (size_t item : last_live_ops_[i]) {
      bool not_before_any = true;
      // find the op that is not executed before any
      for (size_t other_item : last_live_ops_[i]) {
        if (dependency_builder_.OpHappensBefore(item, other_item)) {
          VLOG(8) << "happens_before: " << item << "->" << other_item
                  << ", so skip " << item;
          not_before_any = false;
          break;
        }
      }
      if (not_before_any) {
        VLOG(8) << "last live op of var " << i << " "
                << var_scope_.GetNameById(i) << " : " << item << " "
                << vec_instruction_[item].OpBase()->Type();
        minumum_last_live_ops.insert(item);
        vec_instruction_[item].AddGCCheckVar(i);
      }
    }
    last_live_ops_[i] = minumum_last_live_ops;
    vec_meta_info[i].var_ref_count_ = last_live_ops_[i].size();
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    BuildAndCacheInstructionCtx(&vec_instruction_[i]);
  }

  BuildSkipShareLoDInfo();

  bool inplaced = false;
  for (const Instruction& inst : vec_instruction_) {
    if (inst.OpBase()->Type() == "share_buffer" ||
        inst.OpBase()->Type() == "share_data") {
      VLOG(4) << "Already inplaced, skip inplace now.";
      inplaced = true;
    }
  }

  if (FLAGS_new_executor_use_inplace && !inplaced) {
    BuildInplace();
  }

  for (auto& dep : dependecy_count_) {
    deps_.emplace_back(std::make_shared<interpreter::OpDepInfo>(dep));
  }
  for (size_t i = 0; i < vec_meta_info.size(); ++i) {
    refs_.emplace_back(std::make_shared<interpreter::VarRefInfo>(
        vec_meta_info[i].var_ref_count_, var_scope_.VarRef(i)));
  }

  AnalyseExecuteOrderForTrace();
}

void InterpreterCore::BuildSkipShareLoDInfo() {
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    bool can_skip_lod = true;
    for (auto& input : vec_instruction_[i].InnerRuntimeContext()->inputs) {
      for (auto& var : input.second) {
        if (var->IsType<phi::DenseTensor>()) {
          if (var->Get<phi::DenseTensor>().lod().size() != 0) {
            can_skip_lod = false;
            break;
          }
        } else {
          can_skip_lod = false;
          break;
        }
      }
    }
    vec_instruction_[i].InnerInferShapeContext()->SetSkipLoD(can_skip_lod);
  }
}

void InterpreterCore::RunOperator(const Instruction& instr_node) {
  auto* op = instr_node.OpBase();
  auto place = instr_node.DeviceContext().GetPlace();
  Scope* local_scope = HasLocalScope() ? var_scope_.GetMutableLocalScope()
                                       : var_scope_.GetMutableScope();
  VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

  SetDeviceId(place);

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place)) {
    // NOTE(wangxi): nan/inf cannot be detected on NPU by checking the
    // variable values, but only through special `float_status` to checks
    // whether the operation is overflow. More about `float_status`, see:
    // https://gitee.com/ascend/modelzoo/issues/I3NF8V?from=project-issue
    if (FLAGS_check_nan_inf) {
      framework::details::NPUAllocAndClearFloatStatus(*op, *local_scope, place);
    }
  }
#endif

  auto op_with_kernel = dynamic_cast<const framework::OperatorWithKernel*>(op);
  {
    // If it is OperatorBase, InferShape do nothing.
    if (op_with_kernel != nullptr) {
      platform::RecordEvent infershape_event(
          "infer_shape",
          platform::TracerEventType::OperatorInner,
          1,
          platform::EventRole::kInnerOp);

      // see OperatorWithKernel::RunImpl in operator.cc for why
      if (!(op_with_kernel->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op_with_kernel->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        op_with_kernel->Info().infer_shape_(
            instr_node.InnerInferShapeContext().get());
      }
      infershape_event.End();
      platform::RecordOpInfoSupplement(op->Type(),
                                       op->Attrs(),
                                       *(instr_node.InnerInferShapeContext()),
                                       *(instr_node.InnerRuntimeContext()),
                                       op->Id());
    }
  }
  if (op_with_kernel != nullptr && FLAGS_new_executor_use_inplace) {
    // TODO(xiongkun03) Does operator base support inplace ?
    for (auto& pair : instr_node.InplaceInfo()) {
      const auto& in = paddle::framework::details::GetTensorFromVar(pair.first);
      auto* out =
          paddle::framework::details::GetMutableTensorFromVar(pair.second);
      if (in.dims() == out->dims()) {
        out->ShareBufferWith(in);
      }
    }
  }

  {
    platform::RecordEvent compute_event(
        "compute",
        platform::TracerEventType::OperatorInner,
        1,
        platform::EventRole::kInnerOp);
    if (op_with_kernel == nullptr) {
      instr_node.OpBase()->Run(*local_scope, place_);
    } else {
      // fit for phi
      if (instr_node.PhiKernel() && instr_node.PhiKernel()->IsValid()) {
        VLOG(4) << "Run phi kernel: " << op->Type();
        VLOG(4) << instr_node.InnerRuntimeContext().get() << " "
                << &instr_node.DeviceContext();
        phi::KernelContext phi_kernel_context;
        op_with_kernel->BuildPhiKernelContext(
            *instr_node.InnerRuntimeContext().get(),
            const_cast<platform::DeviceContext*>(&instr_node.DeviceContext()),
            &phi_kernel_context);

        (*instr_node.PhiKernel())(&phi_kernel_context);

      } else {
        instr_node.KernelFunc()(*instr_node.InnerExecutionContext().get());
      }
    }
  }

  VLOG(4) << "End run " << place << " " << op->DebugStringEx(local_scope);

  if (!instr_node.InplaceBackMap().empty()) {
    platform::RecordEvent inplaceback_event(
        "InplaceVarsBack", platform::TracerEventType::UserDefined, 10);
    auto& m = instr_node.InplaceBackMap();
    // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in operator.cc
    for (auto& p : m) {
      auto* transformed_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
          var_scope_.VarRef(p.first));
      auto* original_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
          var_scope_.VarRef(p.second));
      original_tensor->ShareDataWith(*transformed_tensor);
      VLOG(4) << "Transfer inplace variable back form "
              << var_scope_.GetNameById(p.first) << " to "
              << var_scope_.GetNameById(p.second);
    }
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    instr_node.DeviceContext().Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op->Type()
            << "): context wait and get last error";
#endif
  }

  // for debug nan/inf
  if (op_with_kernel != nullptr && FLAGS_check_nan_inf) {
    VLOG(4) << "Check nan/inf";
    framework::details::CheckOpHasNanOrInf(
        *op,
        *local_scope,
        place);  // TODO(xiongkun03) change it to inner scope.
  }
}

void InterpreterCore::RunInstruction(const Instruction& instr_node) {
  VLOG(5) << __func__ << " OP id:" << instr_node.Id()
          << " name:" << instr_node.OpBase()->Type() << " type:"
          << (instr_node.KernelType() == OpFuncType::kCpuSync
                  ? "kCpuSync"
                  : (instr_node.KernelType() == OpFuncType::kGpuSync
                         ? "kGpuSync"
                         : "kGpuAsync"))
          << " runs on " << platform::GetCurrentThreadName();

  auto* op = instr_node.OpBase();
  platform::RecordEvent instruction_event(
      op->Type(), platform::TracerEventType::Operator, 1);

  try {
    instr_node.WaitEvent(place_);

    if (!instr_node.IsArtificial()) {
      RunOperator(instr_node);
      CheckGC(instr_node);
      interpreter::LogDeviceMemoryStats(place_);
    }

    instr_node.RecordEvent(place_);
  } catch (platform::EnforceNotMet& ex) {
    framework::InsertCallStackInfo(op->Type(), op->Attrs(), &ex);
    exception_holder_.Catch(std::make_exception_ptr(std::move(ex)));
  } catch (platform::EOFException&) {
    exception_holder_.Catch(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << op->Type() << " raises an exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::current_exception());
  } catch (...) {
    LOG(WARNING) << op->Type() << " raises an unknown exception";
    exception_holder_.Catch(std::current_exception());
  }
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr) {
  interpreter::ResetAtomicGuard guard(&deps_, &refs_);
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i));
      async_work_queue_->AddTask(vec_instr.at(i).KernelType(),
                                 [this, i] { RunInstructionAsync(i); });
    }
  }

  auto event_name = main_thread_blocker_.WaitEvent();
  VLOG(1) << "main_thread_blocker_(" << &main_thread_blocker_
          << ") got event_name: " << event_name;

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    // Graceful exit when the executor encountered a fatal error.
    // EOF is not a fatal error.
    if (exception_holder_.Type() != "EOF") {
      async_work_queue_->Cancel();
    }
    VLOG(4) << "Cancel ok";
    PADDLE_ENFORCE_EQ(
        main_thread_blocker_.Clear(),
        0,
        platform::errors::PreconditionNotMet(
            "main_thread_blocker_.Clear() return -1, clear failed"));
    VLOG(4) << "clear ok";
    exception_holder_.ReThrow();
  }
}

void InterpreterCore::RunNextInstructions(
    const Instruction& instr, std::deque<size_t>* reserved_next_ops) {
  platform::RecordEvent record(
      "RunNextInstructions", platform::TracerEventType::UserDefined, 10);

  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  for (size_t next_instr_id : instr.NextInstrsInDifferenceThread()) {
    if (IsReady(next_instr_id)) {
      async_work_queue_->AddTask(
          vec_instruction_[next_instr_id].KernelType(),
          [this, next_instr_id]() { RunInstructionAsync(next_instr_id); });
    }
  }

  for (size_t next_instr_id : instr.NextInstrsInSameThread()) {
    if (IsReady(next_instr_id)) {
      if (vec_instruction_[next_instr_id].GetPriority() == Priority::kLowest) {
        reserved_next_ops->push_back(next_instr_id);
      } else {
        reserved_next_ops->push_front(next_instr_id);
      }
    }
  }
}

void InterpreterCore::RunInstructionAsync(size_t instr_id) {
  std::deque<size_t> ready_ops;
  ready_ops.push_back(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.front();
    ready_ops.pop_front();
    auto& instr_node = vec_instruction_.at(instr_id);

    RunInstruction(instr_node);

    if (UNLIKELY(exception_holder_.IsCaught())) {
      VLOG(4) << "Exception caught";
      if (exception_notifier_ != nullptr) {
        exception_notifier_->NotifyEvent();
      }
      return;
    }

    VLOG(4) << "unfinished_op_number_: " << unfinished_op_number_;
    if (UNLIKELY(unfinished_op_number_.fetch_sub(
                     1, std::memory_order_relaxed) == 1)) {
      if (completion_notifier_ != nullptr) {
        completion_notifier_->NotifyEvent();
      }
    }

    RunNextInstructions(instr_node, &ready_ops);
  }
}

void InterpreterCore::RecordStreamForGC(const Instruction& instr) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_THROW(platform::errors::Unimplemented(
      "RecordStreamForGC is only implemented when compiled with GPU."));
#else
  if (!IsInterpretercoreFastGCEnabled() ||
      instr.KernelType() != OpFuncType::kGpuAsync) {
    return;
  }
  platform::RecordEvent record(
      "RecordStreamForGC", platform::TracerEventType::UserDefined, 10);

  gpuStream_t stream =
      reinterpret_cast<const phi::GPUContext&>(instr.DeviceContext()).stream();
  auto TensorRecordStream = [&stream](phi::DenseTensor& tensor) {
    auto allocation = tensor.Holder();
    if (allocation == nullptr) {
      return;
    }

    const platform::Place& place = allocation->place();
    if (platform::is_gpu_place(place)) {
      memory::RecordStream(allocation, stream);
    } else if (platform::is_cuda_pinned_place(place)) {
      // TODO(Ruibiao): Here should do something to make sure that the tensor
      // is not freed until the H2D copies done. However, simplely launch a
      // CUDA runtime callback to the H2D stream may lead a high performance
      // overhead. As all the cases we meet in H2D are copies from CPUPlace at
      // present, we just log a WARNING here. A better design is required.
      LOG(WARNING) << "Copy data from a CUDAPinned tensor in an asynchronous "
                      "manner may lead a data inconsistent";
    } else {
      // memory copies involve CPUPlace are always synchronous, so just do
      // nothing here
    }
  };

  /* NOTE(Ruibiao)：Cross-stream tensor synchronization is required only when
   * all the following conditions are satisfied:
   * 1. The tensor will be GC after running the instruction, i.e., in
   * instr.GCCheckVars.
   * 2. The stream which initializes this tensor is different from the stream
   * which the instruction run in.
   * 3. The tensor is the instruction's input, cause we assume that
   * instruction will initialize all output tensors with its running stream.
   * 4. In the OP function of this instruction, the tensor is an input of a
   * async CUDA kernel.
   *
   * Here we only process the first condition, because:
   * 1. Since the RecordStream function will directly return when the recored
   * stream is equal to the owning stream, recording a stream same as which
   * initialized this tensor has less time overhead. Conversely, it may take
   * more time if we try to extract those cross-stream input vars from
   * instr.GCCheckVars.
   * 2. Now the instruction has no idea of which vars involving async running
   * in OP function, and thus we can not recognize condition 4. It should be
   * supported later.
   */
  for (int var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC sync " << var_scope_.GetNameById(var_id) << " "
            << var_scope_.VarDesc(var_id);

    // persistable var will be ignore while GC
    if (var_scope_.VarDesc(var_id) &&
        var_scope_.VarDesc(var_id)->Persistable()) {
      continue;
    }

    paddle::framework::Variable* var = var_scope_.VarRef(var_id);
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<phi::DenseTensor>()) {
      TensorRecordStream(*(var->GetMutable<phi::DenseTensor>()));
    } else if (var->IsType<
                   operators::reader::
                       OrderedMultiDeviceLoDTensorBlockingQueueHolder>()) {
      // do nothing
    } else if (var->IsType<phi::SelectedRows>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SelectedRows>()->mutable_value()));
    } else if (var->IsType<LoDTensorArray>()) {
      auto* tensor_arr = var->GetMutable<LoDTensorArray>();
      for (auto& tensor : *tensor_arr) {
        TensorRecordStream(tensor);
      }
    } else if (var->IsType<std::vector<Scope*>>()) {
      // do nothing
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The variable(%s) is not supported in eager deletion.",
          framework::ToTypeName(var->Type())));
    }
  }
#endif
}

void InterpreterCore::CheckGC(const Instruction& instr) {
  platform::RecordEvent record(
      "CheckGC", platform::TracerEventType::UserDefined, 10);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  RecordStreamForGC(instr);
#endif
  auto& var_scope = var_scope_;

  for (auto var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC:" << var_scope_.GetNameById(var_id) << ", id:" << var_id
            << ", ref:" << refs_[var_id]->DynamicRef();
    bool is_ready = refs_[var_id]->CheckAndDecrease();
    // ignore all persistable var while GC
    if (var_scope.VarDesc(var_id) && var_scope.VarDesc(var_id)->Persistable()) {
      continue;
    }
    if (is_ready) {
      VLOG(6) << "Async delete variable with name : "
              << var_scope.GetNameById(var_id);
      gc_->Add(refs_[var_id]->Var(), instr);
    }
  }
}

void InterpreterCore::Prepare(const std::vector<std::string>& feed_names,
                              const std::vector<phi::DenseTensor>& feed_tensors,
                              bool prepare_feed) {
  PADDLE_ENFORCE_EQ(feed_names.size(),
                    feed_tensors.size(),
                    platform::errors::PreconditionNotMet(
                        "Required feed_names.size() == feed_tensors.size(), "
                        "but received %d != %d",
                        feed_names.size(),
                        feed_tensors.size()));
  auto FeedInput = [&] {
    VLOG(4) << "Feed inputs";
    for (size_t i = 0; i < feed_names.size(); ++i) {
      auto* feed_var = local_scope_->FindVar(feed_names[i]);
      PADDLE_ENFORCE_NOT_NULL(
          feed_var,
          platform::errors::NotFound("Variable %s should not be nullptr.",
                                     feed_names[i]));

      auto feed_tensor = feed_var->GetMutable<phi::DenseTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
      feed_tensor->set_lod(feed_tensors[i].lod());
    }
  };

  if (!is_build_) {
    paddle::framework::interpreter::BuildVariableScope(
        block_, &var_scope_, HasLocalScope());
    FeedInput();
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::BuildOpFuncList(
        place_,
        block_,
        execution_config_.skip_gc_vars,
        &op_func_nodes,
        &var_scope_,
        execution_config_,
        HasLocalScope());
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
    UpdateSyncOpNum();
    is_build_ = true;
  }
  // NOTE: Because feed_tensor will be GC after
  // paddle::framework::BuildOpFuncList, so we should
  // call FeedInput again.
  if (prepare_feed) {
    FeedInput();
  }
}

void InterpreterCore::SetFeedVarsInplaceSkip(
    const std::vector<std::string>& feed_names) {
  for (auto& feed_name : feed_names) {
    var_scope_.SetVarSikpInplace(feed_name, true);
  }
}

bool InterpreterCore::HasLocalScope() const { return local_scope_ != nullptr; }

std::shared_ptr<InterpreterCore> CreateInterpreterCore(
    const platform::Place& place,
    const ProgramDesc& prog,
    Scope* scope,
    const std::vector<std::string>& fetch_names,
    const std::set<std::string>& skip_gc_vars) {
  std::shared_ptr<InterpreterCore> core = nullptr;
  // NOTE(Aurelius84): `AddFetch` will modify BlockDesc, so we should copy
  // a new program.
  auto new_prog = std::make_shared<framework::ProgramDesc>(prog);
  auto* block = new_prog->MutableBlock(0);
  interpreter::AddFetch(fetch_names, block);

  core = std::make_shared<InterpreterCore>(place, *block, skip_gc_vars, scope);
  core->SetCopyProgram(new_prog);
  return core;
}

// Note(zhangbo):
// (1) What is "Trace"?
// The OP execute scheduling rule adopted by Interpretercore by default is a
// multi-threaded scheduling mode(see ExecuteInstructionList). By maintaining a
// high-performance thread pool, the OP's execute scheduling is distributed to
// the sub threads maintained by the thread pool, but the main thread does not
// have any tasks. In Trace mode, the executor will execute directly in the main
// thread according to the pre provided OP sequence(trace_execute_order_),
// instead of being distributed to the thread pool.
// (2) When we use "Trace"?
// In dygraph to static, This scheduling causes that the execution of the
// forward and backward OPs and the execution of the dygraph optimizer cannot be
// executed in the same thread. Executing thread switch may cause cpu cache
// miss. When a model is all KQueueAsync type OPs, all OPs will be distributed
// to the DeviceThread for execution, and the multithreading scheduling will not
// have any benefits. Therefore, in the dynamic to static, when the number of
// KQueueAsync Ops is 0, we choose Trace mode.
void InterpreterCore::TraceInstructionList(
    const std::vector<Instruction>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i));
    }
  }

  for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
    auto instr_id = trace_execute_order_[idx];
    auto& instr_node = vec_instruction_.at(instr_id);

    RunInstruction(instr_node);

    if (UNLIKELY(exception_holder_.IsCaught())) {
      VLOG(4) << "Exception caught";
      break;
    }
  }

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    PADDLE_ENFORCE_EQ(
        main_thread_blocker_.Clear(),
        0,
        platform::errors::PreconditionNotMet(
            "main_thread_blocker_.Clear() return -1, clear failed"));
    VLOG(4) << "clear ok";
    exception_holder_.ReThrow();
  }
}

void InterpreterCore::RecordMemcpyD2H(const Instruction& instr_node) {
  // NOTE(zhiqiu): hot fix for jit input var
  if (instr_node.OpBase()->Type() == interpreter::kMemcpyD2H) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* default_dev_ctx = pool.Get(place_);
    for (auto& event : instr_node.EventsToWait()) {
      platform::RecordEvent record(
          "RecordStreamEvent", platform::TracerEventType::UserDefined, 10);
      VLOG(3) << "Record event on default stream in jit_input_var at op: "
              << instr_node.OpBase()->Type();
      event.event_->Record(default_dev_ctx);
    }
  }
}

void InterpreterCore::UpdateSyncOpNum() {
  int64_t sync_op_num = 0;
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    if (vec_instruction_[i].KernelType() == OpFuncType::kCpuSync ||
        vec_instruction_[i].KernelType() == OpFuncType::kGpuSync) {
      sync_op_num = sync_op_num + 1;
    }
  }
  sync_op_num_ = sync_op_num;
  VLOG(4) << "Update sync op num, sync op num is: " << sync_op_num_;
}

// Note(zhangbo):
// When there is a KQueueSync type OP in the model, breadth traversal is better
// than depth traversal. For example: OP(O) ->(direct_run)-> OP(A)
// ->(sync_run)-> OP(B) OP(O) ->(direct_run)-> OP(C) ->(direct_run)-> OP(D) If B
// is run before C, B may always block to wait for A to finish executing, but in
// fact, C can be executed first during this time.
void InterpreterCore::AnalyseExecuteOrderForTrace() {
  VLOG(4) << "Analyze the execution order of Trace scheduling mode.";
  interpreter::ResetAtomicGuard guard(&deps_, &refs_);

  auto op_downstream_map = dependency_builder_.OpDownstreamMap();

  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  std::vector<size_t> trace_order;
  std::deque<size_t> ready_ops;

  for (size_t instr_id = 0; instr_id < dependecy_count_.size(); ++instr_id) {
    if (dependecy_count_[instr_id] == 0) {
      ready_ops.push_back(instr_id);
    }
  }

  while (!ready_ops.empty()) {
    auto now_id = ready_ops.front();
    ready_ops.pop_front();
    trace_order.push_back(now_id);

    auto next_op_set = op_downstream_map[now_id];

    for (size_t next_op_id : next_op_set) {
      if (IsReady(next_op_id)) {
        ready_ops.push_back(next_op_id);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      trace_order.size(),
      dependecy_count_.size(),
      platform::errors::PreconditionNotMet(
          "trace_order size should be equal to dependecy_count_."));

  trace_execute_order_ = trace_order;
}

}  // namespace framework
}  // namespace paddle
