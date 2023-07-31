// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/new_ir_interpreter.h"

#include <unordered_set>

#include "gflags/gflags.h"

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_functor.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"
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
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/phi/backends/device_manager.h"

#include "paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/ir/core/builtin_attribute.h"

namespace paddle {
namespace framework {

NewIRInterpreter::NewIRInterpreter(const platform::Place& place,
                                   std::unique_ptr<::ir::Program> ir_prog,
                                   framework::Scope* scope,
                                   const ExecutionConfig& execution_config)
    : place_(place),
      stream_analyzer_(place),
      execution_config_(execution_config),
      var_scope_(scope),
      scope_(scope),
      ir_program_(std::move(ir_prog)),
      ir_stream_analyzer_(place) {
  VLOG(4) << "NewIRInterpreter(): " << this << " on " << place_;
  static_build_ = FLAGS_new_executor_static_build &&
                  !FLAGS_new_executor_use_cuda_graph &&
                  !execution_config.used_for_control_flow_op;
  //    &&interpreter::BlockCanBeStaticBuilt(block);
  static_build_ = true;

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  if (!FLAGS_new_executor_use_local_scope) {
    execution_config_.create_local_scope = false;
  }
  if (execution_config_.create_local_scope) {
    auto local_scope = &scope_->NewScope();
    local_scope_ = local_scope;
    VLOG(6) << "new ir interpretercore scope: " << scope_ << "\t"
            << "; local scope: " << local_scope_;
  }
  // TODO(zhangbo): delete var_scope
  var_scope_.SetLocalScope(local_scope_);

  execution_config_.AnalyzeThreadPoolConfig(place,
                                            ir_program_->block()->size());
  execution_config_.Log(/*log_level=*/8);

  instruction_scheduling_priority_less = [this](size_t lhs, size_t rhs) {
    SchedulingPriority lhs_scheduling_priority =
        vec_instruction_[lhs].GetSchedulingPriority();
    SchedulingPriority rhs_scheduling_priority =
        vec_instruction_[rhs].GetSchedulingPriority();
    if (lhs_scheduling_priority == rhs_scheduling_priority) {
      return lhs < rhs;
    }
    return lhs_scheduling_priority > rhs_scheduling_priority;
  };

  ir_instruction_scheduling_priority_less = [this](size_t lhs, size_t rhs) {
    SchedulingPriority lhs_scheduling_priority =
        vec_instruction_base_[lhs]->GetSchedulingPriority();
    SchedulingPriority rhs_scheduling_priority =
        vec_instruction_base_[rhs]->GetSchedulingPriority();
    if (lhs_scheduling_priority == rhs_scheduling_priority) {
      return lhs < rhs;
    }
    return lhs_scheduling_priority > rhs_scheduling_priority;
  };

  PrepareForCUDAGraphCapture();
}

NewIRInterpreter::~NewIRInterpreter() {
  // cancle gc's thread
  gc_.reset(nullptr);
  async_work_queue_.reset();
  VLOG(4) << "~NewIRInterpreter(): " << this << " on " << place_;
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

void NewIRInterpreter::RunImpl() {
  // lazy initialization of gc, do not create gc is the program only run once
  if (!gc_) {
    gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_);
  }

  interpreter::ResetAtomicGuard guard(&deps_, &refs_);

  //   if ((execution_config_.used_for_jit || execution_config_.used_for_cinn)
  //   &&
  //       (sync_op_num_ == 0)) {
  VLOG(4) << "Tracing Instruction List";

  TraceInstructionList(vec_instruction_);

  //   } else {
  //     VLOG(4) << "Non-tracing";
  //     // For the program that only run once, it is no need to
  //     // create work_queue, so the async_work_queue_ is created
  //     // until the second step run.
  //     async_work_queue_ = GetWorkQueue();
  //     ExecuteInstructionList(vec_instruction_);
  //   }
  // #ifdef PADDLE_WITH_CUSTOM_DEVICE
  //   if (platform::is_custom_place(place_)) {
  //     platform::DeviceContextPool::Instance().Get(place_)->Wait();
  //   }
  // #endif
}

FetchList NewIRInterpreter::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors) {
  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  bool is_build = is_build_;
  Prepare(feed_names, feed_tensors, is_build);

  if (is_build) {
    RunImpl();
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  auto* fetch_var = local_scope_->FindVar(interpreter::kFetchVarName);
  if (fetch_var) {
    auto fetch_list = std::move(*fetch_var->GetMutable<framework::FetchList>());
#ifdef PADDLE_WITH_CUDA
    if (platform::IsCUDAGraphCapturing()) {
      PADDLE_ENFORCE_EQ(fetch_list.empty(),
                        true,
                        platform::errors::InvalidArgument(
                            "Cannot fetch data when using CUDA Graph."));
    }
#endif
    return fetch_list;
  } else {
    return {};
  }
}

FetchList NewIRInterpreter::Run(const std::vector<std::string>& feed_names,
                                bool need_fetch) {
  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  if (!is_build_) {
    LOG_FIRST_N(INFO, 1) << "New Executor is Running.";
    std::stringstream ss;
    ss << this;
    ::ir::BuildScope(*ir_program_->block(),
                     InnerScope(),
                     ss.str(),
                     &value_2_var_name_,
                     &variable_2_var_name_,
                     &var_name_2_id_,
                     &variable_list_);
    VLOG(4) << DebugValueInfo();

    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    interpreter::BuildOpFuncList(place_,
                                 ir_program_->block(),
                                 &op_func_nodes,
                                 scope_,
                                 local_scope_,
                                 value_2_var_name_,
                                 execution_config_);
    // SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
    UpdateSyncOpNum();
    if (static_build_) {
      VLOG(4) << "RUN impl";
      RunImpl();
    }
    is_build_ = true;
  } else {
    RunImpl();
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  Scope* inner_scope = InnerScope();
  auto* fetch_var = inner_scope->FindVar(interpreter::kFetchVarName);
  if (fetch_var && need_fetch) {
    auto fetch_list = std::move(*fetch_var->GetMutable<framework::FetchList>());
#ifdef PADDLE_WITH_CUDA
    if (platform::IsCUDAGraphCapturing()) {
      PADDLE_ENFORCE_EQ(fetch_list.empty(),
                        true,
                        platform::errors::InvalidArgument(
                            "Cannot fetch data when using CUDA Graph."));
    }
#endif
    return fetch_list;
  } else {
    return {};
  }
}

int NewIRInterpreter::GetIdByName(const std::string& name) const {
  auto it = var_name_2_id_.find(name);
  if (it != var_name_2_id_.end()) {
    return it->second;
  }
  return -1;
}

void NewIRInterpreter::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  copy_program_ = prog;
}

void NewIRInterpreter::SetSkipGcVars(
    const std::set<std::string>& skip_gc_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.skip_gc_vars.empty(),
      true,
      platform::errors::PreconditionNotMet(
          "execution_config_.skip_gc_vars can only be initialized once, now "
          "execution_config_.skip_gc_vars is "
          "not empty, do not call SetSkipGcVars method repeatedly."));
  execution_config_.skip_gc_vars = skip_gc_vars;
}

void NewIRInterpreter::SetJitInputVars(
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

const std::set<std::string>& NewIRInterpreter::JitInputVars() const {
  return execution_config_.jit_input_vars;
}

const VariableScope* NewIRInterpreter::GetVariableScope() const {
  return &var_scope_;
}

void NewIRInterpreter::reset_scope(Scope* new_scope) {
  var_scope_.SetScope(new_scope);
  auto& var_list = var_scope_.MutableVarList();
  for (size_t i = 0; i < var_list.size(); i++) {
    const auto& var_name = var_scope_.GetNameById(i);
    var_list[i] = new_scope->FindVar(var_name);
  }
  // The index should be assured valid, cause the InterpreterCore may not be
  // fully built, but was still cached and used. For example, see unit test
  // `test_assert.py`, it may exit before `NewIRInterpreter::Convert`,
  // but still was cached and used by later tests.
  for (size_t i = 0; i < std::min(refs_.size(), var_list.size()); i++) {
    refs_[i]->ResetVariable(var_list[i]);
  }

  for (size_t i = 0; i < vec_instruction_.size(); i++) {
    BuildAndCacheInstructionCtx(&vec_instruction_[i]);
  }
}

const Scope* NewIRInterpreter::local_scope() const { return local_scope_; }

std::string NewIRInterpreter::GetNameById(int id) const {
  // NOTE(zhiqiu): do not use vec_meta_info_[id].vardesc_->Name() since
  // vec_meta_info_[id] may be nullptr,
  // typically when the target variable is not existed in the original program
  // desc, but created by interpretercore.
  // For example, created and used by d2h_copy or h2d_copy operator.
  auto it = std::find_if(var_name_2_id_.begin(),
                         var_name_2_id_.end(),
                         [id](const auto& pair) { return pair.second == id; });
  if (it != var_name_2_id_.end()) {
    return it->first;
  }
  return "";
}

void NewIRInterpreter::ShareWorkQueueFrom(InterpreterBaseImpl* src) {
  async_work_queue_ = reinterpret_cast<NewIRInterpreter*>(src)->GetWorkQueue();
  VLOG(8) << "Share AsyncWorkQueue from InterpreterCore(" << src
          << ") to InterpreterCore(" << this << ")";
}

void NewIRInterpreter::ShareBuildResultsFrom(const InterpreterBaseImpl& src) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "ShareBuildResultsFrom is not implemented in NewIRInterpreter."));
}

// op dependences
const interpreter::DependencyBuilder& NewIRInterpreter::GetDependencyBuilder()
    const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "GetDependencyBuilder is not implemented in NewIRInterpreter."));
}

std::shared_ptr<std::vector<size_t>> NewIRInterpreter::GetDependencyCount()
    const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "GetDependencyCount is not implemented in NewIRInterpreter."));
}

const interpreter::StreamAnalyzer& NewIRInterpreter::GetStreamAnalyzer() const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "GetStreamAnalyzer is not implemented in NewIRInterpreter."));
}

bool NewIRInterpreter::IsSharedResultsBuild() const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "IsSharedResultsBuild is not implemented in NewIRInterpreter."));
}

bool NewIRInterpreter::BuildInplaceCheckVarIsOnlyInput(
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

std::shared_ptr<interpreter::AsyncWorkQueue> NewIRInterpreter::GetWorkQueue() {
  if (async_work_queue_ == nullptr) {
    async_work_queue_ = std::make_shared<interpreter::AsyncWorkQueue>(
        execution_config_.host_num_threads,
        execution_config_.device_num_threads,
        nullptr);
  }
  return async_work_queue_;
}

void NewIRInterpreter::BuildAndCacheInstructionCtx(Instruction* instr_node) {
  Scope* inner_scope = InnerScope();
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
  if (instr_node->OpBase()->Type() == "cinn_launch" ||
      instr_node->OpBase()->Type() == "cinn_instruction_run") {  // OP use scope
                                                                 // in kernel
    Scope* inner_scope = InnerScope();
    instr_node->ResetContextWithScope(ins_map, outs_map, *inner_scope);
  } else {
    instr_node->ResetContext(ins_map, outs_map);
  }
}

void NewIRInterpreter::BuildInplace() {
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

  Scope* local_scope = InnerScope();
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

void NewIRInterpreter::PrepareForCUDAGraphCapture() {
  if (!FLAGS_new_executor_use_cuda_graph) return;
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_EQ(
      platform::IsCUDAGraphCapturing(),
      false,
      platform::errors::PermissionDenied("CUDA Graph is not allowed to capture "
                                         "before prepare."));
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(place_),
                    true,
                    platform::errors::InvalidArgument(
                        "CUDA Graph is only supported on NVIDIA GPU device."));
  // If set true, will call `cudaStreamSynchronize(nccl_stream)`after allreduce.
  // which may cause error in cuda graph. This behavior is consistent with PE.
  PADDLE_ENFORCE_EQ(FLAGS_sync_nccl_allreduce,
                    false,
                    platform::errors::InvalidArgument(
                        "FLAGS_sync_nccl_allreduce must be False to support "
                        "CUDA Graph capturing."));

  // All output vars of coalesce_tensor op should be persistable.
  // If fused output var of coalesce_tensor is gc, it will cause accuracy
  // problem. The specific reasons need to be analyzed.
//   for (auto& op_desc : block_.AllOps()) {
//     if (op_desc->Type() == kCoalesceTensor) {
//       for (auto& out_var_name : op_desc->OutputArgumentNames()) {
//         // The fused var needs to be set to persistable, not just added to
//         // skip_gc_vars.
//         // In the case where the feed fetch var is changed,
//         StandaloneExecutor
//         // will be newly constructed. If the fused var is not persistable,
//         // these vars will be recreated and initialized, resulting in
//         // precision problems.
//         auto* out_var = op_desc->Block()->FindVarRecursive(out_var_name);
//         if (out_var) {
//           out_var->SetPersistable(true);
//           VLOG(4) << "Mark Var(" << out_var_name << ") as Persistable.";
//         }
//       }
//     }
//   }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

void NewIRInterpreter::CheckCUDAGraphBeforeRun(
    const std::vector<std::string>& feed_names) {
#ifdef PADDLE_WITH_CUDA
  if (platform::IsCUDAGraphCapturing()) {
    PADDLE_ENFORCE_EQ(
        feed_names.empty(),
        true,
        platform::errors::InvalidArgument(
            "Feeding data is not permitted when capturing CUDA Graph."));
    PADDLE_ENFORCE_EQ(
        FLAGS_new_executor_use_cuda_graph,
        true,
        platform::errors::InvalidArgument(
            "You must turn on FLAGS_new_executor_use_cuda_graph to True "
            "to enable CUDA Graph capturing."));
    PADDLE_ENFORCE_EQ(
        place_,
        platform::CUDAGraphCapturingPlace(),
        platform::errors::InvalidArgument("The place to capture CUDAGraph is "
                                          "not the same as the place to run."));
  }
#endif
}

void NewIRInterpreter::BuildOperatorDependences() {
  // analysis the dependences between ops, add next_instr_list to each instr,
  // and set the dependecy_count_
  size_t instr_num = vec_instruction_.size();
  dependecy_count_ = std::vector<size_t>(instr_num, 0);
  auto downstream_map = dependency_builder_.Build(vec_instruction_);

  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    Instruction& cur_instr = vec_instruction_[instr_id];
    const std::set<size_t>& next_instr_ids = downstream_map[instr_id];

    if (FLAGS_new_executor_serial_run) {
      for (size_t next_instr_id : next_instr_ids) {
        cur_instr.AddNextInstrInSameThread(next_instr_id);
      }
    } else {
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
    }

    for (size_t next_instr_id : next_instr_ids) {
      ++dependecy_count_[next_instr_id];
    }
  }
}

// At the end of each step, the holder of phi::DenseTensor in LoDTensorArray is
// null. Clear these Tensors and leave LoDTensorArray empty, otherwise an
// exception will occur in the next step
void NewIRInterpreter::ClearLoDTensorArrayInLocalScope() {
  auto vars = local_scope_->LocalVars();
  for (auto var : vars) {
    if (var->IsType<LoDTensorArray>()) {
      auto* lod_tensor_arr = var->GetMutable<LoDTensorArray>();
      lod_tensor_arr->clear();
    }
  }
}

void NewIRInterpreter::Convert(
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  auto& vec_meta_info = var_scope_.MutableVecMetaInfo();
  auto nodes = *op_func_nodes;
  auto op_nums = nodes.size();
  vec_instruction_.clear();
  vec_instruction_.reserve(op_nums);
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    auto& op_func_node = nodes[op_idx];
    auto* dev_ctx_ = stream_analyzer_.ParseDeviceContext(op_func_node);
    vec_instruction_.emplace_back(op_idx, std::move(op_func_node), *dev_ctx_);
#ifdef PADDLE_WITH_CUDA
    if (FLAGS_new_executor_use_cuda_graph) {
      auto& op = op_func_node.operator_base_;
      auto& op_type = op->Type();
      if (op_type == interpreter::kMemcpyD2H ||
          op_type == interpreter::kMemcpyH2D) {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Cuda memory copy d2h/h2d is not allowed while using cuda graph."));
      }
      PADDLE_ENFORCE_EQ(typeid(*dev_ctx_) == typeid(phi::GPUContext),
                        true,
                        platform::errors::InvalidArgument(
                            "Device context of op %s must be [%s] while using "
                            "cuda graph, but got [%s].",
                            op_type,
                            typeid(phi::GPUContext).name(),
                            typeid(*dev_ctx_).name()));
      // cuda graph needs to record all stream
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .RecordCapturingDeviceContext(dev_ctx_);
    }
#endif
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
      if (inst.OpBaseValid() &&
          inst.OpBase()->Type() == interpreter::kMemcpyD2H &&
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
  //   for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
  //     Instruction& instr = vec_instruction_[op_idx];
  //     OpInOutInfo info;
  //     info.Build(instr.OpBase());

  //     std::set<size_t> gc_check_vars;

  //     const std::map<std::string, std::vector<int>>& ins = instr.Inputs();
  //     const std::map<std::string, std::vector<int>>& outs = instr.Outputs();
  //     std::multimap<std::string, std::vector<int>> ins_and_outs{ins.begin(),
  //                                                               ins.end()};
  //     ins_and_outs.insert(outs.begin(), outs.end());

  //     for (auto& item : ins_and_outs) {
  //       for (auto id : item.second) {
  //         if (id == kEmptyVarIndex) {
  //           continue;
  //         }
  //         auto* var_desc = var_scope_.VarDesc(id);
  //         // skip no_need_buffer input vars
  //         if (var_desc && ins.count(item.first) &&
  //             !info.IsInArgBufferNeeded(var_desc->Name())) {
  //           continue;
  //         }
  //         // skip when this var is not in block and not a data_transferred
  //         var,
  //         // which means this var is managed by other block
  //         const auto& var_name = var_scope_.GetNameById(id);
  //         bool not_owned = !block_.HasVar(var_name);
  //         const auto& transferred_vars = var_scope_.DataTransferAddedVars();
  //         bool not_transferred =
  //             std::all_of(transferred_vars.begin(),
  //                         transferred_vars.end(),
  //                         [&](const std::pair<std::string, int>& elem) {
  //                           return elem.first != var_name;
  //                         });
  //         if (not_owned && not_transferred) {
  //           VLOG(10) << "[gc_check_inputs] skip gc: " << var_name;
  //           continue;
  //         }
  //         gc_check_vars.insert(id);
  //       }
  //     }

  //     for (auto var_id : gc_check_vars) {
  //       Scope* inner_scope =
  //           HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
  //       paddle::framework::Variable* var =
  //           inner_scope->FindVar(var_scope_.GetNameById(var_id));
  //       if (var->IsType<phi::DenseTensor>() ||
  //       var->IsType<phi::SelectedRows>() ||
  //           var->IsType<LoDTensorArray>()) {
  //         last_live_ops_[var_id].insert(op_idx);
  //       } else {
  //         VLOG(4) << "not clear " << var_scope_.GetNameById(var_id) << "
  //         after "
  //                 << instr.OpBase()->Type() << " because its type is "
  //                 << framework::ToTypeName(var->Type());
  //       }
  //     }
  //   }

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

  //   for (size_t i = 0; i < vec_instruction_.size(); ++i) {
  //     BuildAndCacheInstructionCtx(&vec_instruction_[i]);
  //   }

  // bool inplaced = false;
  // for (const Instruction& inst : vec_instruction_) {
  //   if (inst.OpBase()->Type() == "share_buffer" ||
  //       inst.OpBase()->Type() == "share_data") {
  //     VLOG(4) << "Already inplaced, skip inplace now.";
  //     inplaced = true;
  //   }
  // }

  //   if (FLAGS_new_executor_use_inplace && !inplaced) {
  //     BuildInplace();
  //   }

  for (auto& dep : dependecy_count_) {
    deps_.emplace_back(std::make_shared<interpreter::OpDepInfo>(dep));
  }
  for (size_t i = 0; i < vec_meta_info.size(); ++i) {
    refs_.emplace_back(std::make_shared<interpreter::VarRefInfo>(
        vec_meta_info[i].var_ref_count_, var_scope_.VarRef(i)));
  }

  AnalyseExecuteOrderForTrace(dependency_builder_.OpDownstreamMap(),
                              instruction_scheduling_priority_less);
}

void NewIRInterpreter::BuildSkipShareLoDInfo() {
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    bool can_skip_lod = true;
    for (auto& input : vec_instruction_[i].InnerRuntimeContext()->inputs) {
      for (auto& var : input.second) {
        if (var->IsType<phi::DenseTensor>()) {
          if (!var->Get<phi::DenseTensor>().lod().empty()) {
            can_skip_lod = false;
            break;
          }
        } else {
          can_skip_lod = false;
          break;
        }
      }
    }
    if (can_skip_lod) {
      VLOG(8) << "skip share lod for: " << vec_instruction_[i].OpBase()->Type()
              << " (" << i << ")";
    }
    vec_instruction_[i].InnerInferShapeContext()->SetSkipLoD(can_skip_lod);
  }
}

void NewIRInterpreter::RunOperator(const Instruction& instr_node) {
  auto* op = instr_node.OpBase();
  auto place = instr_node.DeviceContext().GetPlace();
  Scope* local_scope = InnerScope();
  VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

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
    if (op_with_kernel == nullptr) {  // operator base
      instr_node.OpBase()->Run(*local_scope, place_);
    } else {
      phi::Kernel* kernel = instr_node.PhiKernel();
      if (kernel && kernel->IsValid()) {  // phi kernel
        if (kernel->GetKernelRegisteredType() ==
            phi::KernelRegisteredType::FUNCTION) {
          VLOG(4) << "Run function kernel: " << op->Type();
          VLOG(4) << instr_node.InnerRuntimeContext().get() << " "
                  << &instr_node.DeviceContext();
          phi::KernelContext phi_kernel_context;
          op_with_kernel->BuildPhiKernelContext(
              *instr_node.InnerRuntimeContext().get(),
              const_cast<platform::DeviceContext*>(&instr_node.DeviceContext()),
              &phi_kernel_context);

          (*kernel)(&phi_kernel_context);
        } else {
          VLOG(4) << "Run structure kernel: " << op->Type();
          (*kernel)(instr_node.InnerExecutionContext().get());
        }
      } else {  // fluid kernel
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

  for (auto& hook : hookfuncs_) {
    hook(op, local_scope);
  }

  // for debug nan/inf
  if (op_with_kernel != nullptr && FLAGS_check_nan_inf) {
    VLOG(4) << "Check nan/inf";
    try {
      framework::details::CheckOpHasNanOrInf(
          *op,
          *local_scope,
          place);  // TODO(xiongkun03) change it to inner scope.
    } catch (...) {
      const std::vector<std::string>* callstack = nullptr;
      auto attrs = op->Attrs();
      auto iter =
          attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
      if (iter != attrs.end()) {
        callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
        if (callstack->empty()) callstack = nullptr;
      }
      std::ostringstream sout;
      if (callstack) {
        if (FLAGS_call_stack_level > 1) {
          sout << "\n\n  Compile Traceback (most recent call last):";
        } else {
          sout << "In user code:\n";
        }
        for (auto& line : *callstack) {
          sout << "\n  " << line;
        }
      }
      std::cout << sout.str() << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }
}

void NewIRInterpreter::RunInstruction(const Instruction& instr_node) {
  OperatorBase* op = nullptr;
  if (instr_node.OpBaseValid()) {
    op = instr_node.OpBase();
    platform::RecordEvent instruction_event(
        op->Type(), platform::TracerEventType::Operator, 1);
  }

  SetDeviceId(instr_node.DeviceContext().GetPlace());

  try {
    instr_node.WaitEvent(place_);

    if (instr_node.PreDefineContext()) {
      VLOG(5) << "run new ir selected kernel";
      auto op_func_node = const_cast<OpFuncNode*>((instr_node.OpFunc()));
      VLOG(5) << "begin to run op " << op_func_node->phi_op_name_;
      if (op_func_node->infer_meta_interface_) {
        op_func_node->infer_meta_interface_->infer_meta_(
            &(op_func_node->infer_meta_context_));
      }
      VLOG(5) << "after run infer meta";
      if (op_func_node->fluid_op) {
        // run fluid op
        ExecutionContext exe_ctx(*(op_func_node->operator_base_.get()),
                                 *scope_,
                                 *(op_func_node->dev_ctx_),
                                 *(op_func_node->runtime_ctx_.get()));
        (*(op_func_node->phi_kernel_))(&exe_ctx);

      } else {
        (*(op_func_node->phi_kernel_))(&(op_func_node->kernel_context_));
      }
      VLOG(5) << "after run kernel";
    } else if (!instr_node.IsArtificial()) {
      RunOperator(instr_node);
      CheckGC(instr_node);
      interpreter::LogDeviceMemoryStats(place_);
    }

    instr_node.RecordEvent(place_);
  } catch (platform::EnforceNotMet& ex) {
    LOG(WARNING) << instr_node.OpFunc()->phi_op_name_
                 << " raises an EnforceNotMet exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::make_exception_ptr(std::move(ex)));
  } catch (platform::EOFException&) {
    exception_holder_.Catch(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << instr_node.OpFunc()->phi_op_name_ << " raises an exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::current_exception());
  } catch (...) {
    LOG(WARNING) << instr_node.OpFunc()->phi_op_name_
                 << " raises an unknown exception";
    exception_holder_.Catch(std::current_exception());
  }
}

std::string NewIRInterpreter::GetDepsString() const {
  std::stringstream ss;
  auto downstream_map = dependency_builder_.OpDownstreamMap();
  ss << "Note: when static_dep is 1, it is ok that the dynamic_dep will not "
        "be decreased to 0."
     << std::endl;
  ss << "unfinished_op_number_:" << unfinished_op_number_ << std::endl;
  for (size_t i = 0; i < deps_.size(); ++i) {
    ss << "op:" << i << ", type: " << vec_instruction_[i].OpBase()->Type()
       << ", static_dep:" << deps_[i]->StaticDep()
       << ", dynamic_dep:" << deps_[i]->DynamicDep() << ", downstream op: ";
    for (auto id : downstream_map[i]) {
      ss << id << ", ";
    }
    ss << std::endl;
  }
  return ss.str();
}

void NewIRInterpreter::ExecuteInstructionList(
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
      if (FLAGS_new_executor_serial_run) {
        RunInstructionAsync(i);
      } else {
        async_work_queue_->AddTask(vec_instr.at(i).KernelType(),
                                   [this, i] { RunInstructionAsync(i); });
      }
    }
  }

  // For debug hang in main_thread_blocker_.WaitEvent(),
  // launch async task to log deps every
  // FLAGS_executor_log_deps_every_microseconds, then cancel the std::async when
  // main_thread_blocker_.WaitEvent() executed. Why not use std::async instead
  // of workqueue? To make sure that the logging thread itself will not affect
  // the workqueue
  //  used in interpretercore.

  std::future<int> logged_times;
  std::atomic_bool cancel_log = ATOMIC_VAR_INIT(false);
  if (FLAGS_executor_log_deps_every_microseconds) {
    logged_times = std::async(
        std::launch::async,
        [this](const std::atomic_bool& cancel) {
          int times = 0;
          while (!cancel) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                FLAGS_executor_log_deps_every_microseconds));
            // check again, since cancel may be changed during sleep
            if (cancel) {
              break;
            }
            VLOG(6) << "deps:\n" << GetDepsString();
            times++;
          }
          return times;
        },
        std::ref(cancel_log));
  }

  auto event_name = main_thread_blocker_.WaitEvent();
  VLOG(1) << "main_thread_blocker_(" << &main_thread_blocker_
          << ") got event_name: " << event_name;

  cancel_log = true;
  if (logged_times.valid()) {
    VLOG(1) << "Logged deps for " << logged_times.get() << " times";
  }

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    // Graceful exit when the executor encountered a fatal error.
    // EOF is not a fatal error.
    if (exception_holder_.Type() != "EOF") {
      async_work_queue_->Cancel();
      async_work_queue_.reset();
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

void NewIRInterpreter::RunNextInstructions(const Instruction& instr,
                                           SchedulingQueue* reserved_next_ops) {
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
      reserved_next_ops->push(next_instr_id);
    }
  }
}

void NewIRInterpreter::RunInstructionAsync(size_t instr_id) {
  // NOTE(Ruibiao): Due to the uncertain order in multi-threading asynchronous
  // scheduling, the priority order involved cross-thread scheduling is not
  // guaranteed. Only Ops scheduled by the same AddTask call have the guarantee
  // of priority order.
  SchedulingQueue ready_ops(instruction_scheduling_priority_less);
  ready_ops.push(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.top();
    ready_ops.pop();
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

void NewIRInterpreter::RecordStreamForGC(const Instruction& instr) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_THROW(platform::errors::Unimplemented(
      "RecordStreamForGC is only implemented when compiled with GPU."));
#else
  if (!IsInterpretercoreFastGCEnabled() ||
      instr.KernelType() != OpFuncType::kGpuAsync) {
    return;
  }
  if (instr.DeviceContext().GetPlace().GetType() ==
      phi::AllocationType::CUSTOM) {
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

void NewIRInterpreter::CheckGC(const Instruction& instr) {
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

::ir::Value NewIRInterpreter::GetValueByName(const std::string& var_name) {
  for (auto kv : value_2_var_name_) {
    if (kv.second == var_name) {
      return kv.first;
    }
  }
  return nullptr;
}

void NewIRInterpreter::Prepare(
    const std::vector<std::string>& feed_names,
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
  // TODO(dev): Support this
  //   if (!is_build_) {
  //     paddle::framework::interpreter::BuildVariableScope(
  //         block_, execution_config_, &var_scope_);
  //     FeedInput();
  //     std::vector<paddle::framework::OpFuncNode> op_func_nodes;
  //     paddle::framework::interpreter::BuildOpFuncList(
  //         place_,
  //         block_,
  //         execution_config_.skip_gc_vars,
  //         &op_func_nodes,
  //         &var_scope_,
  //         execution_config_,
  //         HasLocalScope(),
  //         static_build_);
  //     SetFeedVarsInplaceSkip(feed_names);
  //     // convert vec func_list to graph
  //     Convert(&op_func_nodes);
  //     UpdateSyncOpNum();
  //     if (static_build_) {
  //       VLOG(4) << "RUN impl";
  //       RunImpl();
  //     }
  //     BuildSkipShareLoDInfo();
  //     is_build_ = true;
  //   }
  // NOTE: Because feed_tensor will be GC after
  // paddle::framework::BuildOpFuncList, so we should
  // call FeedInput again.
  if (prepare_feed) {
    FeedInput();
  }
}

void NewIRInterpreter::SetFeedVarsInplaceSkip(
    const std::vector<std::string>& feed_names) {
  for (auto& feed_name : feed_names) {
    var_scope_.SetVarSikpInplace(feed_name, true);
  }
}

bool NewIRInterpreter::HasLocalScope() const { return local_scope_ != nullptr; }

Scope* NewIRInterpreter::InnerScope() {
  return local_scope_ != nullptr ? local_scope_ : scope_;
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
void NewIRInterpreter::TraceInstructionList(
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

  // TODO(phlrain) use orignal order for now, use better dependecy
  for (size_t instr_id = 0; instr_id < vec_instruction_.size(); ++instr_id) {
    /// auto instr_id = trace_execute_order_[idx];
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

void NewIRInterpreter::RecordMemcpyD2H(const Instruction& instr_node) {
  // NOTE(zhiqiu): hot fix for jit input var
  if (instr_node.OpBaseValid() &&
      instr_node.OpBase()->Type() == interpreter::kMemcpyD2H) {
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

void NewIRInterpreter::UpdateSyncOpNum() {
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
void NewIRInterpreter::AnalyseExecuteOrderForTrace(
    std::map<size_t, std::set<size_t>> op_downstream_map,
    InstructionSchedulingPriorityLess compare) {
  VLOG(4) << "Analyze the execution order of Trace scheduling mode.";
  interpreter::ResetAtomicGuard guard(&deps_, &refs_);
  VLOG(4) << "1";
  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  std::vector<size_t> trace_order;
  SchedulingQueue ready_ops(compare);

  for (size_t instr_id = 0; instr_id < dependecy_count_.size(); ++instr_id) {
    if (dependecy_count_[instr_id] == 0) {
      ready_ops.push(instr_id);
    }
  }

  while (!ready_ops.empty()) {
    size_t now_id = ready_ops.top();
    ready_ops.pop();
    trace_order.push_back(now_id);

    auto next_op_set = op_downstream_map[now_id];

    for (size_t next_op_id : next_op_set) {
      if (IsReady(next_op_id)) {
        ready_ops.push(next_op_id);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      trace_order.size(),
      dependecy_count_.size(),
      platform::errors::PreconditionNotMet(
          "trace_order size should be equal to dependecy_count_."));

  trace_execute_order_ = trace_order;

  std::stringstream ss;
  ss << "trace order: ";
  for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
    ss << trace_execute_order_[idx] << " -> ";
  }
  ss << "end\n";
  VLOG(6) << ss.str();
}

/// ======================== ///
///        For new ir        ///
/// ======================== ///

void NewIRInterpreter::BuildInstruction() {
  VLOG(6) << "Build Instructions for new ir ... ";
  vec_instruction_base_.clear();
  size_t op_idx = 0;
  for (auto it = ir_program_->block()->begin();
       it != ir_program_->block()->end();
       ++it) {
    VLOG(6) << "Build Instruction for op: " << op_idx;
    if ((*it)->dialect()->name() == "pd_kernel") {
      auto op_name = (*it)
                         ->attributes()
                         .at("op_name")
                         .dyn_cast<::ir::StrAttribute>()
                         .AsString();
      if (op_name == "builtin.combine" || op_name == "builtin.slice" ||
          op_name == "pd.feed" || op_name == "pd.fetch" ||
          op_name == "builtin.set_parameter" ||
          op_name == "builtin.get_parameter") {
        VLOG(6) << "skip process " << op_name;
        continue;
      }
      vec_instruction_base_.emplace_back(
          std::make_unique<PhiKernelInstruction>(op_idx++,
                                                 place_,
                                                 (*it),
                                                 scope_,
                                                 local_scope_,
                                                 value_2_var_name_,
                                                 var_name_2_id_,
                                                 variable_2_var_name_));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Now only support pd_kernel dialect."));
    }
  }
}

std::string NewIRInterpreter::DebugValueInfo() {
  std::stringstream os;
  os << "value info of interpretercore " << this << "\n"
     << "value -> var_name -> id -> variable*"
     << "\n";
  for (auto kv : value_2_var_name_) {
    PADDLE_ENFORCE((bool)kv.first,
                   platform::errors::PreconditionNotMet(
                       "vlaue(%s) should not be nullptr", kv.second));
    PADDLE_ENFORCE(var_name_2_id_.count(kv.second) > 0,
                   platform::errors::PreconditionNotMet(
                       "var(%s) should exist in var_name_2_id_", kv.second));
    auto* var = InnerScope()->FindVar(kv.second);
    PADDLE_ENFORCE(var != nullptr,
                   platform::errors::PreconditionNotMet(
                       "var(%s) should exist in var_name_2_id_", kv.second));
    os << kv.first.impl() << " -> " << kv.second << " -> "
       << var_name_2_id_.at(kv.second) << " -> " << var << "\n";
  }
  return os.str();
}

void NewIRInterpreter::BuildInstructionDependences() {
  // analysis the dependences between instructions, add next_instr_list to each
  // instr, and set the dependecy_count_
  size_t instr_num = vec_instruction_base_.size();
  dependecy_count_ = std::vector<size_t>(instr_num, 0);

  std::vector<paddle::framework::InstructionBase*> instructions_ptr;
  for (auto& instr : vec_instruction_base_) {
    instructions_ptr.push_back(instr.get());
  }
  auto downstream_map = ir_dependency_builder_.Build(instructions_ptr);

  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    InstructionBase* cur_instr = vec_instruction_base_[instr_id].get();
    const std::set<size_t>& next_instr_ids = downstream_map[instr_id];

    if (FLAGS_new_executor_serial_run) {
      for (size_t next_instr_id : next_instr_ids) {
        cur_instr->AddNextInstrInSameThread(next_instr_id);
      }
    } else {
      if (cur_instr->KernelType() == OpFuncType::kGpuAsync) {
        for (size_t next_instr_id : next_instr_ids) {
          if (vec_instruction_base_[next_instr_id]->KernelType() ==
              OpFuncType::kGpuAsync) {
            cur_instr->AddNextInstrInSameThread(next_instr_id);
          } else {
            cur_instr->AddNextInstrInDifferentThread(next_instr_id);
          }
        }
      } else {
        bool has_instr_in_same_thread = false;
        for (size_t next_instr_id : next_instr_ids) {
          if (!has_instr_in_same_thread &&
              vec_instruction_base_[next_instr_id]->KernelType() !=
                  OpFuncType::kGpuAsync) {
            cur_instr->AddNextInstrInSameThread(next_instr_id);
            has_instr_in_same_thread = true;
          } else {
            cur_instr->AddNextInstrInDifferentThread(next_instr_id);
          }
        }
      }
    }

    for (size_t next_instr_id : next_instr_ids) {
      ++dependecy_count_[next_instr_id];
    }
  }
}

void NewIRInterpreter::RecordMemcpyD2H(InstructionBase* instr_node) {
  // NOTE(zhiqiu): hot fix for jit input var
  if (instr_node->Name() == "pd.memcpy_d2h") {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* default_dev_ctx = pool.Get(place_);
    for (auto& event : instr_node->EventsToWait()) {
      platform::RecordEvent record(
          "RecordStreamEvent", platform::TracerEventType::UserDefined, 10);
      VLOG(3) << "Record event on default stream in jit_input_var at op: "
              << instr_node->Name();
      event.event_->Record(default_dev_ctx);
    }
  }
}

void NewIRInterpreter::RecordStreamForGC(InstructionBase* instr) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_THROW(platform::errors::Unimplemented(
      "RecordStreamForGC is only implemented when compiled with GPU."));
#else
  if (!IsInterpretercoreFastGCEnabled() ||
      instr->KernelType() != OpFuncType::kGpuAsync) {
    return;
  }
  if (instr->DeviceContext().GetPlace().GetType() ==
      phi::AllocationType::CUSTOM) {
    return;
  }
  platform::RecordEvent record(
      "RecordStreamForGC", platform::TracerEventType::UserDefined, 10);

  gpuStream_t stream =
      reinterpret_cast<const phi::GPUContext&>(instr->DeviceContext()).stream();
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
  for (int var_id : instr->GCCheckVars()) {
    VLOG(4) << "GC sync " << GetNameById(var_id);

    // persistable var will be ignore while GC
    ::ir::Value value = GetValueByName(GetNameById(var_id));
    if (value && value.GetDefiningOp()->attributes().count("is_persisable") &&
        value.GetDefiningOp()
            ->attributes()
            .at("is_persisable")
            .dyn_cast<::ir::BoolAttribute>()
            .data()) {
      continue;
    }

    paddle::framework::Variable* var = variable_list_[var_id];
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

void NewIRInterpreter::CheckGC(InstructionBase* instr) {
  platform::RecordEvent record(
      "CheckGC", platform::TracerEventType::UserDefined, 10);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  RecordStreamForGC(instr);
#endif

  for (auto var_id : instr->GCCheckVars()) {
    VLOG(4) << "GC:" << GetNameById(var_id) << ", id:" << var_id
            << ", ref:" << refs_[var_id]->DynamicRef();
    bool is_ready = refs_[var_id]->CheckAndDecrease();
    // ignore all persistable var while GCphi
    ::ir::Value value = GetValueByName(GetNameById(var_id));
    if (value && value.GetDefiningOp()->attributes().count("is_persisable") &&
        value.GetDefiningOp()
            ->attributes()
            .at("is_persisable")
            .dyn_cast<::ir::BoolAttribute>()
            .data()) {
      continue;
    }
    if (is_ready) {
      VLOG(6) << "Async delete variable with name : " << GetNameById(var_id);
      gc_->Add(refs_[var_id]->Var(), instr);
    }
  }
}

void NewIRInterpreter::CalculateLastLiveOps() {
  // calculate last_live_ops_
  for (size_t op_idx = 0; op_idx < vec_instruction_base_.size(); ++op_idx) {
    InstructionBase* instr = vec_instruction_base_[op_idx].get();
    std::set<size_t> gc_check_vars;

    const std::unordered_map<::ir::Value, std::vector<int>>& ins =
        instr->Inputs();
    const std::unordered_map<::ir::Value, std::vector<int>>& outs =
        instr->Outputs();
    std::unordered_multimap<::ir::Value, std::vector<int>> ins_and_outs{
        ins.begin(), ins.end()};
    ins_and_outs.insert(outs.begin(), outs.end());

    for (auto& item : ins_and_outs) {
      for (auto var_id : item.second) {
        // skip no_need_buffer input vars
        if (ins.count(item.first) && instr->NoNeedBuffer().count(item.first)) {
          continue;
        }
        gc_check_vars.insert(var_id);
      }
    }

    for (auto var_id : gc_check_vars) {
      Scope* inner_scope = InnerScope();
      paddle::framework::Variable* var =
          inner_scope->FindVar(GetNameById(var_id));
      if (var->IsType<phi::DenseTensor>() || var->IsType<phi::SelectedRows>() ||
          var->IsType<LoDTensorArray>()) {
        last_live_ops_[var_id].insert(op_idx);
      } else {
        VLOG(4) << "not clear " << GetNameById(var_id) << " after "
                << instr->Name() << " because its type is "
                << framework::ToTypeName(var->Type());
      }
    }
  }
  // clear the last_live_ops list for all vars in skip_gc_vars
  for (const std::string& skip_gc_var : execution_config_.skip_gc_vars) {
    int var_id = GetIdByName(skip_gc_var);
    if (var_id != -1) {
      last_live_ops_[var_id].clear();
      VLOG(8) << "Skip gc for var: " << skip_gc_var;
    }
  }
  VLOG(4) << "calculate last_live_ops_";

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  // For example,
  // b = op1(a)
  // c = op2(a, b)
  // in this case, a is the input of op1 and op2, we only need to check
  // a after op2, because op2 always uses a after op1.
  var_ref_count_.resize(variable_list_.size());
  VLOG(4) << "last_live_ops_.size() : " << last_live_ops_.size();
  for (auto kv : last_live_ops_) {
    for (auto val : kv.second) {
      VLOG(4) << "var: " << kv.first << " -> op: " << val;
    }
  }
  VLOG(4) << "var_ref_count_.size() : " << var_ref_count_.size();
  for (size_t i = 0; i < last_live_ops_.size(); ++i) {
    std::set<size_t> minumum_last_live_ops;
    for (auto val : last_live_ops_[i]) {
      VLOG(4) << "last_live_ops_: " << val;
    }
    for (size_t item : last_live_ops_[i]) {
      bool not_before_any = true;
      // find the op that is not executed before any
      for (size_t other_item : last_live_ops_[i]) {
        if (ir_dependency_builder_.OpHappensBefore(item, other_item)) {
          VLOG(6) << "happens_before: " << item << "->" << other_item
                  << ", so skip " << item;
          not_before_any = false;
          break;
        }
      }
      if (not_before_any) {
        VLOG(6) << "last live op of var " << i << " " << GetNameById(i) << " : "
                << item << " " << vec_instruction_base_[item]->Name();
        minumum_last_live_ops.insert(item);
        vec_instruction_base_[item]->AddGCCheckVar(i);
      }
    }
    last_live_ops_[i] = minumum_last_live_ops;
    var_ref_count_[i] = last_live_ops_[i].size();
  }
  VLOG(4) << "calculate last_live_ops_ 2";

  for (auto& dep : dependecy_count_) {
    deps_.emplace_back(std::make_shared<interpreter::OpDepInfo>(dep));
  }
  for (size_t i = 0; i < variable_list_.size(); ++i) {
    refs_.emplace_back(std::make_shared<interpreter::VarRefInfo>(
        var_ref_count_[i], variable_list_[i]));
  }
  VLOG(4) << "calculate last_live_ops_ 3";
}

void NewIRInterpreter::ConstructEventForJitInput() {
  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      InstructionBase* inst = vec_instruction_base_[i].get();
      if (inst->Name() == "pd.memcpy_d2h" && platform::is_gpu_place(place_)) {
        for (auto& item : inst->Inputs()) {
          for (auto var_id : item.second) {
            auto name = GetNameById(var_id);
            if (JitInputVars().count(name)) {
              auto device_event = std::make_shared<platform::DeviceEvent>(
                  place_, platform::GenerateDeviceEventFlag());
              VLOG(4) << "Add input event for input: " << name << " of "
                      << inst->Name();
              inst->AddEventToWait(
                  i, device_event, ir_stream_analyzer_.GetWaiterType(inst));
            }
          }
        }
      }
    }
  }
}

FetchList NewIRInterpreter::BetaRun(const std::vector<std::string>& feed_names,
                                    bool need_fetch) {
  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  if (!is_build_) {
    LOG_FIRST_N(INFO, 1) << "New Executor is BetaRunning.";
    // Build
    std::stringstream ss;
    ss << this;
    ::ir::BuildScope(*ir_program_->block(),
                     InnerScope(),
                     ss.str(),
                     &value_2_var_name_,
                     &variable_2_var_name_,
                     &var_name_2_id_,
                     &variable_list_);
    VLOG(4) << DebugValueInfo();

    BuildInstruction();
    VLOG(4) << "Done BuildInstruction";

    PreAnalysis();
    VLOG(4) << "Done PreAnalysis";

    // Run
    BetaRunImpl();
  } else {
    BetaRunImpl();
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  Scope* inner_scope = InnerScope();
  auto* fetch_var = inner_scope->FindVar(interpreter::kFetchVarName);
  if (fetch_var && need_fetch) {
    auto fetch_list = std::move(*fetch_var->GetMutable<framework::FetchList>());
#ifdef PADDLE_WITH_CUDA
    if (platform::IsCUDAGraphCapturing()) {
      PADDLE_ENFORCE_EQ(fetch_list.empty(),
                        true,
                        platform::errors::InvalidArgument(
                            "Cannot fetch data when using CUDA Graph."));
    }
#endif
    return fetch_list;
  } else {
    return {};
  }
}

void NewIRInterpreter::NewIrLoopRunImpl() {
  for (size_t instr_id = 0; instr_id < vec_instruction_base_.size();
       ++instr_id) {
    vec_instruction_base_[instr_id]->Run();
  }
}

void NewIRInterpreter::BetaRunImpl() {
  // lazy initialization of gc, do not create gc is the program only run once
  if (!gc_) {
    gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_base_);
  }

  interpreter::ResetAtomicGuard guard(&deps_, &refs_);
  VLOG(4) << "Tracing Instruction List";

  TraceInstructionList(vec_instruction_base_);
  VLOG(4) << "Done BetaRunImpl";
}

void NewIRInterpreter::TraceInstructionList(
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i).get());
    }
  }

  for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
    auto instr_id = trace_execute_order_[idx];
    InstructionBase* instr_node = vec_instruction_base_.at(instr_id).get();

    VLOG(6) << "Run InstructionBase " << instr_id;
    RunInstructionBase(instr_node);

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
  VLOG(4) << "Done TraceInstructionList";
}

void NewIRInterpreter::RunInstructionBase(InstructionBase* instr_node) {
  platform::RecordEvent instruction_event(
      instr_node->Name(), platform::TracerEventType::Operator, 1);

  SetDeviceId(instr_node->DeviceContext().GetPlace());

  try {
    instr_node->WaitEvent(place_);

    VLOG(5) << "begin to run op " << instr_node->Name();
    if (!instr_node->IsArtificial()) {
      instr_node->Run();
      VLOG(4) << "done instruction node run";
      CheckGC(instr_node);
      VLOG(4) << "done CheckGC";
      interpreter::LogDeviceMemoryStats(place_);
    }
    VLOG(5) << "after run kernel";
    instr_node->RecordEvent(place_);
  } catch (platform::EnforceNotMet& ex) {
    LOG(WARNING) << instr_node->Name() << " raises an EnforceNotMet exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::make_exception_ptr(std::move(ex)));
  } catch (platform::EOFException&) {
    exception_holder_.Catch(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << instr_node->Name() << " raises an exception "
                 << platform::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::current_exception());
  } catch (...) {
    LOG(WARNING) << instr_node->Name() << " raises an unknown exception";
    exception_holder_.Catch(std::current_exception());
  }
}

void NewIRInterpreter::PreAnalysis() {
  BuildInstructionDependences();
  VLOG(4) << "Done BuildInstructionDependences";

  ir_stream_analyzer_.ConstructEvents(vec_instruction_base_);
  VLOG(4) << "Done ConstructEvents";

  // add event for the input var of jit program, since there are async copied
  // from gpu_pinned place to gpu place on compute stream.
  ConstructEventForJitInput();
  VLOG(4) << "AddEventToWait for JitInputVars";

  CalculateLastLiveOps();
  VLOG(4) << "Done CalculateLastLiveOps";

  AnalyseExecuteOrderForTrace(ir_dependency_builder_.OpDownstreamMap(),
                              ir_instruction_scheduling_priority_less);
  VLOG(4) << "Done AnalyseExecuteOrderForTrace";
}

}  // namespace framework
}  // namespace paddle
