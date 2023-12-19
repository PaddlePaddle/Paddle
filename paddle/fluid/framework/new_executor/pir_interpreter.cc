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

#include "paddle/fluid/framework/new_executor/pir_interpreter.h"

#include <chrono>
#include <unordered_set>

#include "paddle/utils/flags.h"

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
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/phi/backends/device_manager.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"
#endif

#include "paddle/fluid/framework/new_executor/instruction/builtin_combine_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/has_elements_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/if_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/legacy_kernel_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/select_input_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/tuple_pop_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/tuple_push_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/while_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

PHI_DECLARE_bool(enable_pir_in_executor);
PHI_DECLARE_bool(enable_pir_in_executor_trace_run);

#define CREATE_INSTR(instr_name)                                   \
  vec_instruction_base_.emplace_back(std::make_unique<instr_name>( \
      op_idx++, place_, &op, value_exe_info_.get()));

namespace paddle {
namespace framework {

PirInterpreter::PirInterpreter(const platform::Place& place,
                               const std::vector<std::string>& fetch_var_names,
                               const ::pir::Block* ir_block,
                               framework::Scope* scope,
                               const ExecutionConfig& execution_config)
    : place_(place),
      execution_config_(execution_config),
      var_scope_(scope),
      scope_(scope),
      ir_block_(ir_block),
      ir_stream_analyzer_(place),
      fetch_var_names_(fetch_var_names),
      enable_job_schedule_profiler_(false) {
  VLOG(4) << "PirInterpreter(): " << this << " on " << place_;

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  dependecy_count_ = std::make_shared<std::vector<size_t>>();

  if (!FLAGS_new_executor_use_local_scope) {
    execution_config_.create_local_scope = false;
  }
  if (execution_config_.create_local_scope) {
    auto local_scope = &scope_->NewScope();
    local_scope_ = local_scope;
    VLOG(6) << "pir interpretercore scope: " << scope_ << "\t"
            << "; local scope: " << local_scope_;
  }

  // TODO(zhangbo): delete var_scope
  var_scope_.SetLocalScope(local_scope_);

  execution_config_.AnalyzeThreadPoolConfig(place, 1);
  execution_config_.Log(/*log_level=*/8);

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

  value_exe_info_ = std::make_shared<ValueExecutionInfo>(InnerScope());

  std::stringstream ss;
  ss << this
     << std::chrono::high_resolution_clock::now().time_since_epoch().count();
  BuildScope(*ir_block_, ss.str(), value_exe_info_.get());

#if defined(PADDLE_WITH_CUDA)
  calculate_stream_timer_ = std::make_unique<phi::CalculateStreamTimer>(place);
#endif
}

PirInterpreter::PirInterpreter(
    const platform::Place& place,
    const std::vector<std::string>& fetch_var_names,
    const ::pir::Block* ir_block,
    framework::Scope* scope,
    std::shared_ptr<ValueExecutionInfo> value_exe_info,
    const ExecutionConfig& execution_config)
    : place_(place),
      execution_config_(execution_config),
      var_scope_(scope),
      scope_(scope),
      ir_block_(ir_block),
      value_exe_info_(value_exe_info),
      ir_stream_analyzer_(place),
      fetch_var_names_(fetch_var_names) {
  VLOG(4) << "PirInterpreter(): " << this << " on " << place_;

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  dependecy_count_ = std::make_shared<std::vector<size_t>>();

  if (!FLAGS_new_executor_use_local_scope) {
    execution_config_.create_local_scope = false;
  }
  if (execution_config_.create_local_scope) {
    auto local_scope = &scope_->NewScope();
    local_scope_ = local_scope;
    VLOG(6) << "pir interpretercore scope: " << scope_ << "\t"
            << "; local scope: " << local_scope_;
  }
  // TODO(zhangbo): delete var_scope
  var_scope_.SetLocalScope(local_scope_);

  execution_config_.AnalyzeThreadPoolConfig(place, 1);
  execution_config_.Log(/*log_level=*/8);

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

  std::stringstream ss;
  ss << this
     << std::chrono::high_resolution_clock::now().time_since_epoch().count();
  BuildScope(*ir_block_, ss.str(), value_exe_info_.get());
}

PirInterpreter::~PirInterpreter() {
  // cancle gc's thread
  gc_.reset(nullptr);
  async_work_queue_.reset();
  VLOG(4) << "~PirInterpreter(): " << this << " on " << place_;

#ifdef PADDLE_WITH_DNNL
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

std::shared_ptr<ProgramDesc> PirInterpreter::GetMutableCopyProgram() {
  PADDLE_THROW(platform::errors::Unimplemented(
      "GetMutableCopyProgram is not implemented in PirInterpreter."));
}

void PirInterpreter::SetSkipGcVars(const std::set<std::string>& skip_gc_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.skip_gc_vars.empty(),
      true,
      platform::errors::PreconditionNotMet(
          "execution_config_.skip_gc_vars can only be initialized once, now "
          "execution_config_.skip_gc_vars is "
          "not empty, do not call SetSkipGcVars method repeatedly."));
  execution_config_.skip_gc_vars = skip_gc_vars;
}

void PirInterpreter::SetJitInputVars(
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

const std::set<std::string>& PirInterpreter::JitInputVars() const {
  return execution_config_.jit_input_vars;
}

const VariableScope* PirInterpreter::GetVariableScope() const {
  return &var_scope_;
}

void PirInterpreter::reset_scope(Scope* new_scope) {
  var_scope_.SetScope(new_scope);
  scope_ = new_scope;
  for (size_t i = 0; i < value_exe_info_->GetVarList().size(); i++) {
    const auto& var_name = value_exe_info_->GetNameById(static_cast<int>(i));
    value_exe_info_->ResetVarList(static_cast<int>(i),
                                  new_scope->FindVar(var_name));
  }
  // The index should be assured valid, cause the InterpreterCore may not be
  // fully built, but was still cached and used. For example, see unit test
  // `test_assert.py`, it may exit before `PirInterpreter::Convert`,
  // but still was cached and used by later tests.
  for (size_t i = 0;
       i < std::min(refs_.size(), value_exe_info_->GetVarList().size());
       i++) {
    refs_[i]->ResetVariable(value_exe_info_->GetVarList()[i]);
  }
}

const Scope* PirInterpreter::local_scope() const { return local_scope_; }

void PirInterpreter::ShareWorkQueueFrom(InterpreterBaseImpl* src) {
  async_work_queue_ = reinterpret_cast<PirInterpreter*>(src)->GetWorkQueue();
  VLOG(8) << "Share AsyncWorkQueue from InterpreterCore(" << src
          << ") to InterpreterCore(" << this << ")";
}

void PirInterpreter::ShareBuildResultsFrom(const InterpreterBaseImpl& src) {
  const PirInterpreter& impl = dynamic_cast<const PirInterpreter&>(src);
  if (is_shared_results_build_ || !impl.IsSharedResultsBuild()) {
    return;
  }
  // share op dependency
  ir_dependency_builder_.ShareDependencyFrom(impl.GetPirDependencyBuilder());
  dependecy_count_ = impl.GetDependencyCount();
  // share event analysis
  ir_stream_analyzer_.ShareEventInfoFrom(impl.GetPirStreamAnalyzer());
  is_shared_results_build_ = true;
  VLOG(8) << "Share Build Results from InterpreterCore(" << &impl
          << ") to InterpreterCore(" << this << ")";
}

std::tuple<double, double> PirInterpreter::InterpreterRunTime() {
  double start_time = 0, end_time = 0;
#if defined(PADDLE_WITH_CUDA)
  start_time = calculate_stream_timer_->StartTime();
  end_time = calculate_stream_timer_->EndTime();
#endif
  return std::make_tuple(start_time, end_time);
}

const interpreter::PirDependencyBuilder&
PirInterpreter::GetPirDependencyBuilder() const {
  return ir_dependency_builder_;
}

std::shared_ptr<std::vector<size_t>> PirInterpreter::GetDependencyCount()
    const {
  return dependecy_count_;
}

const interpreter::PirStreamAnalyzer& PirInterpreter::GetPirStreamAnalyzer()
    const {
  return ir_stream_analyzer_;
}

bool PirInterpreter::IsSharedResultsBuild() const {
  return is_shared_results_build_;
}

std::shared_ptr<interpreter::AsyncWorkQueue> PirInterpreter::GetWorkQueue() {
  if (async_work_queue_ == nullptr) {
    async_work_queue_ = std::make_shared<interpreter::AsyncWorkQueue>(
        execution_config_.host_num_threads,
        execution_config_.device_num_threads,
        nullptr);
  }
  return async_work_queue_;
}

void PirInterpreter::PrepareForCUDAGraphCapture() {
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
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

void PirInterpreter::CheckCUDAGraphBeforeRun(
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

void PirInterpreter::ClearLoDTensorArrayInLocalScope() {
  auto vars = local_scope_->LocalVars();
  for (auto var : vars) {
    if (var->IsType<LoDTensorArray>()) {
      auto* lod_tensor_arr = var->GetMutable<LoDTensorArray>();
      lod_tensor_arr->clear();
    }
  }
}

std::string PirInterpreter::GetDepsString() const {
  std::stringstream ss;
  auto downstream_map = ir_dependency_builder_.OpDownstreamMap();
  ss << "Note: when static_dep is 1, it is ok that the dynamic_dep will not "
        "be decreased to 0."
     << std::endl;
  ss << "unfinished_op_number_:" << unfinished_op_number_ << std::endl;
  for (size_t i = 0; i < deps_.size(); ++i) {
    ss << "op:" << i << ", type: " << vec_instruction_base_[i]->Name()
       << ", static_dep:" << deps_[i]->StaticDep()
       << ", dynamic_dep:" << deps_[i]->DynamicDep() << ", downstream op: ";
    for (auto id : downstream_map[i]) {
      ss << id << ", ";
    }
    ss << std::endl;
  }
  return ss.str();
}

bool PirInterpreter::HasLocalScope() const { return local_scope_ != nullptr; }

Scope* PirInterpreter::InnerScope() const {
  return local_scope_ != nullptr ? local_scope_ : scope_;
}

std::string PirInterpreter::GetNameByValue(::pir::Value value) const {
  return value_exe_info_->GetVarName(value);
}

void PirInterpreter::UpdateSyncOpNum() {
  int64_t sync_op_num = 0;
  for (auto& ins : vec_instruction_base_) {
    if (ins->KernelType() == OpFuncType::kCpuSync ||
        ins->KernelType() == OpFuncType::kGpuSync) {
      sync_op_num = sync_op_num + 1;
    }
  }
  sync_op_num_ = sync_op_num;
  VLOG(4) << "Update sync op num, sync op num is: " << sync_op_num_;
}

void PirInterpreter::UpdateNcclOpNum() {
  static std::set<std::string> nccl_op_set = {
      "pd_op.c_softmax_with_cross_entropy",
      "pd_op.c_allgather",
      "pd_op.c_allreduce_max",
      "pd_op.c_allreduce_min",
      "pd_op.c_allreduce_sum",
      "pd_op.c_allreduce_prod",
      "pd_op.c_reduce_max",
      "pd_op.c_reduce_min",
      "pd_op.c_reduce_prod",
      "pd_op.c_reducescatter",
      "pd_op.c_broadcast",
      "pd_op.c_scatter",
      "pd_op.partial_send",
      "pd_op.partial_recv",
      "pd_op.partial_allgather",
      "pd_op.recv_v2",
      "pd_op.send_v2",
      "pd_op.mp_allreduce_sum",
      "pd_op.barrier",
      "pd_op.alltoall",
      "pd_op.global_gather",
      "pd_op.distributed_fused_lamb",
      "pd_op.margin_cross_entropy",
      "pd_op.sync_batch_norm",
      "pd_op.data_norm",
      "pd_op.class_center_sample",
      "pd_op.all_to_all",
      "pd_op.dist_concat",
      "pd_op.all_gather",
      "pd_op.broadcast",
      "pd_op.p_recv",
      "pd_op.p_send",
      "pd_op.reduce_scatter",
      "pd_op.all_reduce",
      "pd_op.reduce",
      "pd_op.c_softmax_with_cross_entropy_grad",
      "pd_op.c_allgather_grad",
      "pd_op.c_allreduce_max_grad",
      "pd_op.c_allreduce_min_grad",
      "pd_op.c_allreduce_sum_grad",
      "pd_op.c_allreduce_prod_grad",
      "pd_op.c_reduce_max_grad",
      "pd_op.c_reduce_min_grad",
      "pd_op.c_reduce_prod_grad",
      "pd_op.c_reducescatter_grad",
      "pd_op.c_broadcast_grad",
      "pd_op.c_scatter_grad",
      "pd_op.partial_send_grad",
      "pd_op.partial_recv_grad",
      "pd_op.partial_allgather_grad",
      "pd_op.recv_v2_grad",
      "pd_op.send_v2_grad",
      "pd_op.mp_allreduce_sum_grad",
      "pd_op.barrier_grad",
      "pd_op.alltoall_grad",
      "pd_op.global_gather_grad",
      "pd_op.distributed_fused_lamb_grad",
      "pd_op.margin_cross_entropy_grad",
      "pd_op.sync_batch_norm_grad",
      "pd_op.data_norm_grad",
      "pd_op.class_center_sample_grad",
      "pd_op.all_to_all_grad",
      "pd_op.dist_concat_grad",
      "pd_op.all_gather_grad",
      "pd_op.broadcast_grad",
      "pd_op.p_recv_grad",
      "pd_op.p_send_grad",
      "pd_op.reduce_scatter_grad",
      "pd_op.all_reduce_grad",
      "pd_op.reduce_grad",
      "pd_op.c_softmax_with_cross_entropy_",
      "pd_op.c_allgather_",
      "pd_op.c_allreduce_max_",
      "pd_op.c_allreduce_min_",
      "pd_op.c_allreduce_sum_",
      "pd_op.c_allreduce_prod_",
      "pd_op.c_reduce_max_",
      "pd_op.c_reduce_min_",
      "pd_op.c_reduce_prod_",
      "pd_op.c_reducescatter_",
      "pd_op.c_broadcast_",
      "pd_op.c_scatter_",
      "pd_op.partial_send_",
      "pd_op.partial_recv_",
      "pd_op.partial_allgather_",
      "pd_op.recv_v2_",
      "pd_op.send_v2_",
      "pd_op.mp_allreduce_sum_",
      "pd_op.barrier_",
      "pd_op.alltoall_",
      "pd_op.global_gather_",
      "pd_op.distributed_fused_lamb_",
      "pd_op.margin_cross_entropy_",
      "pd_op.sync_batch_norm_",
      "pd_op.data_norm_",
      "pd_op.class_center_sample_",
      "pd_op.all_to_all_",
      "pd_op.dist_concat_",
      "pd_op.all_gather_",
      "pd_op.broadcast_",
      "pd_op.p_recv_",
      "pd_op.p_send_",
      "pd_op.reduce_scatter_",
      "pd_op.all_reduce_",
      "pd_op.reduce_",
      "pd_op.c_softmax_with_cross_entropy_grad_",
      "pd_op.c_allgather_grad_",
      "pd_op.c_allreduce_max_grad_",
      "pd_op.c_allreduce_min_grad_",
      "pd_op.c_allreduce_sum_grad_",
      "pd_op.c_allreduce_prod_grad_",
      "pd_op.c_reduce_max_grad_",
      "pd_op.c_reduce_min_grad_",
      "pd_op.c_reduce_prod_grad_",
      "pd_op.c_reducescatter_grad_",
      "pd_op.c_broadcast_grad_",
      "pd_op.c_scatter_grad_",
      "pd_op.partial_send_grad_",
      "pd_op.partial_recv_grad_",
      "pd_op.partial_allgather_grad_",
      "pd_op.recv_v2_grad_",
      "pd_op.send_v2_grad_",
      "pd_op.mp_allreduce_sum_grad_",
      "pd_op.barrier_grad_",
      "pd_op.alltoall_grad_",
      "pd_op.global_gather_grad_",
      "pd_op.distributed_fused_lamb_grad_",
      "pd_op.margin_cross_entropy_grad_",
      "pd_op.sync_batch_norm_grad_",
      "pd_op.data_norm_grad_",
      "pd_op.class_center_sample_grad_",
      "pd_op.all_to_all_grad_",
      "pd_op.dist_concat_grad_",
      "pd_op.all_gather_grad_",
      "pd_op.broadcast_grad_",
      "pd_op.p_recv_grad_",
      "pd_op.p_send_grad_",
      "pd_op.reduce_scatter_grad_",
      "pd_op.all_reduce_grad_",
      "pd_op.reduce_grad_"};
  int64_t nccl_op_num = 0;
  for (auto& ins : vec_instruction_base_) {
    if (nccl_op_set.count(ins->Name())) {
      nccl_op_num = nccl_op_num + 1;
    } else if (ins->Operation()->HasAttribute("ring_id")) {
      nccl_op_num = nccl_op_num + 1;
    }
  }
  nccl_op_num_ = nccl_op_num;
  VLOG(4) << "Update nccl op num, nccl op num is: " << nccl_op_num;
}

// Note(zhangbo):
// When there is a KQueueSync type OP in the model, breadth traversal is better
// than depth traversal. For example: OP(O) ->(direct_run)-> OP(A)
// ->(sync_run)-> OP(B) OP(O) ->(direct_run)-> OP(C) ->(direct_run)-> OP(D) If B
// is run before C, B may always block to wait for A to finish executing, but in
// fact, C can be executed first during this time.
void PirInterpreter::AnalyseExecuteOrderForTrace(
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

  for (size_t instr_id = 0; instr_id < dependecy_count_->size(); ++instr_id) {
    if ((*dependecy_count_)[instr_id] == 0) {
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
      dependecy_count_->size(),
      platform::errors::PreconditionNotMet(
          "trace_order size should be equal to dependecy_count_."));

  trace_execute_order_ = trace_order;

  if (VLOG_IS_ON(6)) {
    std::stringstream ss;
    ss << "trace order: ";
    for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
      ss << vec_instruction_base_[trace_execute_order_[idx]]->Name() << "["
         << trace_execute_order_[idx] << "]"
         << " -> ";
    }
    ss << "end\n";
    VLOG(6) << ss.str();
  }
}

void PirInterpreter::BuildInstruction() {
  VLOG(6) << "Build Instructions for pir ... ";
  vec_instruction_base_.clear();
  size_t op_idx = 0;
  for (auto& op : *ir_block_) {
    VLOG(6) << "Build Instruction for op: " << op_idx;
    if (op.dialect()->name() == "builtin") {
      if (op.isa<pir::CombineOp>()) {
        vec_instruction_base_.emplace_back(
            std::make_unique<BuiltinCombineInstruction>(
                op_idx++, place_, &op, value_exe_info_.get()));
      } else if (interpreter::GetSpecialOpNames().count(op.name())) {
        VLOG(6) << "skip process builtin dialect op: " << op.name();
        continue;
      }
    } else if (op.dialect()->name() == "cf") {
      if (op.isa<pir::TuplePushOp>()) {
        CREATE_INSTR(TuplePushInstruction);
      } else if (op.isa<pir::TuplePopOp>()) {
        CREATE_INSTR(TuplePopInstruction);
      } else {
        VLOG(6) << "skip process cf dialect op: " << op.name();
        continue;
      }
    } else if (op.dialect()->name() == "pd_op") {
      if (op.isa<paddle::dialect::IfOp>()) {
        auto skip_gc_vars = execution_config_.skip_gc_vars;
        vec_instruction_base_.emplace_back(std::make_unique<IfInstruction>(
            op_idx++, place_, &op, value_exe_info_.get(), skip_gc_vars));
        sub_blocks_.insert(
            {&op.dyn_cast<paddle::dialect::IfOp>().true_block(),
             dynamic_cast<IfInstruction*>(vec_instruction_base_.back().get())
                 ->TrueBranchInterpreter()});
        sub_blocks_.insert(
            {&op.dyn_cast<paddle::dialect::IfOp>().false_block(),
             dynamic_cast<IfInstruction*>(vec_instruction_base_.back().get())
                 ->FalseBranchInterpreter()});
      } else if (op.isa<paddle::dialect::WhileOp>()) {
        auto skip_gc_vars = execution_config_.skip_gc_vars;
        vec_instruction_base_.emplace_back(std::make_unique<WhileInstruction>(
            op_idx++, place_, &op, value_exe_info_.get(), skip_gc_vars));
        sub_blocks_.insert(
            {&op.dyn_cast<paddle::dialect::WhileOp>().body(),
             dynamic_cast<WhileInstruction*>(vec_instruction_base_.back().get())
                 ->BodyInterpreter()});
      } else if (op.isa<paddle::dialect::HasElementsOp>()) {
        CREATE_INSTR(HasElementsInstruction);
      } else if (op.isa<paddle::dialect::SelectInputOp>()) {
        CREATE_INSTR(SelectInputInstruction);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Now only support pd_kernel and cinn dialect."));
      }
    } else if (op.dialect()->name() == "pd_kernel") {
      auto op_name = op.attributes()
                         .at("op_name")
                         .dyn_cast<::pir::StrAttribute>()
                         .AsString();
      if (interpreter::GetSpecialOpNames().count(op_name)) {
        VLOG(6) << "skip process " << op_name;
        continue;
      }
      VLOG(6) << "process " << op_name;

      if (op.isa<paddle::dialect::LegacyKernelOp>()) {
        CREATE_INSTR(LegacyKernelInstruction);
      } else {
        CREATE_INSTR(PhiKernelInstruction);
      }
#ifdef PADDLE_WITH_CINN
    } else if (op.dialect()->name() == "cinn_runtime") {
      CREATE_INSTR(CinnJitInstruction);
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Now only support pd_kernel and cinn dialect."));
    }
  }
}

std::string PirInterpreter::DebugInstructions() {
  std::stringstream ss;
  ss << "\n"
     << std::left << std::setw(7) << "id" << std::setw(40) << "instruction"
     << std::setw(40) << "inputs_id" << std::setw(40) << "outputs_id"
     << std::endl;
  uint64_t idx = 0;
  for (auto& instr : vec_instruction_base_) {
    ss << std::setw(7) << idx++ << std::setw(40) << instr->Name();

    std::stringstream ss_inputs;
    for (auto& input : instr->Inputs()) {
      ss_inputs << "{";
      for (auto id : input.second) {
        ss_inputs << id << " ";
      }
      ss_inputs << "} ";
    }
    ss << std::setw(40) << ss_inputs.str();

    std::stringstream ss_outputs;
    for (auto& output : instr->Outputs()) {
      ss_outputs << "{";
      for (auto id : output.second) {
        ss_outputs << id << " ";
      }
      ss_outputs << "} ";
    }
    ss << std::setw(40) << ss_outputs.str() << std::endl;
  }
  return ss.str();
}

std::string PirInterpreter::DebugValueInfo() {
  std::stringstream os;
  os << "value info of interpretercore " << this << "\n"
     << "value -> var_name -> id -> variable*"
     << "\n";

  interpreter::PrintValuesAndVariables(*ir_block_,
                                       value_exe_info_->GetValue2VarName(),
                                       value_exe_info_->GetVar2VarName());

  for (auto kv : value_exe_info_->GetValue2VarName()) {
    PADDLE_ENFORCE((bool)kv.first,
                   platform::errors::PreconditionNotMet(
                       "vlaue(%s) should not be nullptr", kv.second));
    PADDLE_ENFORCE(value_exe_info_->HasVar(kv.second),
                   platform::errors::PreconditionNotMet(
                       "var(%s) should exist in var_name_2_id_", kv.second));
    auto* var = InnerScope()->FindVar(kv.second);
    PADDLE_ENFORCE(
        var != nullptr,
        platform::errors::PreconditionNotMet(
            "var(%s) should exist in scope (%p)", kv.second, InnerScope()));
    os << kv.first.impl() << " -> " << kv.second << " -> "
       << value_exe_info_->GetVarId(kv.first) << " -> " << var << "\n";
  }
  return os.str();
}

void PirInterpreter::BuildInstructionDependences() {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << DebugInstructions();
  }
  // analysis the dependences between instructions, add next_instr_list to each
  // instr, and set the dependecy_count_
  size_t instr_num = vec_instruction_base_.size();
  dependecy_count_ = GetDependencyCount();
  if (!is_shared_results_build_) {
    dependecy_count_->assign(instr_num, 0);
  }
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

    if (!is_shared_results_build_) {
      for (size_t next_instr_id : next_instr_ids) {
        ++(*dependecy_count_)[next_instr_id];
      }
    }
  }
}

void PirInterpreter::RecordMemcpyD2H(InstructionBase* instr_node) {
  // NOTE(zhiqiu): hot fix for jit input var
  if (instr_node->Name() == "pd_op.memcpy_d2h") {
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

void PirInterpreter::RecordStreamForGC(InstructionBase* instr) {
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
// TODO(lizhiyu): Only analyse the 'send_v2' for GPT pp strategy right now.
// To support all the operators for communicating in the future.
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (instr->Name() == "pd_op.send_v2") {
    ::pir::Operation* op = instr->Operation();
    if (op->HasAttribute("use_calc_stream") &&
        op->attribute<::pir::BoolAttribute>("use_calc_stream").data() ==
            false) {
      int ring_id = op->attribute<::pir::Int32Attribute>("ring_id").data();
      if (FLAGS_dynamic_static_unified_comm) {
        const auto& comm_context_manager =
            phi::distributed::CommContextManager::GetInstance();
        stream = static_cast<phi::distributed::NCCLCommContext*>(
                     comm_context_manager.Get(std::to_string(ring_id)))
                     ->GetStream();
      } else {
        stream = platform::NCCLCommContext::Instance()
                     .Get(ring_id, instr->DeviceContext().GetPlace())
                     ->stream();
      }
    }
  }
#endif
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
    VLOG(4) << "GC sync " << value_exe_info_->GetNameById(var_id);

    // persistable var will be ignore while GC
    if (parameter_var_names_.count(value_exe_info_->GetNameById(var_id))) {
      VLOG(4) << value_exe_info_->GetNameById(var_id)
              << " is a parameter, skip gc";
      continue;
    }

    paddle::framework::Variable* var = value_exe_info_->GetVarList()[var_id];
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<phi::DenseTensor>()) {
      TensorRecordStream(*(var->GetMutable<phi::DenseTensor>()));
    } else if (
        var->IsType<
            operators::reader::
                OrderedMultiDeviceLoDTensorBlockingQueueHolder>()) {  // NOLINT
      // do nothing
    } else if (var->IsType<phi::SelectedRows>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SelectedRows>()->mutable_value()));
    } else if (var->IsType<LoDTensorArray>()) {
      auto* tensor_arr = var->GetMutable<LoDTensorArray>();
      for (auto& tensor : *tensor_arr) {
        TensorRecordStream(tensor);
      }
    } else if (var->IsType<phi::SparseCooTensor>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCooTensor>()->mutable_indices()));
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCooTensor>()->mutable_values()));
    } else if (var->IsType<phi::SparseCsrTensor>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_cols()));
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_crows()));
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_values()));
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

void PirInterpreter::CheckGC(InstructionBase* instr) {
  platform::RecordEvent record(
      "CheckGC", platform::TracerEventType::UserDefined, 10);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  RecordStreamForGC(instr);
#endif

  for (auto var_id : instr->GCCheckVars()) {
    VLOG(4) << "GC:" << value_exe_info_->GetNameById(static_cast<int>(var_id))
            << ", id:" << var_id << ", ref:" << refs_[var_id]->DynamicRef();
    bool is_ready = refs_[var_id]->CheckAndDecrease();
    // ignore all persistable var while GCphi
    if (parameter_var_names_.count(
            value_exe_info_->GetNameById(static_cast<int>(var_id)))) {
      VLOG(4) << value_exe_info_->GetNameById(static_cast<int>(var_id))
              << " is a parameter, skip gc";
      continue;
    }

    if (is_ready) {
      VLOG(6) << "Async delete variable with name : "
              << value_exe_info_->GetNameById(static_cast<int>(var_id));
      gc_->Add(refs_[var_id]->Var(), instr);
    }
  }

  for (auto var : instr->EagerGCVars()) {
    gc_->Add(var, instr);
  }
  instr->ClearEagerGCVars();
}

void PirInterpreter::CalculateLastLiveOps() {
  VLOG(4) << "PirInterpreter(): " << this << " start CalculateLastLiveOps";
  // calculate last_live_ops_
  for (size_t op_idx = 0; op_idx < vec_instruction_base_.size(); ++op_idx) {
    InstructionBase* instr = vec_instruction_base_[op_idx].get();
    std::set<size_t> gc_check_vars;

    const std::unordered_map<::pir::Value, std::vector<int>>& ins =
        instr->Inputs();
    const std::unordered_map<::pir::Value, std::vector<int>>& outs =
        instr->Outputs();
    std::unordered_multimap<::pir::Value, std::vector<int>> ins_and_outs{
        ins.begin(), ins.end()};

    if (instr->Name() != "pd_op.fetch") {
      ins_and_outs.insert(outs.begin(), outs.end());
    }

    VLOG(4) << "get gc check vars for: " << instr->Name();

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
      paddle::framework::Variable* var = inner_scope->FindVar(
          value_exe_info_->GetNameById(static_cast<int>(var_id)));
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::NotFound("Var(id=%d) should not be nullptr.",
                                     static_cast<int>(var_id)));
      if (var->IsType<phi::DenseTensor>() || var->IsType<phi::SelectedRows>() ||
          var->IsType<LoDTensorArray>() ||
          var->IsType<phi::SparseCooTensor>() ||
          var->IsType<phi::SparseCsrTensor>()) {
        last_live_ops_[var_id].insert(op_idx);
      } else {
        VLOG(4) << "not clear "
                << value_exe_info_->GetNameById(static_cast<int>(var_id))
                << " after " << instr->Name() << " because its type is "
                << framework::ToTypeName(var->Type());
      }
    }
    VLOG(4) << "update last_live_ops for: " << instr->Name();
  }
  // clear the last_live_ops list for all vars in skip_gc_vars
  for (const std::string& skip_gc_var : execution_config_.skip_gc_vars) {
    int var_id = value_exe_info_->GetIdByName(skip_gc_var);
    if (var_id != -1) {
      last_live_ops_[var_id].clear();
      VLOG(8) << "Skip gc for var: " << skip_gc_var;
    }
  }
  VLOG(4) << "clear the last_live_ops list for all vars in skip_gc_vars";

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  // For example,
  // b = op1(a)
  // c = op2(a, b)
  // in this case, a is the input of op1 and op2, we only need to check
  // a after op2, because op2 always uses a after op1.
  var_ref_count_.resize(value_exe_info_->GetVarList().size());
  VLOG(4) << "last_live_ops_.size() : " << last_live_ops_.size();
  for (auto kv : last_live_ops_) {
    for (auto val : kv.second) {
      VLOG(4) << "var: " << kv.first << " -> op: " << val;
    }
  }
  VLOG(4) << "var_ref_count_.size() : " << var_ref_count_.size();
  for (size_t i = 0; i < last_live_ops_.size(); ++i) {
    std::set<size_t> minumum_last_live_ops;
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
        VLOG(6) << "last live op of var " << i << " "
                << value_exe_info_->GetNameById(static_cast<int>(i)) << " : "
                << item << " " << vec_instruction_base_[item]->Name();
        minumum_last_live_ops.insert(item);
        vec_instruction_base_[item]->AddGCCheckVar(i);
      }
    }
    last_live_ops_[i] = minumum_last_live_ops;
    var_ref_count_[i] = static_cast<int>(last_live_ops_[i].size());
  }
  VLOG(4) << "shrink the last_live_ops list for all vars in skip_gc_vars";

  for (auto& dep : *dependecy_count_) {
    deps_.emplace_back(std::make_shared<interpreter::OpDepInfo>(dep));
  }
  for (size_t i = 0; i < value_exe_info_->GetVarList().size(); ++i) {
    refs_.emplace_back(std::make_shared<interpreter::VarRefInfo>(
        var_ref_count_[i], value_exe_info_->GetVarList()[i]));
  }
  VLOG(4) << "done CalculateLastLiveOps";
}

void PirInterpreter::ConstructEventForJitInput() {
  for (size_t i = 0; i < dependecy_count_->size(); ++i) {
    if ((*dependecy_count_)[i] == 0) {
      InstructionBase* inst = vec_instruction_base_[i].get();
      if (inst->Name() == "pd_op.memcpy_d2h" &&
          platform::is_gpu_place(place_)) {
        for (auto& item : inst->Inputs()) {
          for (auto var_id : item.second) {
            auto name = value_exe_info_->GetNameById(var_id);
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

paddle::framework::FetchList PirInterpreter::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors,
    bool need_fetch,
    bool enable_job_schedule_profiler) {
  enable_job_schedule_profiler_ = enable_job_schedule_profiler;

  auto FeedInput = [&] {
    VLOG(4) << "Feed inputs";
    for (size_t i = 0; i < feed_names.size(); ++i) {
      auto* feed_var = InnerScope()->FindVar(feed_names[i]);
      PADDLE_ENFORCE_NOT_NULL(
          feed_var,
          platform::errors::NotFound("Variable %s should not be nullptr.",
                                     feed_names[i]));

      auto feed_tensor = feed_var->GetMutable<phi::DenseTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
      feed_tensor->set_lod(feed_tensors[i].lod());
    }
  };

  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_DNNL
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  FeedInput();

  if (!is_build_) {
    LOG_FIRST_N(INFO, 1) << "New Executor is Running ...";
    VLOG(4) << DebugValueInfo();

    SolvePersisableVarNames();

    if (VLOG_IS_ON(6)) {
      std::stringstream ss;
      for (auto parameter : parameter_var_names_) {
        ss << parameter << ", ";
      }
      VLOG(6) << "Parameter value include: " << ss.str();
    }

    BuildInstruction();
    VLOG(4) << "Done BuildInstruction";

    PreAnalysis();
    VLOG(4) << "Done PreAnalysis";

    // Run
    if (FLAGS_enable_pir_in_executor_trace_run || nccl_op_num_ > 1 ||
        execution_config_.used_for_inference ||
        ((execution_config_.used_for_jit || execution_config_.used_for_cinn) &&
         (sync_op_num_ == 0))) {
      LOG_FIRST_N(INFO, 1) << "pir interpreter is running by trace mode ...";
      TraceRunImpl();
    } else {
      LOG_FIRST_N(INFO, 1)
          << "pir interpreter is running by multi-thread mode ...";
      MultiThreadRunImpl();
    }

    is_build_ = true;
    is_shared_results_build_ = true;
  } else {
    if (FLAGS_enable_pir_in_executor_trace_run || nccl_op_num_ > 1 ||
        execution_config_.used_for_inference ||
        ((execution_config_.used_for_jit || execution_config_.used_for_cinn) &&
         (sync_op_num_ == 0))) {
      TraceRunImpl();
    } else {
      MultiThreadRunImpl();
    }
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  // return Fetch Tensors
  Scope* inner_scope = InnerScope();
  framework::FetchList fetch_res;
  if (need_fetch) {
    for (auto& var_name : fetch_var_names_) {
      auto* var = inner_scope->FindVar(var_name);
      VLOG(4) << "fetch " << var_name << "[" << var << "]";
      fetch_res.push_back(var->Get<phi::DenseTensor>());
    }
  }

  VLOG(4) << "get fetch list size: " << fetch_res.size();
  return fetch_res;
}

FetchList PirInterpreter::Run(const std::vector<std::string>& feed_names,
                              bool need_fetch,
                              bool enable_job_schedule_profiler,
                              bool enable_op_profiling) {
  enable_job_schedule_profiler_ = enable_job_schedule_profiler;

  if (enable_op_profiling) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently PIR does not support op runtime profiling feature."));
  }

  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_DNNL
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  if (!is_build_) {
    LOG_FIRST_N(INFO, 1) << "New Executor is Running ...";
    VLOG(4) << DebugValueInfo();

    SolvePersisableVarNames();

    if (VLOG_IS_ON(6)) {
      std::stringstream ss;
      for (auto parameter : parameter_var_names_) {
        ss << parameter << ", ";
      }
      VLOG(6) << "Parameter value include: " << ss.str();
    }

    BuildInstruction();
    VLOG(4) << "Done BuildInstruction";

    PreAnalysis();
    VLOG(4) << "Done PreAnalysis";

    // Run
    if (FLAGS_enable_pir_in_executor_trace_run || nccl_op_num_ > 1 ||
        execution_config_.used_for_inference ||
        ((execution_config_.used_for_jit || execution_config_.used_for_cinn) &&
         (sync_op_num_ == 0))) {
      LOG_FIRST_N(INFO, 1) << "pir interpreter is running by trace mode ...";
      TraceRunImpl();
    } else {
      LOG_FIRST_N(INFO, 1)
          << "pir interpreter is running by multi-thread mode ...";
      MultiThreadRunImpl();
    }

    is_build_ = true;
    is_shared_results_build_ = true;
  } else {
    if (FLAGS_enable_pir_in_executor_trace_run || nccl_op_num_ > 1 ||
        execution_config_.used_for_inference ||
        ((execution_config_.used_for_jit || execution_config_.used_for_cinn) &&
         (sync_op_num_ == 0))) {
      TraceRunImpl();
    } else {
      MultiThreadRunImpl();
    }
  }

  if (HasLocalScope()) {
    ClearLoDTensorArrayInLocalScope();
  }

  framework::FetchList fetch_res;
  if (need_fetch) {
    // return Fetch Tensors
    Scope* inner_scope = InnerScope();

    for (auto& var_name : fetch_var_names_) {
      auto* var = inner_scope->FindVar(var_name);
      VLOG(4) << "fetch " << var_name << "[" << var << "]";
      fetch_res.push_back(var->Get<phi::DenseTensor>());
    }

    VLOG(4) << "get fetch list size: " << fetch_res.size();
  }
  return fetch_res;
}

void PirInterpreter::TraceRunImpl() {
  // lazy initialization of gc, do not create gc is the program only run once
  if (!gc_) {
    gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_base_);
  }

  interpreter::ResetAtomicGuard guard(&deps_, &refs_);
  VLOG(4) << "Tracing Instruction List";

  TraceRunInstructionList(vec_instruction_base_);
  VLOG(4) << "Done TraceRunInstructionList";
}

void PirInterpreter::MultiThreadRunImpl() {
  // lazy initialization of gc, do not create gc is the program only run once
  if (!gc_) {
    gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_base_);
  }

  interpreter::ResetAtomicGuard guard(&deps_, &refs_);
  VLOG(4) << "Multi Thread Run Instruction List";

  async_work_queue_ = GetWorkQueue();
  MultiThreadRunInstructionList(vec_instruction_base_);
  VLOG(4) << "Done MultiThreadRunInstructionList";
}

void PirInterpreter::TraceRunInstructionList(
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  if (enable_job_schedule_profiler_) {
    for (int i = trace_execute_order_.size() - 1; i >= 0; --i) {
      auto instr_id = trace_execute_order_[i];
      auto* instr_node = vec_instruction_base_.at(instr_id).get();
      std::string op_name = instr_node->Name();
      ::pir::Operation* op = instr_node->Operation();
      if (op_name != "pd_op.feed" && !op->HasAttribute("ring_id")) {
        VLOG(3) << "Last calculated op type: " << op_name;
        last_calculate_instr_id_ = instr_node->Id();
        break;
      }
    }
  }

  for (size_t i = 0; i < dependecy_count_->size(); ++i) {
    if ((*dependecy_count_)[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i).get());
    }
  }

  for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
    auto instr_id = trace_execute_order_[idx];
    InstructionBase* instr_node = vec_instruction_base_.at(instr_id).get();

    VLOG(6) << "Run InstructionBase " << instr_node->Name() << "[" << instr_id
            << "]";
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
  VLOG(4) << "Done TraceRunInstructionList";
}

void PirInterpreter::MultiThreadRunInstructionList(
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  if (enable_job_schedule_profiler_) {
    for (int i = vec_instr.size() - 1; i >= 0; --i) {
      auto* instr_node = vec_instr.at(i).get();
      std::string op_name = instr_node->Name();
      ::pir::Operation* op = instr_node->Operation();
      if (op_name != "pd_op.feed" && !op->HasAttribute("ring_id")) {
        VLOG(3) << "Last calculated op type: " << op_name;
        last_calculate_instr_id_ = vec_instr.at(i)->Id();
        break;
      }
    }
  }

  for (size_t i = 0; i < dependecy_count_->size(); ++i) {
    if ((*dependecy_count_)[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i).get());
      if (FLAGS_new_executor_serial_run) {
        RunInstructionBaseAsync(i);
      } else {
        async_work_queue_->AddTask(vec_instr.at(i)->KernelType(),
                                   [this, i] { RunInstructionBaseAsync(i); });
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
            VLOG(0) << "deps:\n" << GetDepsString();
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

void PirInterpreter::RunInstructionBaseAsync(size_t instr_id) {
  // NOTE(Ruibiao): Due to the uncertain order in multi-threading asynchronous
  // scheduling, the priority order involved cross-thread scheduling is not
  // guaranteed. Only Ops scheduled by the same AddTask call have the guarantee
  // of priority order.
  SchedulingQueue ready_ops(ir_instruction_scheduling_priority_less);
  ready_ops.push(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.top();
    ready_ops.pop();
    auto* instr_node = vec_instruction_base_.at(instr_id).get();

    RunInstructionBase(instr_node);

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

void PirInterpreter::RunNextInstructions(InstructionBase* instr,
                                         SchedulingQueue* reserved_next_ops) {
  platform::RecordEvent record(
      "RunNextInstructions", platform::TracerEventType::UserDefined, 10);

  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  for (size_t next_instr_id : instr->NextInstrsInDifferenceThread()) {
    if (IsReady(next_instr_id)) {
      async_work_queue_->AddTask(
          vec_instruction_base_[next_instr_id]->KernelType(),
          [this, next_instr_id]() { RunInstructionBaseAsync(next_instr_id); });
    }
  }

  for (size_t next_instr_id : instr->NextInstrsInSameThread()) {
    if (IsReady(next_instr_id)) {
      reserved_next_ops->push(next_instr_id);
    }
  }
}

void PirInterpreter::RunInstructionBase(InstructionBase* instr_node) {
  platform::RecordEvent instruction_event(
      instr_node->Name(), platform::TracerEventType::Operator, 1);

  auto cur_place = instr_node->DeviceContext().GetPlace();
  SetDeviceId(cur_place);

  try {
    instr_node->WaitEvent(cur_place);
#if defined(PADDLE_WITH_CUDA)
    if (enable_job_schedule_profiler_) {
      std::string op_name = instr_node->Name();
      ::pir::Operation* op = instr_node->Operation();
      if (!calculate_stream_timer_->IsStarted() && op_name != "pd_op.feed" &&
          !op->HasAttribute("ring_id")) {
        VLOG(3) << "Start calculated stream timer from op: " << op_name;
        calculate_stream_timer_->Start();
      }
    }
#endif
    VLOG(4) << "begin to run op " << instr_node->Name();
    VLOG(4) << "begin: " << __func__ << " OP id:" << instr_node->Id()
            << " name:" << instr_node->Name() << " type:"
            << (instr_node->KernelType() == OpFuncType::kCpuSync
                    ? "kCpuSync"
                    : (instr_node->KernelType() == OpFuncType::kGpuSync
                           ? "kGpuSync"
                           : "kGpuAsync"))
            << " runs on " << platform::GetCurrentThreadName();

    VLOG(4) << cur_place << " Before:"
            << instr_node->DebugStringEx(scope_, value_exe_info_.get());
    if (!instr_node->IsArtificial()) {
      instr_node->Run();

      if (FLAGS_benchmark) {
        instr_node->DeviceContext().Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
        VLOG(4) << "Operator(" << instr_node->Name()  // NOLINT
                << "): context wait and get last error";
#endif
      }
      VLOG(4) << "done instruction node run";
      VLOG(4) << "done: " << __func__ << " OP id:" << instr_node->Id()
              << " name:" << instr_node->Name() << " type:"
              << (instr_node->KernelType() == OpFuncType::kCpuSync
                      ? "kCpuSync"
                      : (instr_node->KernelType() == OpFuncType::kGpuSync
                             ? "kGpuSync"
                             : "kGpuAsync"))
              << " runs on " << platform::GetCurrentThreadName();

      VLOG(4) << cur_place << " After:"
              << instr_node->DebugStringEx(scope_, value_exe_info_.get());
      CheckGC(instr_node);
      VLOG(4) << "done CheckGC";
      memory::LogDeviceMemoryStats(cur_place, instr_node->Name());
    }
    VLOG(5) << "after run kernel";
    instr_node->RecordEvent(cur_place);
#if defined(PADDLE_WITH_CUDA)
    if (enable_job_schedule_profiler_) {
      if (instr_node->Id() == last_calculate_instr_id_ &&
          calculate_stream_timer_->IsStarted()) {
        VLOG(3) << "Stop calculated stream timer from op: "
                << instr_node->Name();
        calculate_stream_timer_->Stop();
      }
    }
#endif
  } catch (platform::EnforceNotMet& ex) {
    auto* op = instr_node->Operation();
    const std::vector<std::string> op_callstack_attr =
        interpreter::GetInstructionCallStack(op->name(), op->attributes());
    framework::InsertCallStackInfo(op->name(), op_callstack_attr, &ex);
    LOG(WARNING) << " OP id:" << instr_node->Id() << " " << instr_node->Name()
                 << " raises an EnforceNotMet exception "
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

void PirInterpreter::PreAnalysis() {
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

  UpdateSyncOpNum();
  VLOG(4) << "Done UpdateSyncOpNum";

  UpdateNcclOpNum();
  VLOG(4) << "Done UpdateNcclOpNum";
}

::pir::Value PirInterpreter::GetValueByName(const std::string& var_name) {
  for (auto kv : value_exe_info_->GetValue2VarName()) {
    if (kv.second == var_name) {
      return kv.first;
    }
  }
  return nullptr;
}

void PirInterpreter::SolvePersisableVarNames() {
  VLOG(6) << "SolvePersisableVarNames";
  for (auto kv : value_exe_info_->GetValue2VarName()) {
    ::pir::Value value = kv.first;
    const std::string& var_name = kv.second;
    auto bool_attr = value.attribute<::pir::BoolAttribute>(kAttrIsPersisable);
    if (bool_attr && bool_attr.data()) {
      parameter_var_names_.insert(var_name);
    }
  }
}

Variable* PirInterpreter::DebugVar(const std::string& name) const {
  Scope* scope = HasLocalScope() ? local_scope_ : scope_;
  auto* var = scope->FindVar(name);
  if (var != nullptr) {
    return var;
  }
  for (auto kv : sub_blocks_) {
    var = kv.second->DebugVar(name);
    if (var != nullptr) {
      return var;
    }
  }
  return var;
}

void PirInterpreter::Build(
    const std::vector<std::string>& feed_names,
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "Build is not implemented in PirInterpreter."));
}

void PirInterpreter::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "SetCopyProgram is not implemented in PirInterpreter."));
}

}  // namespace framework
}  // namespace paddle
