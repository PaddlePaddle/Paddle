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
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_functor.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/event_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/fast_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/kernel_context.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

PADDLE_DEFINE_EXPORTED_bool(new_executor_use_inplace, true,
                            "Use inplace in new executor");
PADDLE_DEFINE_EXPORTED_bool(new_executor_use_local_scope, true,
                            "Use local_scope in new executor(especially used "
                            "in UT), can turn off for better performance");

DECLARE_bool(check_nan_inf);
DECLARE_bool(benchmark);
DECLARE_bool(fast_eager_deletion_mode);

constexpr const char* kExceptionCaught = "ExceptionCaught";
constexpr const char* kTaskCompletion = "TaskCompletion";

namespace paddle {
namespace framework {
// NOTE(Aurelius84): Need a better strategy to determine it.
static constexpr size_t kHostNumThreads = 4;
static constexpr size_t kDeviceNumThreads = 1;

bool IsInterpretercoreFastGCEnabled() {
  return memory::allocation::AllocatorFacade::Instance()
             .IsStreamSafeCUDAAllocatorUsed() &&
         FLAGS_fast_eager_deletion_mode;
}

InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const BlockDesc& block,
                                 VariableScope* global_scope)
    : place_(place),
      block_(block),
      global_scope_(global_scope),
      stream_analyzer_(place) {
  VLOG(4) << "InterpreterCore(): " << this << " on " << place_;
  is_build_ = false;
  async_work_queue_.reset(new interpreter::AsyncWorkQueue(
      kHostNumThreads, kDeviceNumThreads, &main_thread_blocker_));

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (IsInterpretercoreFastGCEnabled()) {
    gc_ = std::make_unique<InterpreterCoreFastGarbageCollector>();
  } else {
    gc_ = std::make_unique<InterpreterCoreEventGarbageCollector>();
  }
#else
  gc_ = std::make_unique<InterpreterCoreEventGarbageCollector>();
#endif

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  create_local_scope_ = FLAGS_new_executor_use_local_scope;
  if (FLAGS_new_executor_use_local_scope) {
    auto local_scope = &global_scope->GetMutableScope()->NewScope();
    local_scope->AddListener(global_scope->Listener());
    local_scope_ = local_scope;
  }
  VLOG(4) << "create_local_scope_ is " << create_local_scope_;

  // prune

  // optmize graph pass

  // convert to run graph
}

InterpreterCore::~InterpreterCore() {
  // cancle gc's thread
  gc_.reset(nullptr);

  async_work_queue_.reset(nullptr);
  VLOG(4) << "~InterpreterCore(): " << this;
  VLOG(4) << " on" << place_;

#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

void InterpreterCore::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  copy_program_ = prog;
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors) {
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif
  bool is_build = is_build_;
  global_scope_->SetLocalScope(local_scope_);
  Prepare(feed_names, feed_tensors, is_build);

  if (is_build) {
    // add listener before run and is_build=true
    global_scope_->ResetListener();

    ExecuteInstructionList(vec_instruction_);
  }

  if (create_local_scope_) {
    ClearLoDTensorArrayInLocalScope();
  }

  // clear the listener after run
  global_scope_->ClearListener();

  // return Fetch Tensors
  auto* fetch_var = global_scope_->Var(interpreter::kFetchVarName);
  return std::move(*fetch_var->GetMutable<framework::FetchList>());
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names) {
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif
  if (!is_build_) {
    if (create_local_scope_ &&
        global_scope_->GetMutableLocalScope() !=
            global_scope_->GetMutableScope() &&
        global_scope_->GetMutableLocalScope()) {
      VLOG(4) << "Clear previous local scope before run";
      VLOG(4) << global_scope_->GetMutableScope() << " "
              << global_scope_->GetMutableLocalScope();
      platform::DeviceContextPool::Instance().Get(place_)->Wait();
      // TODO(zhiqiu): clear the tensor holder of all vars in previous local
      // scope?
    }
    global_scope_->SetLocalScope(local_scope_);
    paddle::framework::interpreter::build_variable_scope(block_, global_scope_,
                                                         create_local_scope_);
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::build_op_func_list(
        place_, block_, &op_func_nodes, global_scope_, create_local_scope_);
    is_build_ = true;
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);

  } else {
    // add listener before run and is_build=true
    global_scope_->ResetListener();

    ExecuteInstructionList(vec_instruction_);
  }

  if (create_local_scope_) {
    ClearLoDTensorArrayInLocalScope();
  }

  // clear the listener after run
  global_scope_->ClearListener();

  // return Fetch Tensors
  auto* fetch_var = global_scope_->Var(interpreter::kFetchVarName);
  return std::move(*fetch_var->GetMutable<framework::FetchList>());
}

// At the end of each step, the holder of Tensor in LoDTensorArray is null.
// Clear these Tensors and leave LoDTensorArray empty, otherwise an exception
// will occur in the next step
void InterpreterCore::ClearLoDTensorArrayInLocalScope() {
  auto vars = local_scope_->LocalVars();
  for (auto var : vars) {
    if (var->IsType<LoDTensorArray>()) {
      auto* lod_tensor_arr = var->GetMutable<LoDTensorArray>();
      lod_tensor_arr->clear();
    }
  }
}

void InterpreterCore::BuildOperatorDependences() {
  // analysis the dependences between ops, set the dependecy_count_ and Call
  // Schedule
  auto op_nums = vec_instruction_.size();
  dependecy_count_.resize(op_nums);
  auto op2downstream = interpreter::build_op_downstream_map(
      vec_instruction_, &op_happens_before_);
  for (size_t op = 0; op < vec_instruction_.size(); ++op) {
    auto op_list = op2downstream[op];
    std::vector<size_t> downsteam_vector(op_list.begin(), op_list.end());
    stream_analyzer_.Schedule(downsteam_vector, &vec_instruction_, op);

    for (auto inst_id : op_list) {
      dependecy_count_[inst_id]++;
    }
  }
}

void InterpreterCore::Convert(
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  auto& vec_meta_info = global_scope_->MutableVecMetaInfo();
  auto var_nums = global_scope_->VarSize();
  input_var2op_info_.resize(var_nums);
  auto nodes = *op_func_nodes;

  auto op_nums = nodes.size();
  vec_instruction_.reserve(op_nums);
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    auto& op_func_node = nodes[op_idx];
    auto* dev_ctx_ = stream_analyzer_.ParseDeviceContext(op_func_node);
    vec_instruction_.emplace_back(op_idx, std::move(op_func_node), *dev_ctx_);
  }

  BuildOperatorDependences();

  // calculate last_live_ops_
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    auto& instr = vec_instruction_[op_idx];
    OpInOutInfo info;
    std::set<size_t> gc_check_inputs;

    for (auto& item : instr.Inputs()) {
      for (auto id : item.second) {
        if (id == kEmptyVarIndex) {
          continue;
        }
        input_var2op_info_.at(id).push_back(op_idx);
        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(instr.OpBase());
        }
        auto* var_desc = global_scope_->VarDesc(id);
        if (var_desc) {
          if (info.IsInArgBufferNeeded(var_desc->Name())) {
            gc_check_inputs.insert(id);
          }
        } else {
          gc_check_inputs.insert(id);
        }
      }
    }

    for (auto var_id : gc_check_inputs) {
      paddle::framework::Variable* var = global_scope_->Var(var_id);
      if (var->IsType<LoDTensor>() || var->IsType<phi::SelectedRows>() ||
          var->IsType<LoDTensorArray>()) {
        last_live_ops_[var_id].insert(op_idx);
      } else {
        VLOG(4) << "not clear " << global_scope_->GetNameById(var_id)
                << " after " << instr.OpBase()->Type()
                << " because its type is "
                << framework::ToTypeName(var->Type());
      }
    }
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    // checkout ouput
    for (auto& item : vec_instruction_[i].Outputs()) {
      for (auto var_id : item.second) {
        if (input_var2op_info_.at(var_id).size() == 0) {
          last_live_ops_[var_id].insert(i);
        }
      }
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
        if (op_happens_before_[item][other_item]) {
          VLOG(8) << "happens_before: " << item << "->" << other_item
                  << ", so skip " << item;
          not_before_any = false;
          break;
        }
      }
      if (not_before_any) {
        VLOG(8) << "last live op of var " << i << " "
                << global_scope_->GetNameById(i) << " : " << item << " "
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

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    gc_event_.emplace_back(vec_instruction_[i].DeviceContext().GetPlace(),
                           platform::GenerateDeviceEventFlag());
  }
  bool inplaced = false;
  for (auto inst : vec_instruction_) {
    if (inst.OpBase()->Type() == "share_buffer" ||
        inst.OpBase()->Type() == "share_data") {
      VLOG(4) << "Already inplaced, skip inplace now.";
      inplaced = true;
    }
  }

  if (FLAGS_new_executor_use_inplace && !inplaced) {
    BuildInplace();
  }

  // prepare for the first time.
  async_work_queue_->PrepareAtomicDeps(dependecy_count_);
  async_work_queue_->PrepareAtomicVarRef(vec_meta_info);
}

bool InterpreterCore::BuildInplaceCheckVarIsOnlyInput(size_t var_index) {
  if (!global_scope_->VarDesc(var_index)) {
    return input_var2op_info_.at(var_index).size() == 1;
  } else {
    int is_input_cnt = 0;
    for (auto inst_id : input_var2op_info_.at(var_index)) {
      OpInOutInfo info;
      info.Build(vec_instruction_.at(inst_id).OpBase());
      if (info.IsInArgBufferNeeded(global_scope_->VarDesc(var_index)->Name())) {
        is_input_cnt++;
      }
    }
    return is_input_cnt == 1;
  }
}

void InterpreterCore::BuildInplace() {
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
        auto in_var_desc = global_scope_->VarDesc(iter->second[0]);
        if (in_var_desc && in_var_desc->Persistable()) {
          continue;
        }
        if (global_scope_->GetVarSikpInplace(iter->second[0])) {
          continue;
        }
        if (BuildInplaceCheckVarIsOnlyInput(iter->second[0])) {
          auto iterout = outputs.find(pair.second);
          if (iterout != outputs.end() && !iterout->second.empty()) {
            auto invar = global_scope_->Var(iter->second[0]);
            auto outvar = global_scope_->Var(iterout->second[0]);
            if (invar && outvar && invar->IsType<LoDTensor>() &&
                outvar->IsType<LoDTensor>()) {
              instr.AddInplace(invar, outvar);
              VLOG(3) << "inplace " << vec_instruction_[i].OpBase()->Type()
                      << " " << global_scope_->GetNameById(iter->second[0])
                      << " -> "
                      << global_scope_->GetNameById(iterout->second[0])
                      << std::endl;
            }
          }
        }
      }
    }
  }
}

void InterpreterCore::BuildAndCacheInstructionCtx(Instruction* instr_node) {
  VariableValueMap ins_map;
  for (auto& var_name_item : instr_node->Inputs()) {
    std::vector<Variable*> input_vars;

    input_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      input_vars.emplace_back(global_scope_->Var(id));
    }
    ins_map.emplace(var_name_item.first, std::move(input_vars));
  }

  VariableValueMap outs_map;
  for (auto& var_name_item : instr_node->Outputs()) {
    std::vector<Variable*> out_vars;

    out_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      out_vars.emplace_back(global_scope_->Var(id));
    }
    outs_map.emplace(var_name_item.first, std::move(out_vars));
  }

  // set runtime_ctx and infershape_ctx_
  if (instr_node->OpBase()->Type() == "cinn_launch") {  // OP use scope in
                                                        // kernel
    Scope* local_scope = create_local_scope_
                             ? global_scope_->GetMutableLocalScope()
                             : global_scope_->GetMutableScope();
    instr_node->ResetContextWithScope(ins_map, outs_map, *local_scope);
  } else {
    instr_node->ResetContext(ins_map, outs_map);
  }
}

void InterpreterCore::BuildSkipShareLoDInfo() {
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    bool can_skip_lod = true;
    for (auto& input : vec_instruction_[i].InnerRuntimeContext()->inputs) {
      for (auto& var : input.second) {
        if (var->IsType<LoDTensor>()) {
          if (var->Get<LoDTensor>().lod().size() != 0) {
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

void InterpreterCore::RunInstruction(const Instruction& instr_node) {
  auto* op = instr_node.OpBase();
  auto place = instr_node.DeviceContext().GetPlace();
  VLOG(4) << "Start run " << place << " " << op->DebugStringEx(global_scope_);
  Scope* local_scope = create_local_scope_
                           ? global_scope_->GetMutableLocalScope()
                           : global_scope_->GetMutableScope();
  auto op_with_kernel = dynamic_cast<const framework::OperatorWithKernel*>(op);
  {
    // If it is OperatorBase, InferShape do nothing.
    if (op_with_kernel != nullptr) {
      platform::RecordEvent infershape_event(
          "infer_shape", platform::TracerEventType::OperatorInner, 1,
          platform::EventRole::kInnerOp);

      // see OperatorWithKernel::RunImpl in operator.cc for why
      if (!(op_with_kernel->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op_with_kernel->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        op_with_kernel->Info().infer_shape_(
            instr_node.InnerInferShapeContext().get());
      }
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
        "compute", platform::TracerEventType::OperatorInner, 1,
        platform::EventRole::kInnerOp);
    if (op_with_kernel == nullptr) {
      instr_node.OpBase()->Run(*local_scope, place_);
    } else {
      // fit for phi
      if (instr_node.PhiKernel() && instr_node.PhiKernel()->IsValid()) {
        VLOG(4) << "Run phi kernel: " << op->Type();
        VLOG(4) << instr_node.InnerRuntimeContext().get() << " "
                << &instr_node.DeviceContext();
        phi::KernelContext pt_kernel_context;
        op_with_kernel->BuildPhiKernelContext(
            *instr_node.InnerRuntimeContext().get(),
            const_cast<platform::DeviceContext*>(&instr_node.DeviceContext()),
            &pt_kernel_context);

        (*instr_node.PhiKernel())(&pt_kernel_context);

      } else {
        instr_node.KernelFunc()(*instr_node.InnerExecutionContext().get());
      }
    }
  }

  VLOG(4) << "End run " << place << " " << op->DebugStringEx(global_scope_);

  if (!instr_node.InplaceBackMap().empty()) {
    platform::RecordEvent inplaceback_event(
        "InplaceVarsBack", platform::TracerEventType::UserDefined, 10);
    auto& m = instr_node.InplaceBackMap();
    // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in operator.cc
    for (auto& p : m) {
      auto* transformed_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
          global_scope_->Var(p.first));
      auto* original_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
          global_scope_->Var(p.second));
      original_tensor->ShareDataWith(*transformed_tensor);
      VLOG(4) << "Transfer inplace variable back form "
              << global_scope_->GetNameById(p.first) << " to "
              << global_scope_->GetNameById(p.second);
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
        *op, *global_scope_,
        place);  // TODO(xiongkun03) change it to inner scope.
  }
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr) {
  unfinished_op_numer_ = vec_instr.size();
  if (unfinished_op_numer_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  platform::RecordEvent record_prepare(
      "PrepareAtomic", platform::TracerEventType::UserDefined, 1);
  // NOTE(zhiqiu): get the prepared deps from std::future, and async prepare
  // those for the next step
  auto atomic_deps = async_work_queue_->AtomicDeps();
  auto atomic_var_ref = async_work_queue_->AtomicVarRef();

  async_work_queue_->PrepareAtomicDeps(dependecy_count_);
  async_work_queue_->PrepareAtomicVarRef(global_scope_->VecMetaInfo());
  record_prepare.End();

  exception_holder_.Clear();

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      async_work_queue_->AddTask(vec_instr.at(i).KernelType(), [
        this, i, atomic_deps = atomic_deps.get(),
        atomic_var_ref = atomic_var_ref.get()
      ] { RunInstructionAsync(i, atomic_deps, atomic_var_ref); });
    }
  }

  auto event_name = main_thread_blocker_.WaitEvent();
  VLOG(1) << "event_name: " << event_name;

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    // Graceful exit when the executor encountered a fatal error.
    // EOF is not a fatal error.
    if (exception_holder_.Type() != "EOF") {
      async_work_queue_->Cancel();
    }
    VLOG(4) << "Cancel ok";
    PADDLE_ENFORCE_EQ(
        main_thread_blocker_.Clear(), 0,
        platform::errors::PreconditionNotMet(
            "main_thread_blocker_.Clear() return -1, clear failed"));
    VLOG(4) << "clear ok";
    exception_holder_.ReThrow();
  }
}

void InterpreterCore::RunNextInstructions(
    const Instruction& instr, std::queue<size_t>* reserved_next_ops,
    std::vector<std::atomic<size_t>>* atomic_deps,
    std::vector<std::atomic<size_t>>* atomic_var_ref) {
  platform::RecordEvent record("RunNextInstructions",
                               platform::TracerEventType::UserDefined, 10);
  VLOG(4) << "atomic 1:" << atomic_deps;
  auto& next_instr = instr.NextInstructions();

  auto IsReady = [atomic_deps](size_t next_id) {
    VLOG(4) << "atomic:" << atomic_deps << " op_id: " << next_id
            << ", remain deps: " << (*atomic_deps)[next_id];
    return (*atomic_deps)[next_id].fetch_sub(1, std::memory_order_relaxed) == 1;
  };

  if (instr.KernelType() == OpFuncType::kQueueAsync) {
    // move all sync_ops into other threads
    for (auto next_id : next_instr.SyncRunIds()) {
      if (IsReady(next_id)) {
        async_work_queue_->AddTask(
            vec_instruction_[next_id].KernelType(),
            [this, next_id, atomic_deps, atomic_var_ref]() {
              RunInstructionAsync(next_id, atomic_deps, atomic_var_ref);
            });
      }
    }
    // keep all async_ops running in current thread
    for (auto next_id : next_instr.DirectRunIds()) {
      if (IsReady(next_id)) {
        reserved_next_ops->push(next_id);
      }
    }
    for (auto next_id : next_instr.EventRunIds()) {
      if (IsReady(next_id)) {
        reserved_next_ops->push(next_id);
      }
    }
  } else {
    // move async_ops into async_thread
    for (auto next_id : next_instr.EventRunIds()) {
      if (IsReady(next_id)) {
        async_work_queue_->AddTask(
            vec_instruction_[next_id].KernelType(),
            [this, next_id, atomic_deps, atomic_var_ref] {
              RunInstructionAsync(next_id, atomic_deps, atomic_var_ref);
            });
      }
    }
    auto direct_run_ops = interpreter::merge_vector(next_instr.SyncRunIds(),
                                                    next_instr.DirectRunIds());
    size_t first_op = 0;
    for (auto next_id : direct_run_ops) {
      if (IsReady(next_id)) {
        // only keep one op running in current thread
        if (first_op == 0) {
          first_op = next_id;
          continue;
        }
        // move rest ops into other threads
        async_work_queue_->AddTask(
            vec_instruction_[next_id].KernelType(),
            [this, next_id, atomic_deps, atomic_var_ref] {
              RunInstructionAsync(next_id, atomic_deps, atomic_var_ref);
            });
      }
    }
    if (first_op != 0) reserved_next_ops->push(first_op);
  }
}

void InterpreterCore::RunInstructionAsync(
    size_t instr_id, std::vector<std::atomic<size_t>>* atomic_deps,
    std::vector<std::atomic<size_t>>* atomic_var_ref) {
  std::queue<size_t> ready_ops;
  ready_ops.push(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.front();
    ready_ops.pop();
    auto& instr_node = vec_instruction_.at(instr_id);
    VLOG(5) << __func__ << " OP id:" << instr_node.Id()
            << " name:" << instr_node.OpBase()->Type()
            << " type:" << (instr_node.KernelType() == OpFuncType::kQueueSync
                                ? "kQueueSync"
                                : "kQueueAsync")
            << " runs on " << platform::GetCurrentThreadName();

    auto* op = instr_node.OpBase();
    platform::RecordEvent instruction_event(
        op->Type(), platform::TracerEventType::Operator, 1);

    try {
      interpreter::WaitEvent(instr_node, place_);

      RunInstruction(instr_node);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      RecordStreamForGC(instr_node);
#endif
      CheckGC(instr_node, atomic_var_ref);

      interpreter::RecordEvent(instr_node, place_);
    } catch (platform::EnforceNotMet& ex) {
      framework::InsertCallStackInfo(op->Type(), op->Attrs(), &ex);
      exception_holder_.Catch(std::make_exception_ptr(std::move(ex)));
    } catch (platform::EOFException&) {
      exception_holder_.Catch(std::current_exception());
    } catch (std::exception& ex) {
      LOG(WARNING) << op->Type() << " raises an exception "
                   << platform::demangle(typeid(ex).name()) << ", "
                   << ex.what();
      exception_holder_.Catch(std::current_exception());
    } catch (...) {
      LOG(WARNING) << op->Type() << " raises an unknown exception";
      exception_holder_.Catch(std::current_exception());
    }

    if (UNLIKELY(exception_holder_.IsCaught())) {
      VLOG(4) << "Exception caught";
      if (exception_notifier_ != nullptr) {
        exception_notifier_->NotifyEvent();
      }
      return;
    }

    VLOG(4) << "unfinished_op_numer_: " << unfinished_op_numer_;
    if (UNLIKELY(unfinished_op_numer_.fetch_sub(1, std::memory_order_relaxed) ==
                 1)) {
      if (completion_notifier_ != nullptr) {
        completion_notifier_->NotifyEvent();
      }
    }

    RunNextInstructions(instr_node, &ready_ops, atomic_deps, atomic_var_ref);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void InterpreterCore::RecordStreamForGC(const Instruction& instr) {
  if (!IsInterpretercoreFastGCEnabled() ||
      instr.KernelType() != OpFuncType::kQueueAsync) {
    return;
  }
  platform::RecordEvent record("RecordStreamForGC",
                               platform::TracerEventType::UserDefined, 10);

  gpuStream_t stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                           instr.DeviceContext())
                           .stream();
  auto TensorRecordStream = [&stream](Tensor& tensor) {
    auto allocation = tensor.Holder();
    if (allocation == nullptr) {
      return;
    }

    const platform::Place& place = allocation->place();
    if (platform::is_gpu_place(place)) {
      memory::RecordStream(allocation, stream);
    } else if (platform::is_cuda_pinned_place(place)) {
      // TODO(Ruibiao): Here should do something to make sure that the tensor is
      // not freed until the H2D copies done. However, simplely launch a CUDA
      // runtime callback to the H2D stream may lead a high performance
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
   * 3. The tensor is the instruction's input, cause we assume that instruction
   * will initialize all output tensors with its running stream.
   * 4. In the OP function of this instruction, the tensor is an input of a
   * async CUDA kernel.
   *
   * Here we only process the first condition, because:
   * 1. Since the RecordStream function will directly return when the recored
   * stream is equal to the owning stream, recording a stream same as which
   * initialized this tensor has less time overhead. Conversely, it may take
   * more time if we try to extract those cross-stream input vars from
   * instr.GCCheckVars.
   * 2. Now the instruction has no idea of which vars involving async running in
   * OP function, and thus we can not recognize condition 4. It should be
   * supported later.
   */
  for (int var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC sync " << global_scope_->GetNameById(var_id) << " "
            << global_scope_->VarDesc(var_id);

    // persistable var will be ignore while GC
    if (global_scope_->VarDesc(var_id) &&
        global_scope_->VarDesc(var_id)->Persistable()) {
      continue;
    }

    paddle::framework::Variable* var = global_scope_->Var(var_id);
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<LoDTensor>()) {
      TensorRecordStream(*(var->GetMutable<LoDTensor>()));
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
}
#endif

void InterpreterCore::CheckGC(
    const Instruction& instr,
    std::vector<std::atomic<size_t>>* atomic_var_ref) {
  platform::RecordEvent record("CheckGC",
                               platform::TracerEventType::UserDefined, 10);
  size_t instr_id = instr.Id();
  auto& var_scope = *global_scope_;

  for (auto var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC " << global_scope_->GetNameById(var_id) << " "
            << var_scope.VarDesc(var_id);
    VLOG(4) << "atomic:" << atomic_var_ref << " " << &(*atomic_var_ref)[var_id]
            << " " << var_id;
    bool is_ready =
        (*atomic_var_ref)[var_id].fetch_sub(1, std::memory_order_relaxed) == 1;
    // ignore all persistable var while GC
    if (var_scope.VarDesc(var_id) && var_scope.VarDesc(var_id)->Persistable()) {
      continue;
    }
    if (is_ready) {
      VLOG(6) << "Async delete variable with name : "
              << var_scope.GetNameById(var_id);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (IsInterpretercoreFastGCEnabled()) {
        static_cast<InterpreterCoreFastGarbageCollector*>(gc_.get())->Add(
            var_scope.Var(var_id));

      } else {
        static_cast<InterpreterCoreEventGarbageCollector*>(gc_.get())->Add(
            var_scope.Var(var_id), &gc_event_.at(instr_id),
            &instr.DeviceContext());
      }
#else
      static_cast<InterpreterCoreEventGarbageCollector*>(gc_.get())->Add(
          var_scope.Var(var_id), &gc_event_.at(instr_id),
          &instr.DeviceContext());
#endif
    }
  }
}

void InterpreterCore::Prepare(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors, bool prepare_feed) {
  PADDLE_ENFORCE_EQ(feed_names.size(), feed_tensors.size(),
                    platform::errors::PreconditionNotMet(
                        "Required feed_names.size() == feed_tensors.size(), "
                        "but received %d != %d",
                        feed_names.size(), feed_tensors.size()));

  auto FeedInput = [&] {
    VLOG(4) << "Feed inputs";
    for (size_t i = 0; i < feed_names.size(); ++i) {
      auto* feed_var = global_scope_->FindVar(feed_names[i]);
      PADDLE_ENFORCE_NOT_NULL(
          feed_var, platform::errors::NotFound(
                        "Variable %s should not be nullptr.", feed_names[i]));

      auto feed_tensor = feed_var->GetMutable<framework::LoDTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
      feed_tensor->set_lod(feed_tensors[i].lod());
    }
  };

  if (!is_build_) {
    paddle::framework::interpreter::build_variable_scope(block_, global_scope_,
                                                         create_local_scope_);
    FeedInput();
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::build_op_func_list(
        place_, block_, &op_func_nodes, global_scope_, create_local_scope_);
    is_build_ = true;
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
  }
  // NOTE: Because feed_tensor will be GC after
  // paddle::framework::build_op_func_list, so we should
  // call FeedInput again.
  if (prepare_feed) {
    FeedInput();
  }
}

interpreter::CostInfo InterpreterCore::DryRun(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors) {
  global_scope_->SetLocalScope(local_scope_);
  Prepare(feed_names, feed_tensors, true);
  interpreter::CostInfo cost_info;
  {
    interpreter::ProfilerGuard(place_, &cost_info);
    ExecuteInstructionList(vec_instruction_);
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
  }

  if (create_local_scope_) {
    ClearLoDTensorArrayInLocalScope();
  }

  return cost_info;
}

void InterpreterCore::SetFeedVarsInplaceSkip(
    const std::vector<std::string>& feed_names) {
  for (auto& feed_name : feed_names) {
    global_scope_->SetVarSikpInplace(feed_name, true);
  }
}

}  // namespace framework
}  // namespace paddle
