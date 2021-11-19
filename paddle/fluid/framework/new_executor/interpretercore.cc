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
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/profiler.h"

PADDLE_DEFINE_EXPORTED_bool(new_executor_use_inplace, true,
                            "Use inplace in new executor");

DECLARE_bool(check_nan_inf);
DECLARE_bool(benchmark);

constexpr const char* kExceptionCaught = "ExceptionCaught";

namespace paddle {
namespace framework {
// NOTE(Aurelius84): Need a better strategy to determine it.
static constexpr size_t kHostNumThreads = 4;

InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const BlockDesc& block,
                                 VariableScope* global_scope)
    : place_(place),
      block_(block),
      global_scope_(global_scope),
      stream_analyzer_(place) {
  is_build_ = false;
  async_work_queue_.reset(
      new interpreter::AsyncWorkQueue(kHostNumThreads, &main_thread_blocker_));
  gc_.reset(new InterpreterCoreGarbageCollector());

  exception_notifier_ = main_thread_blocker_.RegisterEvent(
      kExceptionCaught, [this]() { return exception_holder_.IsCaught(); });

  // prune

  // optmize graph pass

  // convert to run graph
}

InterpreterCore::~InterpreterCore() {
  // cancle gc's thread
  gc_.reset(nullptr);

  async_work_queue_.reset(nullptr);
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors) {
  bool is_build = is_build_;
  Prepare(feed_names, feed_tensors, is_build);

  if (is_build) {
    ExecuteInstructionList(vec_instruction_);
  }

  // return Fetch Tensors
  auto* fetch_var = global_scope_->Var(interpreter::kFetchVarName);
  return *(fetch_var->GetMutable<framework::FetchList>());
}

paddle::framework::FetchList InterpreterCore::Run() {
  if (!is_build_) {
    paddle::framework::interpreter::build_variable_scope(block_, global_scope_);
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::build_op_func_list(
        place_, block_, &op_func_nodes, global_scope_);
    is_build_ = true;
    // convert vec func_list to graph
    Convert(&op_func_nodes);
  } else {
    ExecuteInstructionList(vec_instruction_);
  }

  // return Fetch Tensors
  auto* fetch_var = global_scope_->Var(interpreter::kFetchVarName);
  return *(fetch_var->GetMutable<framework::FetchList>());
}

void InterpreterCore::BuildOperatorDependences() {
  // analysis the dependences between ops, set the dependecy_count_ and Call
  // Schedule
  auto op_nums = vec_instruction_.size();
  dependecy_count_.resize(op_nums);
  auto op2downstream = interpreter::build_op_downstream_map(vec_instruction_);
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
    auto& instr = vec_instruction_.back();

    OpInOutInfo info;
    std::vector<size_t> gc_check_input_list;

    for (auto& item : op_func_node.input_index) {
      for (auto id : item.second) {
        if (id == kEmptyVarIndex) {
          continue;
        }
        input_var2op_info_.at(id).push_back(op_idx);
        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op_func_node.operator_base_.get());
        }
        auto* var_desc = global_scope_->VarDesc(id);
        if (var_desc) {
          if (info.IsInArgBufferNeeded(var_desc->Name())) {
            gc_check_input_list.push_back(id);
          }
        } else {
          gc_check_input_list.push_back(id);
        }
      }
    }
    std::sort(gc_check_input_list.begin(), gc_check_input_list.end());
    auto last =
        std::unique(gc_check_input_list.begin(), gc_check_input_list.end());
    gc_check_input_list.erase(last, gc_check_input_list.end());

    for (auto var_id : gc_check_input_list) {
      vec_meta_info[var_id].var_ref_count_++;
      instr.AddGCCheckVar(var_id);
      VLOG(4) << "clear " << global_scope_->GetNameById(var_id) << " after "
              << instr.OpBase()->Type();
    }
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    // checkout ouput
    for (auto& item : vec_instruction_[i].Outputs()) {
      for (auto id : item.second) {
        if (input_var2op_info_.at(id).size() == 0) {
          // output var not be used by any kernel
          vec_instruction_[i].AddGCCheckVar(id);
          VLOG(4) << "clear " << global_scope_->GetNameById(id) << " after "
                  << vec_instruction_[i].OpBase()->Type();
          vec_meta_info[id].var_ref_count_++;
        }
      }
    }
  }

  BuildOperatorDependences();

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    BuildAndCacheInstructionCtx(&vec_instruction_[i]);
  }

  BuildSkipShareLoDInfo();

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    gc_event_.emplace_back(vec_instruction_[i].DeviceContext().GetPlace(),
                           platform::GenerateDeviceEventFlag());
  }

  if (FLAGS_new_executor_use_inplace) {
    BuildInplace();
  }
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
  instr_node->ResetContext(ins_map, outs_map);
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
  VLOG(4) << "Start run" << place << " " << op->DebugStringEx(global_scope_);

  auto op_with_kernel = dynamic_cast<const framework::OperatorWithKernel*>(op);
  {
    platform::RecordEvent infershape_event("InferShape");
    // If it is OperatorBase, InferShape do nothing.
    if (op_with_kernel != nullptr)
      op_with_kernel->InferShape(instr_node.InnerInferShapeContext().get());
  }

  if (op_with_kernel != nullptr &&
      FLAGS_new_executor_use_inplace) {  // TODO(xiongkun03) Does operator
                                         // base support
                                         // inplace ?
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
    platform::RecordEvent compute_event("Compute");
    if (op_with_kernel == nullptr)
      instr_node.OpBase()->Run(*global_scope_->GetScope(), place_);
    else
      instr_node.KernelFunc()(*instr_node.InnerExecutionContext().get());
  }

  VLOG(4) << "End run" << place << " " << op->DebugStringEx(global_scope_);

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    instr_node.DeviceContext().Wait();
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaGetLastError());
    VLOG(4) << "Operator(" << op->Type()
            << "): context wait and get last error";
#endif
#if defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_CUDA_SUCCESS(hipGetLastError());
    VLOG(4) << "Operator(" << op->Type()
            << "): context wait and get last error";
#endif
  }

  // for debug nan/inf
  if (FLAGS_check_nan_inf) {
    VLOG(4) << "Check nan/inf";
    framework::details::CheckOpHasNanOrInf(
        *op, *global_scope_,
        place);  // TODO(xiongkun03) change it to inner scope.
  }
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr) {
  async_work_queue_->PrepareAtomicDeps(dependecy_count_);
  async_work_queue_->PrepareAtomicVarRef(global_scope_->VecMetaInfo());
  op_run_number_ = 0;

  exception_holder_.Clear();

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      async_work_queue_->AddTask(vec_instr.at(i).KernelType(),
                                 [&, i] { RunInstructionAsync(i); });
    }
  }

  auto event_id = main_thread_blocker_.WaitEvent();
  VLOG(3) << "event_id " << event_id;

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(4) << "Exception caught " << exception_holder_.Type();
    exception_holder_.ReThrow();
  }

  PADDLE_ENFORCE_EQ(
      op_run_number_.load(), vec_instr.size(),
      platform::errors::Fatal(
          "Required op_run_number == %d, but received op_run_number = %d.",
          vec_instr.size(), op_run_number_.load()));
}

void InterpreterCore::RunNextInstructions(
    const Instruction& instr, std::queue<size_t>* reserved_next_ops) {
  auto& next_instr = instr.NextInstructions();
  auto& atomic_deps = async_work_queue_->AtomicDeps();
  auto IsReady = [&](size_t next_id) {
    return atomic_deps[next_id]->fetch_sub(1, std::memory_order_relaxed) == 1;
  };

  if (instr.KernelType() == OpFuncType::kQueueAsync) {
    // move all sync_ops into other threads
    for (auto next_id : next_instr.SyncRunIds()) {
      if (IsReady(next_id)) {
        async_work_queue_->AddTask(
            vec_instruction_[next_id].KernelType(),
            [&, next_id] { RunInstructionAsync(next_id); });
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
            [&, next_id] { RunInstructionAsync(next_id); });
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
            [&, next_id] { RunInstructionAsync(next_id); });
      }
    }
    if (first_op != 0) reserved_next_ops->push(first_op);
  }
}

void InterpreterCore::RunInstructionAsync(size_t instr_id) {
  std::queue<size_t> ready_ops;
  ready_ops.push(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.front();
    ready_ops.pop();
    auto& instr_node = vec_instruction_.at(instr_id);
    auto* op = instr_node.OpBase();
    platform::RecordEvent instruction_event(op->Type());
    interpreter::WaitEvent(instr_node, place_);

    try {
      RunInstruction(instr_node);
      // GC infomation
      CheckGC(instr_node);
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

    interpreter::RecordEvent(instr_node, place_);
    op_run_number_.fetch_add(1, std::memory_order_relaxed);

    RunNextInstructions(instr_node, &ready_ops);
  }
}

void InterpreterCore::CheckGC(const Instruction& instr) {
  size_t instr_id = instr.Id();
  auto& var_scope = *global_scope_;
  auto& atomic_var_ref = async_work_queue_->AtomicVarRef();

  for (auto var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC " << global_scope_->GetNameById(var_id) << " "
            << var_scope.VarDesc(var_id);

    bool is_ready =
        atomic_var_ref[var_id]->fetch_sub(1, std::memory_order_relaxed) == 1;
    // ignore all persistable var while GC
    if (var_scope.VarDesc(var_id) && var_scope.VarDesc(var_id)->Persistable()) {
      continue;
    }
    if (is_ready) {
      VLOG(6) << "Async delete variable with name : "
              << var_scope.GetNameById(var_id);
      gc_->Add(var_scope.Var(var_id), gc_event_.at(instr_id),
               &instr.DeviceContext());
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
      PADDLE_ENFORCE_NOT_NULL(feed_var, platform::errors::NotFound(
                                            "feed_var shall not be nullptr."));

      auto feed_tensor = feed_var->GetMutable<framework::LoDTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
      feed_tensor->set_lod(feed_tensors[i].lod());
    }
  };

  if (!is_build_) {
    paddle::framework::interpreter::build_variable_scope(block_, global_scope_);
    FeedInput();
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::build_op_func_list(
        place_, block_, &op_func_nodes, global_scope_);
    is_build_ = true;
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
  Prepare(feed_names, feed_tensors, true);
  interpreter::CostInfo cost_info;
  {
    interpreter::ProfilerGuard(place_, &cost_info);
    ExecuteInstructionList(vec_instruction_);
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
  }

  return cost_info;
}

}  // namespace framework
}  // namespace paddle
