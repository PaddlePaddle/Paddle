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
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"

#include <unordered_set>

namespace paddle {
namespace framework {

namespace {

/*
 * Parse the var_ids that need to be associated with an event.
 * The caller should guarantee front_op and back_op satisfy the
 * following conditions:
 *   1. kQueueAsync -> kQueueAsync
 *   2. kQueueAsync -> kQueueSync
 *
 * For example: matmul(gpu) -> out_var -> memcpy_d2h
 * out_var should be associated with an event.
 */
std::vector<size_t> ParseEventVarIds(const Instruction& cur_instr,
                                     const Instruction& next_instr) {
  std::unordered_set<size_t> unique_var_ids;
  for (auto& item : cur_instr.output_index_) {
    unique_var_ids.insert(item.second.begin(), item.second.end());
  }

  std::vector<size_t> new_event_var_ids;
  for (auto& item : next_instr.input_index_) {
    for (auto var_id : item.second) {
      if (unique_var_ids.count(var_id) > 0) {
        new_event_var_ids.push_back(var_id);
      }
    }
  }
  return new_event_var_ids;
}

void AssociateInputWithEvents(
    const platform::Place& place, const std::vector<size_t>& new_event_var_id,
    Instruction* next_instr,
    std::map<size_t, std::shared_ptr<platform::DeviceEvent>>* var_id2event,
    bool is_sync) {
  for (auto var_id : new_event_var_id) {
    if (var_id2event->count(var_id) == 0) {
      auto device_event = std::make_shared<platform::DeviceEvent>(
          place, platform::GenerateDeviceEventFlag());
      var_id2event->emplace(var_id, std::move(device_event));
    }
    // Add events for next_instr.inputs
    next_instr->intput_events_.emplace_back(var_id, var_id2event->at(var_id),
                                            is_sync);
  }
}

void ParseDirectAndEventRunOps(
    const platform::Place& place, const std::vector<OpFuncNode>& op_func_nodes,
    const std::vector<size_t>& downstream_ops, size_t op_index,
    std::map<size_t, std::shared_ptr<platform::DeviceEvent>>* var_id2event,
    std::vector<Instruction>* instructions) {
  auto& op_func_type = op_func_nodes[op_index].type_;
  auto& cur_instr = instructions->at(op_index);
  auto& next_instruction = cur_instr.next_instruction_;

  if (op_func_type == OpFuncType::kQueueSync) {
    // all downstream ops of kQueueSync can directly run, such as CPU -> Any
    next_instruction.direct_run_ = downstream_ops;
  } else {  // kQueueAsync
    std::vector<size_t> event_var_ids;
    for (auto next_op_id : downstream_ops) {
      auto& next_instr = instructions->at(next_op_id);
      // case 1: GPU -> GPU(same stream)
      if (cur_instr.dev_ctx_ == next_instr.dev_ctx_) {
        next_instruction.direct_run_.emplace_back(next_op_id);
        continue;
      }
      // Always insert events between different stream
      auto new_event_var_ids = ParseEventVarIds(cur_instr, next_instr);
      event_var_ids.insert(event_var_ids.end(), new_event_var_ids.begin(),
                           new_event_var_ids.end());

      bool is_sync =
          (op_func_nodes[next_op_id].type_ == OpFuncType::kQueueSync);
      AssociateInputWithEvents(place, new_event_var_ids, &next_instr,
                               var_id2event, is_sync);

      if (is_sync) {  // GPU -> CPU
        next_instruction.synchronize_run_.emplace_back(next_op_id);
      } else {  // GPU -> GPU(different stream)
        next_instruction.event_wait_run_.emplace_back(next_op_id);
      }
    }
    // Create events for these cross-stream vars
    VLOG(3) << cur_instr.kernel_func_.operator_base_->Type()
            << " event_var_ids.size: " << event_var_ids.size();
    for (auto var_id : event_var_ids) {
      cur_instr.output_events_.emplace_back(var_id, var_id2event->at(var_id),
                                            false /*not used*/);
    }
  }
}
}  // namespace

InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const ProgramDesc& main_prog,
                                 VariableScope* global_scope,
                                 const std::vector<std::string>& feed_names,
                                 const std::vector<std::string>& fetch_names)
    : place_(place),
      main_program_(main_prog),
      global_scope_(global_scope),
      d2h_ctx_pool_({place}),
      h2d_ctx_pool_({place}) {
  is_build_ = false;

  feed_names_ = feed_names;

  // Step1: add feedop and fetchop to main_program
  AddFetch(fetch_names);

  // prune

  // optmize graph pass

  // convert to run graph
}

void InterpreterCore::AddFetch(const std::vector<std::string>& fetch_names) {
  auto* fetch_holder = main_program_.MutableBlock(0)->Var("fetch_vars");
  fetch_holder->SetType(proto::VarType::FETCH_LIST);
  fetch_holder->SetPersistable(true);

  int i = 0;
  for (auto& fetch_name : fetch_names) {
    // append fetch op
    auto* op = main_program_.MutableBlock(0)->AppendOp();
    op->SetType("fetch_v2");
    op->SetInput("X", {fetch_name});
    op->SetOutput("Out", {"fetch_vars"});
    op->SetAttr("col", {static_cast<int>(i)});
    op->CheckAttrs();
    i++;
  }
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<framework::Tensor>& feed_tensors) {
  auto FeedInput = [&] {
    for (size_t i = 0; i < feed_names_.size(); ++i) {
      auto it = global_scope_->name2id.find(feed_names_[i]);
      assert(it != global_scope_->name2id.end());

      auto feed_tensor = global_scope_->var_list[it->second]
                             ->GetMutable<framework::LoDTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
    }
  };

  if (is_build_ == false) {
    paddle::framework::interpretercore::build_variable_scope(main_program_,
                                                             global_scope_);
    FeedInput();
    paddle::framework::interpretercore::build_op_func_list(
        place_, main_program_, &op_list_, &vec_func_list_, global_scope_);
    is_build_ = true;
    // convert vec func_list to graph
    Convert();
  } else {
    FeedInput();
    ExecuteInstructionList(vec_instruction_, *global_scope_, place_);
  }

  // return Fetch Tensors
  return *(global_scope_->var_list[global_scope_->name2id["fetch_vars"]]
               ->GetMutable<framework::FetchList>());
}

void InterpreterCore::Convert() {
  input_var2op_info_.resize(global_scope_->var_list.size());

  vec_instruction_.reserve(vec_func_list_.size());
  dependecy_count_.resize(vec_func_list_.size());
  vec_meta_info_.resize(global_scope_->var_list.size());
  for (size_t i = 0; i < vec_func_list_.size(); ++i) {
    Instruction temp_inst;
    auto* op_base = op_list_[i];
    temp_inst.dev_ctx_ =
        ParseDeviceContextForInstruction(vec_func_list_[i], *op_base);
    temp_inst.kernel_func_.compute_func_ = vec_func_list_[i].kernel_func_;
    temp_inst.kernel_func_.operator_base_ = op_base;
    temp_inst.input_index_ = vec_func_list_[i].input_index;
    temp_inst.output_index_ = vec_func_list_[i].output_index;

    OpInOutInfo info;

    std::vector<size_t> gc_check_input_list;
    for (auto& item : vec_func_list_[i].input_index) {
      for (auto id : item.second) {
        input_var2op_info_[id].push_back(i);
        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op_list_[i]);
        }
        if (global_scope_->vec_meta_info_[id].vardesc_) {
          if (info.IsInArgBufferNeeded(
                  global_scope_->vec_meta_info_[id].vardesc_->Name())) {
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
      vec_meta_info_[var_id].var_ref_count_++;
    }

    temp_inst.gc_check_var_list.swap(gc_check_input_list);

    vec_instruction_.push_back(temp_inst);
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    // checkout ouput
    for (auto& item : vec_instruction_[i].output_index_) {
      for (auto id : item.second) {
        if (input_var2op_info_[id].size() == 0) {
          // output var not be used by any kernel
          vec_instruction_[i].gc_check_var_list.push_back(id);
          vec_meta_info_[id].var_ref_count_++;
        }
      }
    }
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    std::vector<size_t> vec_temp;
    for (auto& item : vec_instruction_[i].output_index_) {
      for (auto id : item.second) {
        vec_temp =
            interpretercore::merge_vector(vec_temp, input_var2op_info_[id]);
      }
    }

    // In Program, op order is a very import information.
    // Op can noly add op after it as next as next ops.
    std::vector<size_t> filter_next;
    filter_next.reserve(vec_temp.size());
    for (auto item : vec_temp) {
      if (item > i) {
        filter_next.push_back(item);
      }
    }

    ParseDirectAndEventRunOps(place_, vec_func_list_, filter_next, i,
                              &var_id2event_, &vec_instruction_);

    for (auto inst_id : filter_next) {
      dependecy_count_[inst_id]++;
    }
    vec_instruction_[i].next_instruction_.all_next_ops_ =
        std::move(filter_next);
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    BuildAndCacheInstructionCtx(&vec_instruction_[i], *global_scope_, place_);
  }

  BuildSkipShareLoDInfo();

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    gc_event_.emplace_back(vec_instruction_[i].execution_ctx_.get()->GetPlace(),
                           platform::GenerateDeviceEventFlag());
  }
}

void InterpreterCore::BuildAndCacheInstructionCtx(
    Instruction* instr_node, const VariableScope& var_scope,
    const platform::Place& place) {
  auto op_base = instr_node->kernel_func_.operator_base_;

  VariableValueMap ins_map;
  for (auto& var_name_item : instr_node->input_index_) {
    std::vector<Variable*> input_vars;

    input_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      input_vars.emplace_back(var_scope.var_list[id]);
    }
    ins_map.emplace(var_name_item.first, std::move(input_vars));
  }

  VariableValueMap outs_map;
  for (auto& var_name_item : instr_node->output_index_) {
    std::vector<Variable*> out_vars;

    out_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      out_vars.emplace_back(var_scope.var_list[id]);
    }
    outs_map.emplace(var_name_item.first, std::move(out_vars));
  }

  instr_node->runtime_ctx_.reset(new RuntimeContext({}, {}));
  instr_node->runtime_ctx_->inputs.swap(ins_map);
  instr_node->runtime_ctx_->outputs.swap(outs_map);

  instr_node->infershape_ctx_.reset(new InterpretercoreInferShapeContext(
      *op_base, *instr_node->runtime_ctx_.get()));

  auto* dev_ctx = instr_node->dev_ctx_;
  Scope scope;

  instr_node->execution_ctx_.reset(new ExecutionContext(
      *op_base, scope, *dev_ctx, *instr_node->runtime_ctx_.get()));
}

void InterpreterCore::BuildSkipShareLoDInfo() {
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    bool can_skip_lod = true;
    for (auto& input : vec_instruction_[i].runtime_ctx_.get()->inputs) {
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
    vec_instruction_[i].infershape_ctx_.get()->SetSkipLoD(can_skip_lod);
  }
}

void InterpreterCore::RunInstruction(const Instruction& instr_node) {
  VLOG(3) << "RunInstruction:  "
          << instr_node.kernel_func_.operator_base_->Type();

  static_cast<const framework::OperatorWithKernel*>(
      instr_node.kernel_func_.operator_base_)
      ->InferShape(instr_node.infershape_ctx_.get());

  instr_node.kernel_func_.compute_func_(*instr_node.execution_ctx_.get());
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr, const VariableScope& var_scope,
    const platform::Place& place, bool is_dry_run) {
  std::queue<size_t> working_queue;
  auto working_dependecy_count = dependecy_count_;
  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      working_queue.push(i);
    }
  }

  auto working_var_ref = vec_meta_info_;

  size_t run_op_number = 0;
  while (!working_queue.empty()) {
    auto instr_id = working_queue.front();
    working_queue.pop();
    auto& instr_node = vec_instr[instr_id];
    // step1 : stream_wait (non-block host) or sync (block host)
    StreamWaitEventOrSync(instr_node);
    // step2: run instruction
    RunInstruction(instr_node);
    ++run_op_number;

    if (is_dry_run) {
      dry_run_profiler_.ParseMemoryInfo(var_scope.var_list);
    }

    // step3: insert event for out_vars if needed
    RecordEventInstruction(instr_node, vec_func_list_[instr_id]);

    // step4: update working_queue
    auto& next_instr = instr_node.next_instruction_.all_next_ops_;

    for (auto next_i : next_instr) {
      --working_dependecy_count[next_i];
      if (working_dependecy_count[next_i] == 0) {
        working_queue.push(next_i);
      }
    }

    // GC infomation
    CheckGC(instr_id, instr_node.gc_check_var_list, var_scope, place,
            working_var_ref);
  }
}

void InterpreterCore::CheckGC(size_t instr_id,
                              const std::vector<size_t>& gc_check_list,
                              const VariableScope& var_scope,
                              const platform::Place& place,
                              std::vector<VariableMetaInfo>& working_var_ref) {
  for (auto var_id : gc_check_list) {
    --working_var_ref[var_id].var_ref_count_;
    if (var_scope.vec_meta_info_[var_id].vardesc_ &&
        !var_scope.vec_meta_info_[var_id].vardesc_->Persistable() &&
        working_var_ref[var_id].var_ref_count_ == 0) {
      gc_.Add(var_scope.var_list[var_id], gc_event_[instr_id],
              vec_instruction_[instr_id].dev_ctx_);
    }
  }
}

void InterpreterCore::DryRunPrepare(
    const std::vector<framework::Tensor>& feed_tensors) {
  auto FeedInput = [&] {
    for (size_t i = 0; i < feed_names_.size(); ++i) {
      auto it = global_scope_->name2id.find(feed_names_[i]);
      assert(it != global_scope_->name2id.end());

      auto feed_tensor = global_scope_->var_list[it->second]
                             ->GetMutable<framework::LoDTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
    }
  };

  if (is_build_ == false) {
    paddle::framework::interpretercore::build_variable_scope(main_program_,
                                                             global_scope_);
    FeedInput();
    paddle::framework::interpretercore::build_op_func_list(
        place_, main_program_, &op_list_, &vec_func_list_, global_scope_);
    is_build_ = true;
    // convert vec func_list to graph
    Convert();
  }
  // NOTE: Because feed_tensor will be GC after
  // paddle::framework::build_op_func_list, so we should
  // call
  // FeedInput again.
  FeedInput();
}

const CostInfo& InterpreterCore::DryRun(
    const std::vector<framework::Tensor>& feed_tensors) {
  DryRunPrepare(feed_tensors);
  // DryRun may be called many times.
  dry_run_profiler_.Reset();
  dry_run_profiler_.Start();
  ExecuteInstructionList(vec_instruction_, *global_scope_, place_,
                         /*is_dry_run=*/true);
  platform::DeviceContextPool::Instance().Get(place_)->Wait();

  dry_run_profiler_.Pause();
  dry_run_profiler_.TotalCUDAAllocatedMemorySize(place_);
  return dry_run_profiler_.GetCostInfo();
}

platform::DeviceContext* InterpreterCore::ParseDeviceContextForInstruction(
    const OpFuncNode& op_func_node, const OperatorBase& op_base) {
  auto& op_type = op_base.Type();
  auto* dev_ctx = op_func_node.dev_ctx_;
  if (op_type == interpretercore::kMemcpyH2D) {
    VLOG(3) << "Get dev_ctx from d2h_context_pool_";
    dev_ctx = d2h_ctx_pool_.Get(place_);
  } else if (op_type == interpretercore::kMemcpyD2H) {
    VLOG(3) << "Get dev_ctx from h2d_context_pool_";
    dev_ctx = h2d_ctx_pool_.Get(place_);
  }

  return dev_ctx;
}

void InterpreterCore::RecordEventInstruction(const Instruction& instruction,
                                             const OpFuncNode& op_func_node) {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place_)) return;

  for (auto& event : instruction.output_events_) {
    VLOG(3) << "Record event in out_var_id: " << event.var_id_;
    event.event_->Record(instruction.dev_ctx_);
  }
}

void InterpreterCore::WaitOrSync(const std::vector<EventInter>& events,
                                 const platform::DeviceContext* dev_ctx) {
  for (auto& event_iter : events) {
    if (event_iter.is_sync_) {
      VLOG(3) << "host sync wait in_var_id " << event_iter.var_id_;
      event_iter.event_->Wait(platform::kCPU, dev_ctx);
    } else {
      VLOG(3) << "stream async wait in_var_id " << event_iter.var_id_;
      event_iter.event_->Wait(platform::kCUDA, dev_ctx);
    }
  }
}

void InterpreterCore::StreamWaitEventOrSync(const Instruction& instruction) {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place_)) return;

  VLOG(3) << "Deal StreamWaitEventOrSync for "
          << instruction.kernel_func_.operator_base_->Type();
  auto* dev_ctx = instruction.dev_ctx_;

  WaitOrSync(instruction.intput_events_, dev_ctx);
}
}  // namespace framework
}  // namespace paddle
