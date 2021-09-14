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

InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const ProgramDesc& main_prog,
                                 VariableScope* global_scope,
                                 const std::vector<std::string>& feed_names,
                                 const std::vector<std::string>& fetch_names)
    : place_(place),
      main_program_(main_prog),
      global_scope_(global_scope),
      stream_analyzer_(place) {
  is_build_ = false;

  feed_names_ = feed_names;

  std::vector<WorkQueueOptions> group_options(2);
  group_options[0].num_threads = 1;
  group_options[0].track_task = true;
  group_options[1].num_threads = 4;
  group_options[1].track_task = true;
  group_thread_pool_ = CreateWorkQueueGroup(group_options);

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
    ExecuteInstructionList(vec_instruction_);
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
        stream_analyzer_.ParseDeviceContext(vec_func_list_[i], *op_base);
    temp_inst.kernel_func_.compute_func_ = vec_func_list_[i].kernel_func_;
    temp_inst.kernel_func_.operator_base_ = op_base;
    temp_inst.input_index_ = vec_func_list_[i].input_index;
    temp_inst.output_index_ = vec_func_list_[i].output_index;
    temp_inst.type_ = vec_func_list_[i].type_;

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

    stream_analyzer_.Schedule(vec_func_list_, filter_next, i,
                              &vec_instruction_);

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

AtomicVectorSizeT InterpreterCore::PrepareAtomicDeps() {
  AtomicVectorSizeT working_dependecy_count(dependecy_count_.size());
  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    working_dependecy_count[i] =
        std::make_unique<std::atomic<size_t>>(dependecy_count_[i]);
  }
  return std::move(working_dependecy_count);
}

AtomicVectorSizeT InterpreterCore::PrepareAtomicVarRef() {
  AtomicVectorSizeT working_var_ref(vec_meta_info_.size());

  for (size_t i = 0; i < vec_meta_info_.size(); ++i) {
    working_var_ref[i] =
        std::make_unique<std::atomic<size_t>>(vec_meta_info_[i].var_ref_count_);
  }
  return std::move(working_var_ref);
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr, bool is_dry_run) {
  auto working_dependecy_count = PrepareAtomicDeps();
  auto working_var_ref = PrepareAtomicVarRef();
  std::atomic<size_t> op_run_number{0};

  for (size_t i = 0; i < dependecy_count_.size(); ++i) {
    if (dependecy_count_[i] == 0) {
      if (vec_instr[i].type_ == OpFuncType::kQueueAsync) {
        group_thread_pool_->AddTask(
            static_cast<size_t>(OpFuncType::kQueueAsync), [&, i, is_dry_run]() {
              RunInstructionAsync(i, &working_dependecy_count, &working_var_ref,
                                  &op_run_number, is_dry_run);
            });
      } else {
        group_thread_pool_->AddTask(
            static_cast<size_t>(OpFuncType::kQueueSync), [&, i, is_dry_run]() {
              RunInstructionAsync(i, &working_dependecy_count, &working_var_ref,
                                  &op_run_number, is_dry_run);
            });
      }
    }
  }
  // TODO(Aurelius84): [ Why we need a while_loop to check op_run_number ? ]
  // Because two WorkQueue can't communicate with each other, it will lead that
  // even though we called WaitQueueEmpty(), it still can't guarantee all ops
  // are finished.
  group_thread_pool_->WaitQueueGroupEmpty();

  PADDLE_ENFORCE_EQ(
      op_run_number.load(), vec_instr.size(),
      platform::errors::Fatal(
          "Required op_run_number == %d, but received op_run_number = %d.",
          vec_instr.size(), op_run_number.load()));
}

void InterpreterCore::RunInstructionAsync(
    size_t instr_id, AtomicVectorSizeT* working_dependecy_count,
    AtomicVectorSizeT* working_var_ref, std::atomic<size_t>* op_run_number,
    bool is_dry_run) {
  VLOG(3) << "Start to run instr_id: " << instr_id;
  auto& instr_node = vec_instruction_[instr_id];
  event_manager_.WaitEvent(instr_node, place_);
  RunInstruction(instr_node);
  event_manager_.RecordEvent(instr_node, place_);
  op_run_number->fetch_add(1);
  VLOG(3) << "end to run instr_id: " << instr_id;

  if (is_dry_run) {
    dry_run_profiler_.ParseMemoryInfo(global_scope_->var_list);
  }

  // step4: update working_queue
  auto& next_instr = instr_node.next_instruction_.all_next_ops_;

  for (auto next_i : next_instr) {
    working_dependecy_count->at(next_i)->fetch_sub(1);
    if (working_dependecy_count->at(next_i)->load() == 0) {
      if (vec_instruction_[next_i].type_ == OpFuncType::kQueueAsync) {
        group_thread_pool_->AddTask(1, [=]() {
          RunInstructionAsync(next_i, working_dependecy_count, working_var_ref,
                              op_run_number, is_dry_run);
        });
      } else {
        group_thread_pool_->AddTask(0, [=]() {
          RunInstructionAsync(next_i, working_dependecy_count, working_var_ref,
                              op_run_number, is_dry_run);
        });
      }
    }
  }
  // GC infomation
  CheckGC(instr_id, instr_node.gc_check_var_list, working_var_ref);
}

void InterpreterCore::CheckGC(size_t instr_id,
                              const std::vector<size_t>& gc_check_list,
                              AtomicVectorSizeT* working_var_ref) {
  auto& var_scope = *global_scope_;

  for (auto var_id : gc_check_list) {
    working_var_ref->at(var_id)->fetch_sub(1);
    if (var_scope.vec_meta_info_[var_id].vardesc_ &&
        !var_scope.vec_meta_info_[var_id].vardesc_->Persistable() &&
        working_var_ref->at(var_id)->load() == 0) {
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
  ExecuteInstructionList(vec_instruction_, /*is_dry_run=*/true);
  platform::DeviceContextPool::Instance().Get(place_)->Wait();

  dry_run_profiler_.Pause();
  dry_run_profiler_.TotalCUDAAllocatedMemorySize(place_);
  return dry_run_profiler_.GetCostInfo();
}

}  // namespace framework
}  // namespace paddle
