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

#if !defined(_WIN32)
#include <sched.h>
#else
#include <windows.h>
#endif  // !_WIN32

#include "paddle/fluid/framework/new_executor/interpretercore.h"

#include <unordered_set>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/new_executor/interpretercore_gc_helper.h"

namespace paddle {
namespace framework {

static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";
namespace {
std::string GetMemcpyType(const platform::Place& src_place,
                          const platform::Place& dst_place) {
  PADDLE_ENFORCE_EQ(platform::is_same_place(src_place, dst_place), false,
                    platform::errors::PreconditionNotMet(
                        "Required src_place shall be different with dst_place, "
                        "but received same place: %s",
                        src_place));
  if (platform::is_gpu_place(dst_place)) {
    return kMemcpyH2D;
  } else if (platform::is_gpu_place(src_place)) {
    return kMemcpyD2H;
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Not support Memcpy typ : %s -> %s", src_place, dst_place));
  }
}

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
      h2d_ctx_pool_({place}),
      fetch_context_pool_({place}) {
  is_build_ = false;

  garbages_.reset(new GarbageQueue());
  max_memory_size_ = static_cast<size_t>(GetEagerDeletionThreshold());
  cur_memory_size_ = 0;
  gc_queue_ = CreateSingleThreadedWorkQueue();

  feed_names_ = feed_names;

  // Step1: add feedop and fetchop to main_program
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

  // prune

  // optmize graph pass

  // convert to run graph
}

paddle::framework::FetchList InterpreterCore::Run(
    const std::vector<framework::Tensor>& feed_tensors) {
  if (is_build_ == false) {
    BuildVariableScope(main_program_, global_scope_);
  }
  for (size_t i = 0; i < feed_names_.size(); ++i) {
    auto it = global_scope_->name2id.find(feed_names_[i]);
    assert(it != global_scope_->name2id.end());

    auto feed_tensor =
        global_scope_->var_list[it->second]->GetMutable<framework::LoDTensor>();
    feed_tensor->ShareDataWith(feed_tensors[i]);
  }

  if (is_build_ == false) {
    BuildOpFuncList(place_, main_program_, &op_list_, &vec_func_list_,
                    global_scope_);
    is_build_ = true;
    // convert vec func_list to graph
    Convert();
  } else {
    ExecuteInstructionList(vec_instruction_, *global_scope_, place_);
  }

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
    gc_event_.emplace_back(place_, platform::GenerateDeviceEventFlag());

    std::vector<size_t> vec_temp;
    for (auto& item : vec_instruction_[i].output_index_) {
      for (auto id : item.second) {
        vec_temp = MergeVector(vec_temp, input_var2op_info_[id]);
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

    for (auto inst_id : filter_next) {
      dependecy_count_[inst_id]++;
    }
    vec_instruction_[i].next_instruction_.all_next_ops_ =
        std::move(filter_next);
  }

  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    BuildInstructionCtx(&vec_instruction_[i], *global_scope_, place_);
  }
}

void InterpreterCore::BuildInstructionCtx(Instruction* instr_node,
                                          const VariableScope& var_scope,
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

  instr_node->infershape_ctx_.reset(
      new RuntimeInferShapeContext(*op_base, *instr_node->runtime_ctx_.get()));

  auto* dev_ctx = instr_node->dev_ctx_;
  if (instr_node->kernel_func_.operator_base_->Type() == "fetch_v2") {
    dev_ctx = fetch_context_pool_.Get(place);
  }
  Scope scope;

  instr_node->execution_ctx_.reset(new ExecutionContext(
      *op_base, scope, *dev_ctx, *instr_node->runtime_ctx_.get()));
}

void InterpreterCore::RunInstruction(const Instruction& instr_node) {
  static_cast<const framework::OperatorWithKernel*>(
      instr_node.kernel_func_.operator_base_)
      ->InferShape(instr_node.infershape_ctx_.get());

  if (instr_node.kernel_func_.operator_base_->Type() == "fetch_v2") {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place_);
    dev_ctx->Wait();  // TODO(wanghuancoder)
  }

  instr_node.kernel_func_.compute_func_(*instr_node.execution_ctx_.get());
}

void InterpreterCore::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr, const VariableScope& var_scope,
    const platform::Place& place) {
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

  fetch_context_pool_.Get(place)->Wait();

  for (size_t i = 0; i < working_var_ref.size(); ++i) {
    if (working_var_ref[i].var_ref_count_ != 0) {
      std::cerr << " var ref is not zero " << i << std::endl;
    }
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
      Variable* var = var_scope.var_list[var_id];
      if (var->IsType<LoDTensor>()) {
        garbages_->emplace_back(
            var->GetMutable<LoDTensor>()->MoveMemoryHolder());
        if (garbages_->back()) {
          cur_memory_size_ += garbages_->back()->size();
        }
      } else if (var->IsType<SelectedRows>()) {
        garbages_->emplace_back(var->GetMutable<SelectedRows>()
                                    ->mutable_value()
                                    ->MoveMemoryHolder());
        if (garbages_->back()) {
          cur_memory_size_ += garbages_->back()->size();
        }
      } else if (var->IsType<LoDTensorArray>()) {
        auto* tensor_arr = var->GetMutable<LoDTensorArray>();
        for (auto& t : *tensor_arr) {
          garbages_->emplace_back(t.MoveMemoryHolder());
          if (garbages_->back()) {
            cur_memory_size_ += garbages_->back()->size();
          }
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The variable(%s) is not supported in eager deletion.",
            framework::ToTypeName(var->Type())));
      }
    }
  }

  if (!garbages_->empty()) {
    if (max_memory_size_ <= 1) {
      gc_event_[instr_id].Record(
          platform::DeviceContextPool::Instance().Get(place));
      gc_queue_->AddTask(
          [ container = garbages_.release(), event = &gc_event_[instr_id] ]() {
            while (!event->Query()) {
#if defined(_WIN32)
              SleepEx(50, FALSE);
#else
              sched_yield();
#endif
              continue;
            }
            delete container;
          });
      garbages_.reset(new GarbageQueue());
    } else if (cur_memory_size_ >= max_memory_size_) {
      gc_event_[instr_id].Record(
          platform::DeviceContextPool::Instance().Get(place));
      gc_queue_->AddTask(
          [ container = garbages_.release(), event = &gc_event_[instr_id] ]() {
            while (!event->Query()) {
#if defined(_WIN32)
              SleepEx(50, FALSE);
#else
              sched_yield();
#endif
              continue;
            }
            delete container;
          });
      garbages_.reset(new GarbageQueue());
      cur_memory_size_ = 0;
    }
  }
}

std::vector<size_t> InterpreterCore::MergeVector(
    const std::vector<size_t>& first, const std::vector<size_t>& second) {
  std::vector<size_t> out(first.size() + second.size());
  std::merge(first.begin(), first.end(), second.begin(), second.end(),
             out.begin());

  std::vector<size_t>::iterator it;
  it = std::unique(out.begin(), out.end());

  out.resize(std::distance(out.begin(), it));

  return out;
}

void InterpreterCore::BuildVariableScope(const framework::ProgramDesc& pdesc,
                                         VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
      auto v = new Variable();
      InitializeVariable(v, var->GetType());
      var_scope->var_list.push_back(v);

      VariableMetaInfo info;
      info.var_ref_count_ = 0;
      info.vardesc_ = var;
      var_scope->vec_meta_info_.push_back(info);
    }
  }
}

void InterpreterCore::BuildOpFuncList(const platform::Place& place,
                                      const framework::ProgramDesc& pdesc,
                                      std::vector<OperatorBase*>* op_list,
                                      std::vector<OpFuncNode>* vec_func_list,
                                      VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);
  auto& all_op_kernels = OperatorWithKernel::AllOpKernels();

  std::vector<OperatorBase*> ops;
  for (auto& op : global_block.AllOps()) {
    VLOG(3) << "Build OpFuncNode from : " << op->Type();

    auto& info = OpInfoMap::Instance().Get(op->Type());

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();
    AttributeMap op_attr_map = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }
    // step 1. Prepare VariableValueMap of input/output
    auto op_base =
        info.Creator()(op->Type(), inputs_names, outputs_names, op_attr_map);
    ops.push_back(op_base);
  }

  auto unused_var_map = get_unused_vars(global_block, ops);

  size_t ops_index = 0;
  for (auto& op : global_block.AllOps()) {
    VLOG(3) << op->Type();
    // << op->Type() << endl;

    auto op_base = ops[ops_index++];

    auto inputs_names = op->Inputs();
    auto outputs_names = op->Outputs();

    VariableValueMap ins_map;
    std::map<std::string, std::vector<int>> ins_name2id;
    for (auto& var_name_item : inputs_names) {
      std::vector<Variable*> input_vars;
      std::vector<int> vec_ids;
      input_vars.reserve(var_name_item.second.size());
      for (auto& var_name : var_name_item.second) {
        auto it = var_scope->name2id.find(var_name);
        assert(it != var_scope->name2id.end());
        input_vars.push_back(var_scope->var_list[it->second]);
        vec_ids.push_back(it->second);
      }
      ins_map[var_name_item.first] = input_vars;
      ins_name2id[var_name_item.first] = vec_ids;
    }

    VariableValueMap outs_map;
    std::map<std::string, std::vector<int>> outs_name2id;
    for (auto& var_name_item : outputs_names) {
      std::vector<Variable*> output_vars;
      std::vector<int> vec_ids;
      output_vars.reserve(var_name_item.second.size());
      for (auto& var_name : var_name_item.second) {
        auto it = var_scope->name2id.find(var_name);
        assert(it != var_scope->name2id.end());
        output_vars.push_back(var_scope->var_list[it->second]);
        vec_ids.push_back(it->second);
      }
      outs_map[var_name_item.first] = output_vars;
      outs_name2id[var_name_item.first] = vec_ids;
    }

    OpFuncNode op_func_node;
    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;
    // step 2: construct RuntimeContext and analysis KernelType
    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);
    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);
    auto kernels_iter = all_op_kernels.find(op->Type());
    PADDLE_ENFORCE_NE(
        kernels_iter, all_op_kernels.end(),
        platform::errors::Unavailable(
            "There are no kernels which are registered in the %s operator.",
            op->Type()));

    OpKernelMap& kernels = kernels_iter->second;

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);
    Scope scope;
    auto expected_kernel_key =
        dynamic_cast<const framework::OperatorWithKernel*>(op_base)
            ->GetExpectedKernelType(
                ExecutionContext(*op_base, scope, *dev_ctx, runtime_context));

    // consider device_guard context
    bool need_change_place =
        (op_base->HasAttr("op_device") &&
         (op_base->Attr<std::string>("op_device").length() > 0));
    if (need_change_place) {
      auto& op_device = op_base->Attr<std::string>("op_device");
      if (op_device == "cpu" || platform::is_cpu_place(place)) {
        VLOG(3) << "Switch into CPUPlace by device_guard.";
        expected_kernel_key.place_ = platform::CPUPlace();
      } else if (op_device.find("gpu") != std::string::npos &&
                 platform::is_gpu_place(place)) {
        VLOG(3) << "Switch into " << place << " by device_guard.";
        expected_kernel_key.place_ = place;
      } else {
        PADDLE_THROW(
            platform::errors::Fatal("Unsupported current place %s", op_device));
      }
    }
    VLOG(3) << "expected_kernel_key : " << expected_kernel_key;

    // step 3. Insert memcpy_op if needed
    VariableValueMap& ins_map_temp = runtime_context.inputs;
    for (auto& var_name_item : ins_map_temp) {
      for (size_t i = 0; i < var_name_item.second.size(); ++i) {
        auto var = var_name_item.second[i];
        auto tensor_in = static_cast<const Tensor*>(&(var->Get<LoDTensor>()));
        if (!tensor_in->IsInitialized()) {
          continue;
        }
        auto kernel_type_for_var =
            static_cast<const framework::OperatorWithKernel*>(op_base)
                ->GetKernelTypeForVar(var_name_item.first, *tensor_in,
                                      expected_kernel_key);
        if (!platform::is_same_place(kernel_type_for_var.place_,
                                     expected_kernel_key.place_)) {
          // need trans place
          // 1. add var in scope
          // 2. add copy op
          std::string new_var_name =
              "temp_1" + std::to_string(var_scope->var_list.size() + 1);
          auto v = new Variable();
          v->GetMutable<LoDTensor>();
          var_scope->name2id[new_var_name] = var_scope->var_list.size();
          var_scope->var_list.push_back(v);

          VariableMetaInfo info;
          info.var_ref_count_ = 0;
          info.vardesc_ = nullptr;
          var_scope->vec_meta_info_.push_back(info);

          VariableNameMap copy_in_map;
          auto x_iter = inputs_names.find(var_name_item.first);
          copy_in_map["X"] = {x_iter->second[i]};
          VariableNameMap copy_out_map;
          copy_out_map["Out"] = {new_var_name};
          AttributeMap attr_map;
          attr_map["dst_place_type"] =
              is_cpu_place(expected_kernel_key.place_)
                  ? 0
                  : is_gpu_place(expected_kernel_key.place_) ? 1 : -1;

          std::map<std::string, std::vector<int>> copy_ins_name2id;
          copy_ins_name2id["X"] = ins_name2id[var_name_item.first];
          std::map<std::string, std::vector<int>> copy_out_name2id;
          copy_out_name2id["Out"] = {var_scope->name2id[new_var_name]};

          op_func_node.input_index[var_name_item.first][i] =
              var_scope->name2id[new_var_name];

          VariableValueMap copy_ins_value_map;
          copy_ins_value_map["X"] = {var};
          VariableValueMap copy_outs_value_map;
          copy_outs_value_map["Out"] = {v};

          // memcpy_d2h, memcpy_h2d
          auto memcpy_op_type = GetMemcpyType(kernel_type_for_var.place_,
                                              expected_kernel_key.place_);
          VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).",
                                     memcpy_op_type, x_iter->second[i],
                                     kernel_type_for_var.place_, new_var_name,
                                     expected_kernel_key.place_);
          auto& copy_info = OpInfoMap::Instance().Get(memcpy_op_type);
          auto copy_op = copy_info.Creator()(memcpy_op_type, copy_in_map,
                                             copy_out_map, attr_map);
          OpFuncNode copy_op_func_node;
          copy_op_func_node.input_index = copy_ins_name2id;
          copy_op_func_node.output_index = copy_out_name2id;

          RuntimeContext copy_runtime_context({}, {});
          copy_runtime_context.inputs.swap(copy_ins_value_map);
          copy_runtime_context.outputs.swap(copy_outs_value_map);
          RuntimeInferShapeContext copy_infer_shape_ctx(*copy_op,
                                                        copy_runtime_context);
          static_cast<const framework::OperatorWithKernel*>(copy_op)
              ->InferShape(&copy_infer_shape_ctx);

          auto kernels_iter = all_op_kernels.find(memcpy_op_type);
          PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                            platform::errors::Unavailable(
                                "There are no kernels which are registered in "
                                "the memcpy operator."));

          OpKernelMap& kernels = kernels_iter->second;
          auto* dev_ctx = pool.Get(place);
          Scope scope;
          auto copy_exec_ctx =
              ExecutionContext(*copy_op, scope, *dev_ctx, copy_runtime_context);
          auto expected_kernel_key =
              dynamic_cast<const framework::OperatorWithKernel*>(copy_op)
                  ->GetExpectedKernelType(copy_exec_ctx);
          auto kernel_iter = kernels.find(expected_kernel_key);
          copy_op_func_node.kernel_func_ =
              OpKernelComputeFunc(kernel_iter->second);
          copy_op_func_node.kernel_func_(copy_exec_ctx);
          VLOG(3) << "Run " << memcpy_op_type << " done.";
          copy_op_func_node.type_ = OpFuncType::kQueueAsync;
          copy_op_func_node.dev_ctx_ = dev_ctx;
          op_list->push_back(copy_op);
          vec_func_list->push_back(copy_op_func_node);

          var_name_item.second[i] = v;
        }
      }
    }
    // step 4. Run op kernel
    op_list->push_back(op_base);
    VLOG(3) << op_base->Type()
            << " : expected_kernel_key : " << expected_kernel_key;

    if (platform::is_gpu_place(expected_kernel_key.place_)) {
      op_func_node.type_ = OpFuncType::kQueueAsync;
    } else if (platform::is_cpu_place(expected_kernel_key.place_)) {
      op_func_node.type_ = OpFuncType::kQueueSync;
    } else {
      PADDLE_THROW(platform::errors::Fatal("Unsupported current place %s",
                                           expected_kernel_key.place_));
    }

    if (!(expected_kernel_key.place_ == dev_ctx->GetPlace())) {
      dev_ctx = pool.Get(expected_kernel_key.place_);
    }
    op_func_node.dev_ctx_ = dev_ctx;

    auto exec_ctx =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);

    auto kernel_iter = kernels.find(expected_kernel_key);
    PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                      platform::errors::NotFound(
                          "Operator (%s) does not have kernel for %s.",
                          op->Type(), KernelTypeToString(expected_kernel_key)));

    op_func_node.kernel_func_ = OpKernelComputeFunc(kernel_iter->second);
    op_func_node.kernel_func_(exec_ctx);
    vec_func_list->push_back(op_func_node);

    // gc---------------------------------------------------------------------------
    auto iter = unused_var_map.find(op_base);
    if (iter == unused_var_map.end()) {
      continue;
    }

    auto& delete_vars = iter->second;
    std::deque<std::shared_ptr<memory::Allocation>>* garbages =
        new std::deque<std::shared_ptr<memory::Allocation>>();

    for (auto& var_name : delete_vars) {
      auto it = var_scope->name2id.find(var_name);
      assert(it != var_scope->name2id.end());
      auto* var = var_scope->var_list[it->second];
      if (var == nullptr) {
        continue;
      }

      VLOG(2) << "Erase variable " << var_name;
      if (var->IsType<LoDTensor>()) {
        garbages->emplace_back(
            var->GetMutable<LoDTensor>()->MoveMemoryHolder());
      } else if (var->IsType<SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<SelectedRows>()
                                   ->mutable_value()
                                   ->MoveMemoryHolder());
      } else if (var->IsType<LoDTensorArray>()) {
        auto* lod_tensor_arr = var->GetMutable<LoDTensorArray>();
        for (auto& t : *lod_tensor_arr) {
          garbages->emplace_back(t.MoveMemoryHolder());
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Type %s of variable %s is not supported eager deletion.",
            framework::ToTypeName(var->Type()), var_name));
      }
    }

    delete garbages;  // free mem

    VLOG(3) << "run " << op_base->Type() << " done.";
  }
}

platform::DeviceContext* InterpreterCore::ParseDeviceContextForInstruction(
    const OpFuncNode& op_func_node, const OperatorBase& op_base) {
  auto& op_type = op_base.Type();
  auto* dev_ctx = op_func_node.dev_ctx_;
  if (op_type == kMemcpyH2D) {
    VLOG(3) << "Get dev_ctx from d2h_context_pool_";
    dev_ctx = d2h_ctx_pool_.Get(place_);
  } else if (op_type == kMemcpyD2H) {
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
