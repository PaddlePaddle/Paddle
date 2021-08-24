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

namespace paddle {
namespace framework {

InterpreterCore::InterpreterCore(const platform::Place& place,
                                 const ProgramDesc& main_prog,
                                 VariableScope* global_scope,
                                 const std::vector<std::string>& feed_names,
                                 const std::vector<std::string>& fetch_names)
    : place_(place), main_program_(main_prog), global_scope_(global_scope) {
  is_build_ = false;
  feed_names_ = feed_names;
  fetch_names_ = fetch_names;
  // add feedop and fetchop to main_program

  // prune

  // optmize graph pass

  // convert to run graph
}

void InterpreterCore::Run(const std::vector<framework::Tensor>& feed_tensors,
                          std::vector<framework::Tensor>* fetch_tensors) {
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

  for (size_t i = 0; i < fetch_names_.size(); ++i) {
    auto it = global_scope_->name2id.find(fetch_names_[i]);
    assert(it != global_scope_->name2id.end());
    PADDLE_ENFORCE_NE(
        it, global_scope_->name2id.end(),
        platform::errors::NotFound(
            "Can't find (%d) the fetch var (%s) in scope", i, fetch_names_[i]));

    auto fetch_tensor =
        global_scope_->var_list[it->second]->GetMutable<framework::LoDTensor>();

    if (platform::is_gpu_place(fetch_tensor->place())) {
      Tensor out;
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(place_);
      dev_ctx->Wait();
      TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
      dev_ctx->Wait();
      fetch_tensors->push_back(out);
    } else {
      Tensor out;
      TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
      fetch_tensors->push_back(out);
    }
  }
}

void InterpreterCore::Convert() {
  input_var2op_info_.resize(global_scope_->var_list.size());

  vec_instruction_.reserve(vec_func_list_.size());
  dependecy_count_.resize(vec_func_list_.size());
  vec_meta_info_.resize(global_scope_->var_list.size());
  for (size_t i = 0; i < vec_func_list_.size(); ++i) {
    Instruction temp_inst;
    temp_inst.kernel_func_.compute_func_ = vec_func_list_[i].kernel_func_;
    temp_inst.kernel_func_.operator_base_ = op_list_[i];
    temp_inst.input_index_ = vec_func_list_[i].input_index;
    temp_inst.output_index_ = vec_func_list_[i].output_index;

    std::vector<size_t> gc_check_input_list;
    for (auto& item : vec_func_list_[i].input_index) {
      for (auto id : item.second) {
        input_var2op_info_[id].push_back(i);
        gc_check_input_list.push_back(id);
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
    vec_instruction_[i].next_instruction_.direct_run_ = filter_next;

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

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  Scope scope;

  instr_node->execution_ctx_.reset(new ExecutionContext(
      *op_base, scope, *dev_ctx, *instr_node->runtime_ctx_.get()));
}

void InterpreterCore::RunInstruction(const Instruction& instr_node) {
  static_cast<const framework::OperatorWithKernel*>(
      instr_node.kernel_func_.operator_base_)
      ->InferShape(instr_node.infershape_ctx_.get());

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
    RunInstruction(instr_node);

    auto& next_instr = instr_node.next_instruction_.direct_run_;
    ++run_op_number;

    for (auto next_i : next_instr) {
      --working_dependecy_count[next_i];
      if (working_dependecy_count[next_i] == 0) {
        working_queue.push(next_i);
      }
    }

    // GC infomation

    auto& gc_check_list = instr_node.gc_check_var_list;
    for (auto var_id : gc_check_list) {
      --working_var_ref[var_id].var_ref_count_;
    }
  }

  for (size_t i = 0; i < working_var_ref.size(); ++i) {
    if (working_var_ref[i].var_ref_count_ != 0) {
      std::cerr << " var ref is not zero " << i << std::endl;
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
    }
  }
}

void InterpreterCore::BuildOpFuncList(const platform::Place& place,
                                      const framework::ProgramDesc& pdesc,
                                      std::vector<OperatorBase*>* op_list,
                                      std::vector<OpFuncNode>* vec_func_list,
                                      VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& op : global_block.AllOps()) {
    VLOG(3) << op->Type();
    // << op->Type() << endl;

    auto& info = OpInfoMap::Instance().Get(op->Type());

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();
    AttributeMap op_attr_map = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }
    auto op_base =
        info.Creator()(op->Type(), inputs_names, outputs_names, op_attr_map);

    OpFuncNode op_func_node;

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

    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;
    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);
    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);
    auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op->Type());
    PADDLE_ENFORCE_NE(
        kernels_iter, all_op_kernels.end(),
        platform::errors::Unavailable(
            "There are no kernels which are registered in the %s operator.",
            op->Type()));

    OpKernelMap& kernels = kernels_iter->second;
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);
    Scope scope;
    auto exec_ctx =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);
    auto expected_kernel_key =
        dynamic_cast<const framework::OperatorWithKernel*>(op_base)
            ->GetExpectedKernelType(exec_ctx);

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

          VariableNameMap copy_in_map;
          auto x_iter = inputs_names.find(var_name_item.first);
          copy_in_map["X"] = {x_iter->second[i]};
          VariableNameMap copy_out_map;
          copy_out_map["Out"] = {new_var_name};
          AttributeMap attr_map;
          attr_map["dst_place_type"] =
              is_cpu_place(place) ? 0 : is_gpu_place(place) ? 1 : -1;

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

          auto& copy_info = OpInfoMap::Instance().Get("memcpy");
          auto copy_op = copy_info.Creator()("memcpy", copy_in_map,
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
          auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
          auto kernels_iter = all_op_kernels.find("memcpy");
          PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                            platform::errors::Unavailable(
                                "There are no kernels which are registered in "
                                "the memcpy operator."));

          OpKernelMap& kernels = kernels_iter->second;
          platform::DeviceContextPool& pool =
              platform::DeviceContextPool::Instance();
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
          op_list->push_back(copy_op);
          vec_func_list->push_back(copy_op_func_node);

          var_name_item.second[i] = v;
        }
      }
    }

    op_list->push_back(op_base);

    auto kernel_iter = kernels.find(expected_kernel_key);
    PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                      platform::errors::NotFound(
                          "Operator (%s) does not have kernel for %s.",
                          op->Type(), KernelTypeToString(expected_kernel_key)));

    op_func_node.kernel_func_ = OpKernelComputeFunc(kernel_iter->second);
    op_func_node.kernel_func_(exec_ctx);
    vec_func_list->push_back(op_func_node);
  }
}

}  // namespace framework
}  // namespace paddle
