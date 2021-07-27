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
#pragma once

#include <iostream>
#include <string>

#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/new_exec_util.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

// USE_OP(fill_constant);
// USE_OP(elementwise_add);

// using namespace std;

namespace paddle {
namespace framework {

using std::cerr;
using std::endl;

framework::ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();

  ProgramDesc program_desc(buffer);
  return program_desc;
}

struct KernelFunc {
  using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
  OpKernelComputeFunc compute_func_;
  OperatorBase* operator_base_;
};

struct VariableMetaInfo {
  int var_ref_count_;
};

struct VariableScope {
  std::vector<Variable*> var_list;
  std::map<std::string, int> name2id;
  std::vector<VariableMetaInfo> vec_meta_info_;
};

struct NextInstruction {
  std::vector<size_t> direct_run_;
};

struct EventInter {};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

struct EventRun {
  EventInter event_inter;
  std::vector<size_t> same_device_run_;
  std::vector<size_t> synchronized_run;
};

struct Instruction {
  KernelFunc kernel_func_;
  std::map<std::string, std::vector<int>> input_index_;
  std::map<std::string, std::vector<int>> output_index_;

  std::vector<size_t> gc_check_var_list;
  NextInstruction next_instruction_;
  std::vector<EventInter> vec_event_list_;
};

struct OpFuncNode {
  // int unsed;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

  using OpKernelFunc = std::function<void(const ExecutionContext&)>;
  OpKernelFunc kernel_func_;
};

int convert(const platform::Place& place) {
  if (is_cpu_place(place)) {
    return 0;
  }
  if (is_gpu_place(place)) {
    return 1;
  }

  return -1;
}

std::vector<size_t> merge_vec(const std::vector<size_t>& first,
                              const std::vector<size_t>& second) {
  std::vector<size_t> out(first.size() + second.size());
  std::merge(first.begin(), first.end(), second.begin(), second.end(),
             out.begin());

  std::vector<size_t>::iterator it;
  it = std::unique(out.begin(), out.end());

  out.resize(std::distance(out.begin(), it));

  return out;
}

void build_variable_outer_scope(const framework::ProgramDesc& pdesc,
                                VariableScope* var_scope, Scope* outer_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    // cerr << "var name "  << var->Name() << endl;
    auto v = outer_scope->Var(var->Name());

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
    }

    // auto v = new Variable();
    // v->GetMutable<LoDTensor>();
    InitializeVariable(v, var->GetType());
    var_scope->var_list.push_back(v);
  }
}

void build_variable_scope(const framework::ProgramDesc& pdesc,
                          VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    // cerr << "var name "  << var->Name() << endl;

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
    }

    auto v = new Variable();
    // v->GetMutable<LoDTensor>();
    InitializeVariable(v, var->GetType());
    var_scope->var_list.push_back(v);
  }
}

void build_op_func_list(const framework::ProgramDesc& pdesc,
                        std::vector<OperatorBase*>* op_list,
                        std::vector<OpFuncNode>* vec_func_list,
                        VariableScope* var_scope,
                        const platform::Place& place) {
  auto& global_block = pdesc.Block(0);

  for (auto& op : global_block.AllOps()) {
    // cerr << op->Type() << endl;
    // bool debug = op->Type() == "softmax_with_cross_entropy_grad";
    bool debug = false;

    // cerr << "create op" << endl;
    // auto op_base_u = OpRegistry::CreateOp(*op);
    auto& info = OpInfoMap::Instance().Get(op->Type());

    VariableNameMap inputs_1 = op->Inputs();
    VariableNameMap outputs_1 = op->Outputs();
    AttributeMap attrs_1 = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&attrs_1);
    }
    auto op_base = info.Creator()(op->Type(), inputs_1, outputs_1, attrs_1);

    auto input_names = op->Inputs();
    auto output_names = op->Outputs();

    OpFuncNode op_func_node;

    VariableValueMap ins_map;
    std::map<std::string, std::vector<int>> ins_name2id;
    for (auto& var_name_item : input_names) {
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
    if (debug) cerr << "1" << endl;

    VariableValueMap outs_map;
    std::map<std::string, std::vector<int>> outs_name2id;
    for (auto& var_name_item : output_names) {
      std::vector<Variable*> output_vars;
      std::vector<int> vec_ids;
      output_vars.reserve(var_name_item.second.size());
      for (auto& var_name : var_name_item.second) {
        auto it = var_scope->name2id.find(var_name);
        assert(it != var_scope->name2id.end());
        // cerr << it->second << "\t" << var_scope.var_list.size() << endl;
        output_vars.push_back(var_scope->var_list[it->second]);
        vec_ids.push_back(it->second);
      }
      outs_map[var_name_item.first] = output_vars;
      // cerr << ToTypeName(output_vars[0]->Type() ) << endl;
      outs_name2id[var_name_item.first] = vec_ids;
    }

    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;
    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);
    // cerr << "create runtime context" << endl;
    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);
    // cerr << "fin infer shape" << endl;
    auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op->Type());
    PADDLE_ENFORCE_NE(
        kernels_iter, all_op_kernels.end(),
        platform::errors::Unavailable(
            "There are no kernels which are registered in the %s operator.",
            op->Type()));

    // cerr << "create kernel" << endl;
    using OpKernelFunc = std::function<void(const ExecutionContext&)>;
    using OpKernelMap =
        std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;
    if (debug) cerr << "2" << endl;
    OpKernelMap& kernels = kernels_iter->second;
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);
    Scope scope;
    auto exec_ctx =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);
    if (debug) cerr << "21" << endl;
    auto expected_kernel_key =
        dynamic_cast<const framework::OperatorWithKernel*>(op_base)
            ->GetExpectedKernelType(exec_ctx);
    if (debug) cerr << "22" << endl;
    // cerr << "22" << endl;

    // add transfer log
    // cerr << "in map size " << ins_map.size() << endl;
    VariableValueMap& ins_map_temp = runtime_context.inputs;
    // cerr << "ins map siz" <<  ins_map_temp.size() << endl;
    for (auto& var_name_item : ins_map_temp) {
      // auto& vec_ids = ins_name2id[ var_name_item.first ];
      for (size_t i = 0; i < var_name_item.second.size(); ++i) {
        auto var = var_name_item.second[i];
        auto tensor_in = static_cast<const Tensor*>(&(var->Get<LoDTensor>()));
        if (!tensor_in->IsInitialized()) {
          continue;
        }
        // cerr << "i " << i << "\t" << tensor_in->IsInitialized() << endl;
        auto kernel_type_for_var =
            static_cast<const framework::OperatorWithKernel*>(op_base)
                ->GetKernelTypeForVar(var_name_item.first, *tensor_in,
                                      expected_kernel_key);
        if (debug) {
          cerr << "var name " << var_name_item.first << endl;
          cerr << expected_kernel_key.place_ << "\t"
               << kernel_type_for_var.place_ << endl;
        }
        if (!platform::is_same_place(kernel_type_for_var.place_,
                                     expected_kernel_key.place_)) {
          if (debug) cerr << "add data transfer" << endl;
          // cerr << "add data transfer " << op->Type() << endl;
          // cerr << " p1 "  << kernel_type_for_var.place_ << "\t" <<
          // expected_kernel_key.place_ << endl;
          // cerr << " var  " << var_name_item.first << endl;
          // need trans place
          // add var in scope
          // add copy op
          std::string new_var_name =
              "temp_1" + std::to_string(var_scope->var_list.size() + 1);
          auto v = new Variable();
          v->GetMutable<LoDTensor>();
          var_scope->name2id[new_var_name] = var_scope->var_list.size();
          var_scope->var_list.push_back(v);

          VariableNameMap copy_in_map;
          // cerr << "ints name is " << input_names[var_name_item.first][i] <<
          // endl;
          copy_in_map["X"] = {input_names[var_name_item.first][i]};
          VariableNameMap copy_out_map;
          copy_out_map["Out"] = {new_var_name};
          AttributeMap attr_map;
          attr_map["dst_place_type"] = convert(place);

          std::map<std::string, std::vector<int>> copy_ins_name2id;
          copy_ins_name2id["X"] = ins_name2id[var_name_item.first];
          std::map<std::string, std::vector<int>> copy_out_name2id;
          copy_out_name2id["Out"] = {var_scope->name2id[new_var_name]};

          // vec_ids[i] = var_scope->name2id[new_var_name];
          // update out runtime_context
          op_func_node.input_index[var_name_item.first][i] =
              var_scope->name2id[new_var_name];

          VariableValueMap copy_ins_value_map;
          copy_ins_value_map["X"] = {var};
          VariableValueMap copy_outs_value_map;
          copy_outs_value_map["Out"] = {v};

          auto& copy_info = OpInfoMap::Instance().Get("memcpy");
          auto copy_op = copy_info.Creator()("memcpy", copy_in_map,
                                             copy_out_map, attr_map);
          if (debug) cerr << "create memcpy" << endl;
          OpFuncNode copy_op_func_node;
          copy_op_func_node.input_index = copy_ins_name2id;
          copy_op_func_node.output_index = copy_out_name2id;

          RuntimeContext copy_runtime_context({}, {});
          copy_runtime_context.inputs.swap(copy_ins_value_map);
          copy_runtime_context.outputs.swap(copy_outs_value_map);
          // cerr << "create runtime context" << endl;
          RuntimeInferShapeContext copy_infer_shape_ctx(*copy_op,
                                                        copy_runtime_context);
          if (debug) cerr << "before infer shape" << endl;
          static_cast<const framework::OperatorWithKernel*>(copy_op)
              ->InferShape(&copy_infer_shape_ctx);
          if (debug) cerr << "infer shape" << endl;
          // cerr << "fin infer shape" << endl;
          auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
          auto kernels_iter = all_op_kernels.find("memcpy");
          PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                            platform::errors::Unavailable(
                                "There are no kernels which are registered in "
                                "the memcpy operator."));

          // cerr << "create kernel" << endl;
          using OpKernelFunc = std::function<void(const ExecutionContext&)>;
          using OpKernelMap = std::unordered_map<OpKernelType, OpKernelFunc,
                                                 OpKernelType::Hash>;

          OpKernelMap& kernels = kernels_iter->second;
          // auto place = platform::CPUPlace();
          // auto place = platform::CUDAPlace(0);

          platform::DeviceContextPool& pool =
              platform::DeviceContextPool::Instance();
          auto* dev_ctx = pool.Get(place);
          Scope scope;
          auto copy_exec_ctx =
              ExecutionContext(*copy_op, scope, *dev_ctx, copy_runtime_context);
          if (debug) cerr << "21" << endl;
          auto expected_kernel_key =
              dynamic_cast<const framework::OperatorWithKernel*>(copy_op)
                  ->GetExpectedKernelType(copy_exec_ctx);
          if (debug) cerr << "22" << endl;
          // cerr << "22" << endl;
          auto kernel_iter = kernels.find(expected_kernel_key);
          copy_op_func_node.kernel_func_ = OpKernelFunc(kernel_iter->second);
          copy_op_func_node.kernel_func_(copy_exec_ctx);
          if (debug) cerr << "run exe ctx" << endl;

          op_list->push_back(copy_op);
          vec_func_list->push_back(copy_op_func_node);

          var_name_item.second[i] = v;
        }
      }
    }

    op_list->push_back(op_base);

    auto kernel_iter = kernels.find(expected_kernel_key);

    if (debug) cerr << "3" << endl;
    op_func_node.kernel_func_ = OpKernelFunc(kernel_iter->second);
    if (debug) cerr << "3-1" << endl;
    op_func_node.kernel_func_(exec_ctx);
    vec_func_list->push_back(op_func_node);
    if (debug) cerr << "5" << endl;
  }
}

void exec_op_func_list(const std::vector<OpFuncNode>& vec_func_list,
                       const std::vector<OperatorBase*>& op_list,
                       const VariableScope& var_scope,
                       const platform::Place& place) {
  for (size_t i = 0; i < vec_func_list.size(); ++i) {
    auto& func_node = vec_func_list[i];
    auto op_base = op_list[i];
    // build runtime cost
    VariableValueMap ins_map;
    for (auto& var_name_item : func_node.input_index) {
      std::vector<Variable*> input_vars;

      input_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        // cerr << var_name_item.first << "\t " << id << endl;
        input_vars.emplace_back(var_scope.var_list[id]);
      }
      ins_map.emplace(var_name_item.first, std::move(input_vars));
    }

    VariableValueMap outs_map;
    for (auto& var_name_item : func_node.output_index) {
      std::vector<Variable*> out_vars;

      out_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        // cerr << var_name_item.first << "\t " << id << endl;
        out_vars.emplace_back(var_scope.var_list[id]);
      }
      outs_map.emplace(var_name_item.first, std::move(out_vars));
    }

    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);

    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);

    // dynamic_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
    // &infer_shape_ctx );
    // RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    auto* dev_ctx = pool.Get(place);
    Scope scope;

    auto exec_context =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);

    func_node.kernel_func_(exec_context);
  }
}

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place, const ProgramDesc& prog,
                  const ProgramDesc& startup_prog, Scope* scope)
      : place_(place), prog_(prog), outer_scope_(scope) {
    paddle::framework::InitDevices();

    is_build = false;

    paddle::framework::build_variable_outer_scope(startup_prog, &global_scope,
                                                  outer_scope_);

    std::vector<paddle::framework::OpFuncNode> vec_func_list;
    std::vector<paddle::framework::OperatorBase*> op_list;
    paddle::framework::build_op_func_list(
        startup_prog, &op_list, &vec_func_list, &global_scope, place_);
    // add variable to outer_scope
  }
  void run(const std::vector<std::string> vec_name,
           const std::vector<framework::Tensor>& vec_tensor,
           const std::vector<std::string>& vec_fetch_name,
           std::vector<framework::Tensor>* vec_out) {
    // cerr << "run" << endl;
    // set static data
    if (is_build == false) {
      paddle::framework::build_variable_scope(prog_, &global_scope);
    }
    for (size_t i = 0; i < vec_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_name[i]);
      // cerr << "find " << ( it != global_scope.name2id.end() ) <<endl;
      assert(it != global_scope.name2id.end());

      auto feed_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();
      // cerr << " get tensor" << endl;
      feed_tensor->ShareDataWith(vec_tensor[i]);
      // cerr << "share buffer with" << endl;
    }

    if (is_build == false) {
      paddle::framework::build_op_func_list(prog_, &op_list, &vec_func_list,
                                            &global_scope, place_);
      is_build = true;
      // convert vec func_list to graph
      convert();
    } else {
      // paddle::framework::exec_op_func_list( vec_func_list, op_list,
      // global_scope, place_ );
      // cerr <<  "exec instr" << endl;
      exec_instruction_list(vec_instruction_, global_scope, place_);
    }

    for (size_t i = 0; i < vec_fetch_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_fetch_name[i]);
      assert(it != global_scope.name2id.end());

      auto fetch_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();

      // cerr << "out  "  << fetch_tensor->data<float>()[0] << endl;
      if (platform::is_gpu_place(fetch_tensor->place())) {
        // cerr << "fetch gpu" << endl;
        Tensor out;
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        auto* dev_ctx = pool.Get(place_);
        dev_ctx->Wait();
        TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
        dev_ctx->Wait();
        // cerr << "out  " << out << endl;
        // cout << out.data<float>()[0] << endl;
        vec_out->push_back(out);
      } else {
        cerr << "out  " << *fetch_tensor << endl;
      }
    }
  }

 private:
  void convert() {
    input_var2op_info_.resize(global_scope.var_list.size());

    vec_instruction_.reserve(vec_func_list.size());
    dependecy_count_.resize(vec_func_list.size());
    global_scope.vec_meta_info_.resize(global_scope.var_list.size());
    // cerr << "in pos 7 is lookup table " <<
    // vec_instruction_[7].kernel_func_.operator_base_->Type() << endl;
    for (size_t i = 0; i < vec_func_list.size(); ++i) {
      Instruction temp_inst;
      temp_inst.kernel_func_.compute_func_ = vec_func_list[i].kernel_func_;
      temp_inst.kernel_func_.operator_base_ = op_list[i];
      temp_inst.input_index_ = vec_func_list[i].input_index;
      temp_inst.output_index_ = vec_func_list[i].output_index;

      std::vector<size_t> gc_check_input_list;
      for (auto item : vec_func_list[i].input_index) {
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
        global_scope.vec_meta_info_[var_id].var_ref_count_++;
      }

      temp_inst.gc_check_var_list.swap(gc_check_input_list);

      vec_instruction_.push_back(temp_inst);
    }

    for (size_t i = 0; i < vec_instruction_.size(); ++i) {
      std::vector<size_t> vec_temp;
      for (auto item : vec_instruction_[i].output_index_) {
        for (auto id : item.second) {
          vec_temp = merge_vec(vec_temp, input_var2op_info_[id]);
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

      for (auto inst_id : filter_next) {
        dependecy_count_[inst_id]++;
      }
    }
  }

  void run_instr(const Instruction& instr_node, const VariableScope& var_scope,
                 const platform::Place& place) {
    auto op_base = instr_node.kernel_func_.operator_base_;
    // build runtime cost
    VariableValueMap ins_map;
    for (auto& var_name_item : instr_node.input_index_) {
      std::vector<Variable*> input_vars;

      input_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        // cerr << var_name_item.first << "\t " << id << endl;
        input_vars.emplace_back(var_scope.var_list[id]);
      }
      ins_map.emplace(var_name_item.first, std::move(input_vars));
    }

    VariableValueMap outs_map;
    for (auto& var_name_item : instr_node.output_index_) {
      std::vector<Variable*> out_vars;

      out_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        // cerr << var_name_item.first << "\t " << id << endl;
        out_vars.emplace_back(var_scope.var_list[id]);
      }
      outs_map.emplace(var_name_item.first, std::move(out_vars));
    }

    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);

    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);

    // dynamic_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
    // &infer_shape_ctx );
    // RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    auto* dev_ctx = pool.Get(place);
    Scope scope;

    auto exec_context =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);

    instr_node.kernel_func_.compute_func_(exec_context);
  }

  void exec_instruction_list(const std::vector<Instruction>& vec_instr,
                             const VariableScope& var_scope,
                             const platform::Place& place) {
    // for( size_t i = 0; i < vec_instr.size(); ++i )
    // {
    //   cerr << vec_instr[i].kernel_func_.operator_base_->Type() <<  " dep " <<
    //   dependecy_count_[i] << endl;
    // }

    std::queue<size_t> working_queue;
    auto working_dependecy_count = dependecy_count_;
    for (size_t i = 0; i < dependecy_count_.size(); ++i) {
      if (dependecy_count_[i] == 0) {
        working_queue.push(i);
      }
    }

    auto working_var_ref = global_scope.vec_meta_info_;

    size_t run_op_number = 0;
    while (!working_queue.empty()) {
      auto instr_id = working_queue.front();
      working_queue.pop();
      // cerr << "run " << instr_id  << "\t" <<
      // vec_instr[instr_id].kernel_func_.operator_base_->Type()  <<  endl;
      auto& instr_node = vec_instr[instr_id];
      run_instr(instr_node, var_scope, place);

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
        cerr << " var ref is not zero " << i << endl;
        //  << i << global_scope.var_list[i].Name() << endl;
      }
    }
    // cerr << "run op number " << run_op_number << endl;
    // cerr << "total op number " << vec_instr.size() << endl;
    // assert( run_op_number == vec_instr.size() );

    /*
    for( size_t i = 0; i < vec_instr.size(); ++i )
    {
        auto& instr_node = vec_instr[i];
        run_instr( instr_node, var_scope, place );

    }
    */
  }

  const platform::Place& place_;
  const ProgramDesc& prog_;
  paddle::framework::VariableScope global_scope;
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  std::vector<paddle::framework::OperatorBase*> op_list;

  bool is_build;

  std::vector<Instruction> vec_instruction_;

  InstructionInfo instruction_info_;

  std::vector<size_t> dependecy_count_;
  std::vector<VariableMetaInfo> ref_coun_info;
  std::vector<std::vector<size_t>> input_var2op_info_;

  Scope* outer_scope_;
};
}  // namespace framework
}  // namespace paddle
