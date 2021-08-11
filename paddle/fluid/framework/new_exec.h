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
#include <unordered_set>
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
#include "paddle/fluid/platform/event.h"
#include "paddle/fluid/platform/init.h"

// USE_OP(fill_constant);
// USE_OP(elementwise_add);

// using namespace std;

namespace paddle {
namespace framework {

using std::cerr;
using std::endl;

using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
using OpKernelMap =
    std::unordered_map<OpKernelType, OpKernelComputeFunc, OpKernelType::Hash>;

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

struct OpKernelFunc {
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

struct EventRun {
  explicit EventRun(size_t op_id) : op_id_(op_id) {}
  size_t op_id_;
};

struct NextInstruction {
  std::vector<size_t> direct_run_;
  std::vector<EventRun> event_wait_run_;
  std::vector<EventRun> synchronize_run_;
  std::vector<size_t> all_next_ops_;
};

struct EventInter {};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

struct Instruction {
  OpKernelFunc kernel_func_;
  std::map<std::string, std::vector<int>> input_index_;
  std::map<std::string, std::vector<int>> output_index_;

  std::vector<size_t> gc_check_var_list;
  NextInstruction next_instruction_;
  std::vector<EventInter> vec_event_list_;
  platform::DeviceContext* dev_ctx_;  // not owned
};
enum class MemcpyType {
  kD2H,
  kH2D,
  kH2H,
  kD2D,
};

enum class OpFuncType {
  kAsync,  // GPU Kernel
  kSync,   // CPU kernel, block host
  kEvent,  // d2h, h2d, send, recv, broadcast
};

struct OpFuncNode {
  // int unsed;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

  OpKernelComputeFunc kernel_func_;
  platform::DeviceContext* dev_ctx_;  // not owned
  OpFuncType type_;
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

std::pair<MemcpyType, std::string> GetMemcpyType(
    const platform::Place& src_place, const platform::Place& dst_place) {
  PADDLE_ENFORCE_EQ(platform::is_same_place(src_place, dst_place), false,
                    platform::errors::PreconditionNotMet(
                        "Required src_place shall be different with dst_place, "
                        "but received same place: %s",
                        src_place));
  if (platform::is_gpu_place(dst_place)) {
    return {MemcpyType::kH2D, "memcpy_h2d"};
  } else if (platform::is_gpu_place(src_place)) {
    return {MemcpyType::kD2H, "memcpy_d2h"};
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Not support Memcpy typ : %s -> %s", src_place, dst_place));
  }
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
    auto v = outer_scope->Var(var->Name());

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
    }

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

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
    }
    VLOG(3) << "Insert var " << var->Name()
            << " with id: " << var_scope->name2id[var->Name()];

    auto v = new Variable();
    InitializeVariable(v, var->GetType());
    var_scope->var_list.push_back(v);
  }
}

/*
 * Parse the var_ids that need to be associated with an event.
 * The caller should guarantee front_op and back_op satisfy the
 * following conditions:
 *   1. kAsync -> kEvent
 *   2. kEvent -> kAsync
 *
 * For example: matmul(gpu) -> out_var -> memcpy_d2h
 * out_var should be associated with an event.
 */
void UpdateEventVarId(
    const std::map<std::string, std::vector<int>>& front_op_out_vars,
    const std::map<std::string, std::vector<int>>& back_op_in_vars,
    std::vector<size_t>* even_var_ids) {
  std::unordered_set<size_t> unique_var_ids;
  for (auto& item : front_op_out_vars) {
    unique_var_ids.insert(item.second.begin(), item.second.end());
  }

  for (auto& item : back_op_in_vars) {
    for (auto var_id : item.second) {
      if (unique_var_ids.count(var_id) > 0) {
        even_var_ids->push_back(var_id);
      }
    }
  }
}

void ParseDirectAndEventRunOps(
    const std::vector<OpFuncNode>& op_func_nodes,
    const std::vector<size_t>& downstream_ops, size_t op_index,
    std::map<size_t, platform::CudaEvent>* var_id2event,
    Instruction* instruction) {
  // In build_op_func_list:
  // 1. all memcpy_op is kEvent
  // 2. all CPU Kernel is kSync
  // 3. all rest GPU Kernel is kAsync temporarily

  auto& op_func_type = op_func_nodes[op_index].type_;
  auto& next_instruction = instruction->next_instruction_;
  std::vector<size_t> event_var_ids;

  if (op_func_type == OpFuncType::kSync) {
    // all downstream ops of CPU can directly run.
    next_instruction.direct_run_ = downstream_ops;
  } else if (op_func_type == OpFuncType::kAsync) {
    for (auto next_op_id : downstream_ops) {
      if (op_func_nodes[next_op_id].type_ == OpFuncType::kAsync) {
        // GPU -> GPU, then next_op can directly run.
        next_instruction.direct_run_.emplace_back(next_op_id);
      } else if (op_func_nodes[next_op_id].type_ == OpFuncType::kEvent) {
        // GPU -> D2H, then should be as stream_wait_event
        UpdateEventVarId(op_func_nodes[op_index].output_index,
                         op_func_nodes[next_op_id].input_index, &event_var_ids);
        next_instruction.event_wait_run_.emplace_back(next_op_id);
      } else {
        PADDLE_THROW(platform::errors::PreconditionNotMet(
            "Unsupported  AyncOp -> SyncOp."));
      }
    }
  } else {  // kEvent;
    for (auto next_op_id : downstream_ops) {
      if (op_func_nodes[next_op_id].type_ == OpFuncType::kAsync) {
        // H2D -> GPU Kernel, then should be as stream_wait_event
        UpdateEventVarId(op_func_nodes[op_index].output_index,
                         op_func_nodes[next_op_id].input_index, &event_var_ids);
        next_instruction.event_wait_run_.emplace_back(next_op_id);
      } else if (op_func_nodes[next_op_id].type_ == OpFuncType::kSync) {
        // // D2H -> CPU Kernel, then should be as synchronize
        UpdateEventVarId(op_func_nodes[op_index].output_index,
                         op_func_nodes[next_op_id].input_index, &event_var_ids);
        next_instruction.synchronize_run_.emplace_back(next_op_id);
      } else {
        PADDLE_THROW(platform::errors::PreconditionNotMet(
            "Unsupported  EventOp -> EventOp."));
      }
    }
  }
#ifdef PADDLE_WITH_CUDA
  // Create event for these cross-stream vars
  VLOG(3) << instruction->kernel_func_.operator_base_->Type()
          << " event_var_ids.size: " << event_var_ids.size();
  for (auto var_id : event_var_ids) {
    if (var_id2event->find(var_id) == var_id2event->end()) {
      // Specific cudaEventDisableTiming to get best performance.
      VLOG(3) << "create event for " << var_id;
      var_id2event->emplace(var_id,
                            platform::get_cuda_flags(false, false, false));
    }
  }
#endif
}

void build_op_func_list(const framework::ProgramDesc& pdesc,
                        const platform::Place& place,
                        std::vector<OperatorBase*>* op_list,
                        std::vector<OpFuncNode>* vec_func_list,
                        VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);
  auto& all_op_kernels = OperatorWithKernel::AllOpKernels();

  for (auto& op : global_block.AllOps()) {
    VLOG(3) << "Build OpFuncNode from : " << op->Type();
    auto& info = OpInfoMap::Instance().Get(op->Type());

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();
    AttributeMap op_attr_map = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }
    auto op_base =
        info.Creator()(op->Type(), inputs_names, outputs_names, op_attr_map);
    // step 1. Prepare VariableValueMap of input/output
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
    if (op_base->HasAttr("op_device")) {
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

    VariableValueMap& ins_map_temp = runtime_context.inputs;

    // step 3. Insert memcpy_op if needed
    for (auto& var_name_item : ins_map_temp) {
      for (size_t i = 0; i < var_name_item.second.size(); ++i) {
        auto var = var_name_item.second[i];
        auto tensor_in = static_cast<const Tensor*>(&(var->Get<LoDTensor>()));
        if (!tensor_in->IsInitialized()) {
          VLOG(3) << "Skip " << var_name_item.first
                  << ", because it's not initialized.";
          continue;
        }
        VLOG(3) << var_name_item.first << ".place: " << tensor_in->place();
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
          attr_map["dst_place_type"] = convert(expected_kernel_key.place_);

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
                                     memcpy_op_type.second, x_iter->second[i],
                                     kernel_type_for_var.place_, new_var_name,
                                     expected_kernel_key.place_);
          auto& copy_info = OpInfoMap::Instance().Get(memcpy_op_type.second);
          auto copy_op = copy_info.Creator()(memcpy_op_type.second, copy_in_map,
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

          auto kernels_iter = all_op_kernels.find(memcpy_op_type.second);
          PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                            platform::errors::Unavailable(
                                "There are no kernels which are registered in "
                                "the memcpy operator."));

          OpKernelMap& kernels = kernels_iter->second;
          platform::DeviceContext* dev_ctx = pool.Get(place);

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
          VLOG(3) << "Run " << memcpy_op_type.second << " done.";
          copy_op_func_node.type_ = OpFuncType::kEvent;
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
      // It will be updated in ParseDirectAndEventRunOps.
      op_func_node.type_ = OpFuncType::kAsync;
    } else if (platform::is_cpu_place(expected_kernel_key.place_)) {
      op_func_node.type_ = OpFuncType::kSync;
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
    VLOG(3) << "run " << op_base->Type() << " done.";
  }
}

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place, const ProgramDesc& prog,
                  const ProgramDesc& startup_prog, Scope* scope)
      : place_(place),
        prog_(prog),
        d2h_context_pool_({place}),
        h2d_context_pool_({place}),
        outer_scope_(scope) {
    paddle::framework::InitDevices();

    is_build_ = false;

    if (outer_scope_ != nullptr) {
      auto name_list = outer_scope_->LocalVarNames();
      for (auto name : name_list) {
        auto v = outer_scope_->Var(name);
        if (global_scope.name2id.find(name) == global_scope.name2id.end()) {
          global_scope.name2id[name] = global_scope.var_list.size();
        }

        global_scope.var_list.push_back(v);
      }
    }

    paddle::framework::build_variable_outer_scope(startup_prog, &global_scope,
                                                  outer_scope_);

    std::vector<paddle::framework::OpFuncNode> vec_func_list;
    std::vector<paddle::framework::OperatorBase*> op_list;
    paddle::framework::build_op_func_list(startup_prog, place_, &op_list,
                                          &vec_func_list, &global_scope);
  }
  void run(const std::vector<std::string>& vec_name,
           const std::vector<framework::Tensor>& vec_tensor,
           const std::vector<std::string>& vec_fetch_name,
           std::vector<framework::Tensor>* vec_out) {
    if (is_build_ == false) {
      paddle::framework::build_variable_scope(prog_, &global_scope);
    }
    for (size_t i = 0; i < vec_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_name[i]);
      assert(it != global_scope.name2id.end());

      auto feed_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();
      feed_tensor->ShareDataWith(vec_tensor[i]);
    }

    if (is_build_ == false) {
      paddle::framework::build_op_func_list(prog_, place_, &op_list,
                                            &vec_func_list, &global_scope);
      is_build_ = true;
      // convert vec func_list to graph
      convert();
    } else {
      exec_instruction_list(vec_instruction_, global_scope, place_);
    }

    for (size_t i = 0; i < vec_fetch_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_fetch_name[i]);
      assert(it != global_scope.name2id.end());
      PADDLE_ENFORCE_NE(it, global_scope.name2id.end(),
                        platform::errors::NotFound(
                            "Can't find (%d) the fetch var (%s) in scope", i,
                            vec_fetch_name[i]));
      VLOG(3) << "start to fetch " << vec_fetch_name[i];
      auto fetch_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();

      if (platform::is_gpu_place(fetch_tensor->place())) {
        VLOG(3) << vec_fetch_name[i]
                << " is one GPU, should wait stream finished.";
        Tensor out;
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        auto* dev_ctx = pool.Get(place_);
        dev_ctx->Wait();
        TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
        dev_ctx->Wait();
        vec_out->push_back(out);
      } else {
        Tensor out;
        TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
        vec_out->push_back(out);
      }
    }
  }

  platform::DeviceContextPool& D2HContextPool() { return d2h_context_pool_; }

  platform::DeviceContextPool& H2DContextPool() { return h2d_context_pool_; }

 private:
  platform::DeviceContext* ParseDeviceContextForInstruction(
      const OpFuncNode& op_func_node, const OperatorBase& op_base) {
    auto& op_type = op_base.Type();
    auto* dev_ctx = op_func_node.dev_ctx_;
    if (op_type == "memcpy_d2h") {
      VLOG(3) << "Get dev_ctx from d2h_context_pool_";
      dev_ctx = d2h_context_pool_.Get(place_);
    } else if (op_type == "memcpy_h2d") {
      VLOG(3) << "Get dev_ctx from h2d_context_pool_";
      dev_ctx = h2d_context_pool_.Get(place_);
    }

    return dev_ctx;
  }

  void convert() {
    input_var2op_info_.resize(global_scope.var_list.size());

    vec_instruction_.reserve(vec_func_list.size());
    dependecy_count_.resize(vec_func_list.size());
    global_scope.vec_meta_info_.resize(global_scope.var_list.size());
    for (size_t i = 0; i < vec_func_list.size(); ++i) {
      Instruction temp_inst;
      auto* op_base = op_list[i];
      temp_inst.dev_ctx_ =
          ParseDeviceContextForInstruction(vec_func_list[i], *op_base);
      temp_inst.kernel_func_.compute_func_ = vec_func_list[i].kernel_func_;
      temp_inst.kernel_func_.operator_base_ = op_base;
      temp_inst.input_index_ = vec_func_list[i].input_index;
      temp_inst.output_index_ = vec_func_list[i].output_index;

      std::vector<size_t> gc_check_input_list;
      for (auto& item : vec_func_list[i].input_index) {
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
      // Get all downstream ops
      std::vector<size_t> vec_temp;
      for (auto& item : vec_instruction_[i].output_index_) {
        for (auto id : item.second) {
          vec_temp = merge_vec(vec_temp, input_var2op_info_[id]);
        }
      }

      // In Program, op order is a very important information.
      // Op can only add op after it as next as next ops.
      std::vector<size_t> filter_next;
      filter_next.reserve(vec_temp.size());
      for (auto item : vec_temp) {
        if (item > i) {
          filter_next.push_back(item);
        }
      }

      ParseDirectAndEventRunOps(vec_func_list, filter_next, i, &var_id2event_,
                                &vec_instruction_[i]);

      for (auto inst_id : filter_next) {
        dependecy_count_[inst_id]++;
      }

      vec_instruction_[i].next_instruction_.all_next_ops_ =
          std::move(filter_next);
    }
  }

  void run_instr(const Instruction& instr_node, const VariableScope& var_scope,
                 const platform::Place& place) {
    auto op_base = instr_node.kernel_func_.operator_base_;
    VLOG(3) << "start  run_instr : " << op_base->Type();
    // build runtime cost
    VariableValueMap ins_map;
    for (auto& var_name_item : instr_node.input_index_) {
      std::vector<Variable*> input_vars;

      input_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        input_vars.emplace_back(var_scope.var_list[id]);
      }
      ins_map.emplace(var_name_item.first, std::move(input_vars));
    }

    VariableValueMap outs_map;
    for (auto& var_name_item : instr_node.output_index_) {
      std::vector<Variable*> out_vars;

      out_vars.reserve(var_name_item.second.size());
      for (auto& id : var_name_item.second) {
        out_vars.emplace_back(var_scope.var_list[id]);
      }
      outs_map.emplace(var_name_item.first, std::move(out_vars));
    }

    RuntimeContext runtime_context({}, {});
    runtime_context.inputs.swap(ins_map);
    runtime_context.outputs.swap(outs_map);

    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);

    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);

    auto* dev_ctx = instr_node.dev_ctx_;
    Scope scope;

    auto exec_context =
        ExecutionContext(*op_base, scope, *dev_ctx, runtime_context);

    instr_node.kernel_func_.compute_func_(exec_context);
  }

  void exec_instruction_list(const std::vector<Instruction>& vec_instr,
                             const VariableScope& var_scope,
                             const platform::Place& place) {
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
      auto& instr_node = vec_instr[instr_id];
      // step1 : stream_wait (non-block host) or sync (block host)
      StreamWaitEventOrSync(instr_node, vec_func_list[instr_id]);

      // step2: run instruction
      run_instr(instr_node, var_scope, place);
      ++run_op_number;

      // step3: insert event for out_vars if needed
      RecordEventInstruction(instr_node, vec_func_list[instr_id]);

      // step4: update working_queue
      auto& next_instr = instr_node.next_instruction_.all_next_ops_;

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
      }
    }
  }

  void RecordEventInstruction(const Instruction& instruction,
                              const OpFuncNode& op_func_node) {
    // If InterpreterCore in on CPUPlace, do nothing.
    if (platform::is_cpu_place(place_)) return;

#ifdef PADDLE_WITH_CUDA
    const platform::CUDADeviceContext* dev_ctx =
        reinterpret_cast<const platform::CUDADeviceContext*>(
            op_func_node.dev_ctx_);
    for (auto& item : instruction.output_index_) {
      for (auto out_var_id : item.second) {
        if (var_id2event_.count(out_var_id) != 0) {
          VLOG(3) << "Record event in out_var_id: " << out_var_id;
          var_id2event_[out_var_id].Record(*(dev_ctx->context()->Stream()));
        }
      }
    }
#endif
  }

  void WaitOrSync(const Instruction& instruction,
                  const platform::DeviceContext* dev_ctx, bool is_sync) {
#ifdef PADDLE_WITH_CUDA
    auto* cuda_dev_ctx =
        reinterpret_cast<const platform::CUDADeviceContext*>(dev_ctx);

    for (auto& item : instruction.input_index_) {
      for (auto in_var_id : item.second) {
        if (var_id2event_.count(in_var_id) != 0) {
          if (is_sync) {
            // block host until event is done
            VLOG(3) << "host sync wait in_var_id " << in_var_id;
            var_id2event_[in_var_id].Synchronize();
          } else {
            // non-block host, just add dependency in dev_ctx.stream to wait
            // event.
            VLOG(3) << "stream async wait in_var_id " << in_var_id;
            cuda_dev_ctx->context()->Stream()->WaitEvent(
                var_id2event_[in_var_id].GetRawCudaEvent());
          }
        }
      }
    }
#endif
  }

  void StreamWaitEventOrSync(const Instruction& instruction,
                             const OpFuncNode& op_func_node) {
    // If InterpreterCore in on CPUPlace, do nothing.
    if (platform::is_cpu_place(place_)) return;

    VLOG(3) << "Deal StreamWaitEventOrSync for "
            << instruction.kernel_func_.operator_base_->Type()
            << " type: " << static_cast<int>(op_func_node.type_);
    auto* dev_ctx = op_func_node.dev_ctx_;
    auto& op_func_type = op_func_node.type_;

    if (op_func_type == OpFuncType::kAsync) {
      WaitOrSync(instruction, dev_ctx, /*is_sync=*/false);
    } else if (op_func_type == OpFuncType::kEvent) {
      if (instruction.kernel_func_.operator_base_->Type() != "memcpy_h2d") {
        WaitOrSync(instruction, dev_ctx, /*is_sync=*/false);
      }
    } else {  // kSync
      WaitOrSync(instruction, dev_ctx, /*is_sync=*/true);
    }
  }

  const platform::Place& place_;
  const ProgramDesc& prog_;
  paddle::framework::VariableScope global_scope;
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  std::vector<paddle::framework::OperatorBase*> op_list;

  platform::DeviceContextPool d2h_context_pool_;
  platform::DeviceContextPool h2d_context_pool_;

  bool is_build_;

  std::vector<Instruction> vec_instruction_;

  InstructionInfo instruction_info_;

  std::vector<size_t> dependecy_count_;
  std::vector<VariableMetaInfo> ref_coun_info;
  std::vector<std::vector<size_t>> input_var2op_info_;

  std::map<size_t, platform::CudaEvent> var_id2event_;

  Scope* outer_scope_;
};
}  // namespace framework
}  // namespace paddle
