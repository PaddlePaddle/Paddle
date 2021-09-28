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
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include <algorithm>

#include "paddle/fluid/framework/executor_gc_helper.h"

namespace paddle {
namespace framework {
namespace interpretercore {

AtomicVectorSizeT& AsyncWorkQueue::PrepareAtomicDeps(
    const std::vector<size_t>& dependecy_count) {
  if (atomic_deps_.size() != dependecy_count.size()) {
    atomic_deps_.clear();
    std::generate_n(std::back_inserter(atomic_deps_), dependecy_count.size(),
                    [] { return std::make_unique<std::atomic<size_t>>(0); });
  }

  for (size_t i = 0; i < dependecy_count.size(); ++i) {
    atomic_deps_[i]->store(dependecy_count[i]);
  }
  return atomic_deps_;
}

AtomicVectorSizeT& AsyncWorkQueue::PrepareAtomicVarRef(
    const std::vector<VariableMetaInfo>& vec_meta_info) {
  if (atomic_var_ref_.size() != vec_meta_info.size()) {
    atomic_var_ref_.clear();
    std::generate_n(std::back_inserter(atomic_var_ref_), vec_meta_info.size(),
                    [] { return std::make_unique<std::atomic<size_t>>(0); });
  }

  for (size_t i = 0; i < vec_meta_info.size(); ++i) {
    atomic_var_ref_[i]->store(vec_meta_info[i].var_ref_count_);
  }
  return atomic_var_ref_;
}

bool var_can_be_deleted(const std::string& name, const BlockDesc& block) {
  auto* var_desc = block.FindVar(name);
  if (var_desc == nullptr || var_desc->Persistable()) {
    return false;
  }

  auto type = var_desc->Proto()->type().type();

  return type == proto::VarType::LOD_TENSOR ||
         type == proto::VarType::SELECTED_ROWS ||
         type == proto::VarType::LOD_TENSOR_ARRAY;
}

std::unordered_map<const paddle::framework::OperatorBase*,
                   std::vector<std::string>>
get_unused_vars(const BlockDesc& block, const std::vector<OperatorBase*>& ops) {
  std::unordered_map<std::string, size_t> var_op_idx_map;

  for (size_t i = 0; i < ops.size(); ++i) {
    auto* op = ops[i];

    OpInOutInfo info;
    for (auto& name_pair : op->Inputs()) {
      for (auto& name : name_pair.second) {
        if (!var_can_be_deleted(name, block)) {
          continue;
        }

        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op);
        }

        if (info.IsInArgBufferNeeded(name)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        } else {
          VLOG(10) << "Skip reference count computing of variable "
                   << name_pair.first << "(" << name << ") in Operator "
                   << op->Type();
        }
      }
    }

    for (auto& name_pair : op->Outputs()) {
      for (auto& name : name_pair.second) {
        if (var_can_be_deleted(name, block)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        }
      }
    }
  }

  std::unordered_map<const OperatorBase*, std::vector<std::string>> result;
  for (auto& name_op_idx_pair : var_op_idx_map) {
    auto& name = name_op_idx_pair.first;
    size_t op_idx = name_op_idx_pair.second;
    result[ops[op_idx]].emplace_back(name);
  }
  return result;
}

std::string get_memcpy_type(const platform::Place& src_place,
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

void build_variable_scope(const framework::ProgramDesc& pdesc,
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
    } else {
      auto var_id = var_scope->name2id[var->Name()];
      if (nullptr == var_scope->vec_meta_info_[var_id].vardesc_) {
        VLOG(3) << "update var:" << var->Name() << " desc from nullptr into "
                << var;
        var_scope->vec_meta_info_[var_id].vardesc_ = var;
      }
    }
  }
}

void build_op_func_list(const platform::Place& place,
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
    InterpretercoreInferShapeContext infer_shape_ctx(*op_base, runtime_context);
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
          if (op_base->Type() == "fetch_v2") {
            op_base->SetAttr("deepcopy", false);
          }
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
          auto memcpy_op_type = get_memcpy_type(kernel_type_for_var.place_,
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
          InterpretercoreInferShapeContext copy_infer_shape_ctx(
              *copy_op, copy_runtime_context);
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
          // NOTE(Aurelius84): memcpy_op is expensive operation, so we tag them
          // as kQueueSync and execute them in thread pool.
          copy_op_func_node.type_ = OpFuncType::kQueueSync;
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

std::vector<size_t> merge_vector(const std::vector<size_t>& first,
                                 const std::vector<size_t>& second) {
  std::vector<size_t> out(first.size() + second.size());
  std::merge(first.begin(), first.end(), second.begin(), second.end(),
             out.begin());

  std::vector<size_t>::iterator it;
  it = std::unique(out.begin(), out.end());

  out.resize(std::distance(out.begin(), it));

  return out;
}

}  // namespace interpretercore
}  // namespace framework
}  // namespace paddle
