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
#include "paddle/fluid/framework/new_executor/data_transfer.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/pten/core/kernel_factory.h"

PADDLE_DEFINE_EXPORTED_bool(
    new_executor_sequential_run, false,
    "Enable sequential execution for standalone executor, used for debug");

namespace paddle {
namespace framework {
namespace interpreter {

void AsyncWorkQueue::AddTask(const OpFuncType& op_func_type,
                             std::function<void()> fn) {
  // NOTE(zhiqiu): use thhe second queue of size of, so only one thread is used.
  if (FLAGS_new_executor_sequential_run) {
    VLOG(4) << "FLAGS_new_executor_sequential_run:"
            << FLAGS_new_executor_sequential_run;
    queue_group_->AddTask(static_cast<size_t>(OpFuncType::kQueueAsync),
                          std::move(fn));
  } else {
    queue_group_->AddTask(static_cast<size_t>(op_func_type), std::move(fn));
  }
}

using VariableIdMap = std::map<std::string, std::vector<int>>;

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
get_unused_vars(const BlockDesc& block,
                const std::vector<std::shared_ptr<OperatorBase>>& ops) {
  std::unordered_map<std::string, size_t> var_op_idx_map;

  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];

    OpInOutInfo info;
    for (auto& name_pair : op->Inputs()) {
      for (auto& name : name_pair.second) {
        if (!var_can_be_deleted(name, block)) {
          continue;
        }

        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op.get());
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

    result[ops[op_idx].get()].emplace_back(name);
  }
  return result;
}

void build_variable_scope(const framework::BlockDesc& block,
                          VariableScope* var_scope, bool use_local_scope) {
  VLOG(3) << "Creating Variables";
  auto inner_scope = var_scope->GetMutableScope();

  // NOTE(zhiqiu): if create_local_scope_ is true, the persistable is
  // created in var_scope.scope_ , and other scope is created in local scope.
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();

  for (auto& var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    // TODO(xiongkun): user may create a variable with name that exists before.
    // under such circumstances, we should raise a error. Currently we can't
    // get the var_desc of startup_program, so leave it later.
    if (var_name == framework::kEmptyVarName) {
      continue;
    }
    if (var_desc->Persistable()) {
      auto* ptr = inner_scope->Var(var_name);

      VLOG(3) << "Initialize Variable " << var_name;
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " global, which pointer is "
              << ptr << " type is " << static_cast<int>(var_desc->GetType());
    } else {
      auto* ptr = local_scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " locally, which pointer is "
              << ptr << "Variable Type "
              << static_cast<int>(var_desc->GetType());
    }
    var_scope->SetVarDesc(var_name, var_desc);
  }
}

void create_all_ops(const framework::BlockDesc& block,
                    std::vector<std::unique_ptr<OperatorBase>>* ops) {
  for (auto& op : block.AllOps()) {
    VLOG(3) << "CreateOp from : " << op->Type();

    auto& info = OpInfoMap::Instance().Get(op->Type());

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();
    AttributeMap op_attr_map = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }
    auto op_base =
        info.Creator()(op->Type(), inputs_names, outputs_names, op_attr_map);
    ops->emplace_back(std::unique_ptr<OperatorBase>(op_base));
  }
}

std::tuple<VariableValueMap, VariableIdMap> build_variable_map(
    const VariableNameMap& var_name_map, VariableScope* var_scope,
    bool enforce_exist = true) {
  VariableValueMap name2var;
  VariableIdMap name2id;
  for (auto& item : var_name_map) {
    std::vector<Variable*> vars;
    std::vector<int> ids;
    vars.reserve(item.second.size());

    for (auto& var_name : item.second) {
      if (!enforce_exist && !var_scope->HasVar(var_name)) {
        // skip the non-exist variable: such as recurrent_grad
        VLOG(4) << var_name << " don't exist in variable scope, skip it!";
        continue;
      }
      auto var_id = var_scope->VarId(var_name);
      auto* in_var = var_scope->Var(var_id);
      vars.push_back(in_var);
      ids.push_back(var_id);
    }
    name2var[item.first] = std::move(vars);
    name2id[item.first] = std::move(ids);
  }
  return std::make_tuple(name2var, name2id);
}

void apply_device_guard(const OperatorBase* op_base,
                        const platform::Place& place,
                        OpKernelType* expected_kernel_key) {
  bool need_change_place =
      (op_base->HasAttr("op_device") &&
       (op_base->Attr<std::string>("op_device").length() > 0));
  if (need_change_place) {
    auto& op_device = op_base->Attr<std::string>("op_device");
    if (op_device == "cpu" || platform::is_cpu_place(place)) {
      VLOG(3) << "Switch into CPUPlace by device_guard.";
      expected_kernel_key->place_ = platform::CPUPlace();
    } else if (op_device.find("gpu") != std::string::npos &&
               (platform::is_gpu_place(place) ||
                platform::is_npu_place(place))) {
      // when the Op that only has CPUKernel is assigned to GPU, the CPUKernel
      // will be executed and a warning will be given at the same time.
      if (op_base->SupportGPU()) {
        expected_kernel_key->place_ = place;
      } else if (op_base->SupportNPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no CUDA implementation. It will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else {
      PADDLE_THROW(
          platform::errors::Fatal("Unsupported current place %s", op_device));
    }
  }
}

void deal_operator_base(const platform::Place& place,
                        const VariableScope* var_scope,
                        std::shared_ptr<OperatorBase> op_base,
                        OpFuncNode* op_func_node, Scope* local_scope) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  // input, output is prepared. set the other attributes.
  op_func_node->operator_base_ = op_base;
  if (platform::is_gpu_place(place)) {
    op_func_node->type_ = OpFuncType::kQueueAsync;
  } else if (platform::is_cpu_place(place)) {
    op_func_node->type_ = OpFuncType::kQueueSync;
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("Unsupported current place %s", place));
  }

  op_func_node->kernel_func_ = nullptr;
  op_base->Run(*local_scope, place);  // Run without data transformer.

  std::unordered_set<int> no_data_transform_index;
  for (auto& it : op_func_node->input_index) {
    for (auto& id : it.second) {
      no_data_transform_index.emplace(id);
    }
  }
  op_func_node->no_data_transform_index =
      no_data_transform_index;  // all index is no-need-transform
  op_func_node->dev_ctx_ = dev_ctx;
}

void build_op_func_list(const platform::Place& place,
                        const framework::BlockDesc& block,
                        std::vector<OpFuncNode>* vec_func_list,
                        VariableScope* var_scope, bool use_local_scope) {
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();
  auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
  std::vector<std::unique_ptr<OperatorBase>>
      ops_unique;  // its elements will be moved to vec_func_list
  // Step 1: create all ops for current block.
  create_all_ops(block, &ops_unique);
  // If gc is enabled and block size > 1
  const ProgramDesc& main_program = *block.Program();
  operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
      main_program, block.ID(), ops_unique);
  operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
      main_program, block.ID(), ops_unique);
  operators::PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
      main_program, block.ID(), ops_unique);

  std::vector<std::shared_ptr<OperatorBase>>
      ops;  // its elements will be moved to vec_func_list
  for (auto& op_unique : ops_unique) {
    ops.emplace_back(std::move(op_unique));
  }
  auto unused_var_map = get_unused_vars(block, ops);

  for (size_t i = 0; i < ops.size(); ++i) {
    auto op = ops[i].get();
    VLOG(6) << "Build OpFuncNode from : " << op->Type();

    auto inputs_names = op->Inputs();
    auto outputs_names = op->Outputs();

    VariableValueMap ins_map;
    VariableIdMap ins_name2id;
    bool enforce_exist = true;
    if (op->Type() == "recurrent_grad" || op->Type() == "rnn_memory_helper" ||
        op->Type() == "rnn_memory_helper_grad" ||
        op->Type() == "conditional_block" ||
        op->Type() == "conditional_block_grad" || op->Type() == "while" ||
        op->Type() == "while_grad") {
      enforce_exist = false;
    }
    std::tie(ins_map, ins_name2id) =
        build_variable_map(inputs_names, var_scope, enforce_exist);

    VariableValueMap outs_map;
    VariableIdMap outs_name2id;
    std::tie(outs_map, outs_name2id) =
        build_variable_map(outputs_names, var_scope, enforce_exist);

    // step 2: build OpFuncNode
    OpFuncNode op_func_node;
    op_func_node.operator_base_ = ops[i];
    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;

    if (dynamic_cast<const framework::OperatorWithKernel*>(op) == nullptr) {
      // op is not a operatorwithkernel, so direcly run OperatorBase::Run()
      deal_operator_base(place, var_scope, ops[i], &op_func_node, local_scope);
    } else {
      auto op_with_kernel =
          static_cast<const framework::OperatorWithKernel*>(op);
      // construct RuntimeContext and analysis KernelType
      RuntimeContext runtime_context({}, {});
      runtime_context.inputs.swap(ins_map);
      runtime_context.outputs.swap(outs_map);

      // see OperatorWithKernel::RunImpl in operator.cc for why
      if (!(op->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        InterpretercoreInferShapeContext infer_shape_ctx(*op, runtime_context);
        // TODO(Aurelius84): In case of control flow ops, they are NOT
        // inheritted
        // from OperatorWithKernel.
        op_with_kernel->Info().infer_shape_(&infer_shape_ctx);
      }

      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(place);
      Scope scope;
      auto expected_kernel_key = op_with_kernel->GetExpectedKernelType(
          ExecutionContext(*op, scope, *dev_ctx, runtime_context));

      // change device by the device_guard()
      apply_device_guard(op, place, &expected_kernel_key);
      VLOG(3) << "expected_kernel_key : " << expected_kernel_key;

      // step 3. apply data transforms and insert data transfer ops
      VariableValueMap& ins_map_temp = runtime_context.inputs;

      // NOTE(zhiqiu): op_func_node->operator_base_ maybe changed in
      // ApplyDataTransform
      ApplyDataTransform(expected_kernel_key, place, &ins_map_temp, var_scope,
                         &op_func_node, vec_func_list, use_local_scope);
      op_with_kernel = static_cast<const framework::OperatorWithKernel*>(
          op_func_node.operator_base_.get());

      // step 4. Run op kernel
      VLOG(3) << op_with_kernel->Type()
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
      VLOG(3) << op_with_kernel->Type()
              << " : expected_kernel_key : " << expected_kernel_key;
      auto exec_ctx =
          ExecutionContext(*op_with_kernel, scope, *dev_ctx, runtime_context);

      auto run_pten_kernel = false;
      if (pten::KernelFactory::Instance().HasCompatiblePtenKernel(
              op_with_kernel->Type())) {
        auto pt_kernel_key = op_with_kernel->ChoosePtenKernel(exec_ctx);
        auto pt_kernel_name = op_with_kernel->PtenKernelSignature()->name;

        if (op_with_kernel->PtenKernel()->IsValid()) {
          run_pten_kernel = true;
        } else {
          auto kernels_iter = all_op_kernels.find(op_with_kernel->Type());
          if (kernels_iter == all_op_kernels.end() ||
              kernels_iter->second.find(expected_kernel_key) ==
                  kernels_iter->second.end()) {
            auto pt_cpu_kernel_key = FallBackToCpu(
                expected_kernel_key, pt_kernel_key, *op_with_kernel);
            op_with_kernel->ResetPtenKernel(
                new pten::Kernel(pten::KernelFactory::Instance().SelectKernel(
                    pt_kernel_name, pt_cpu_kernel_key)));
            if (op_with_kernel->PtenKernel()->IsValid()) {
              VLOG(6) << "Static mode PrepareImpl - kernel name: "
                      << pt_kernel_name
                      << " | kernel key: " << pt_cpu_kernel_key
                      << " | kernel: " << *(op_with_kernel->PtenKernel());
              run_pten_kernel = true;
            }
          }
        }
      }
      VLOG(3) << op_with_kernel->Type()
              << " : expected_kernel_key : " << expected_kernel_key;
      if (run_pten_kernel) {
        pten::KernelContext pt_kernel_context;
        op_with_kernel->BuildPtenKernelContext(runtime_context, dev_ctx,
                                               &pt_kernel_context);
        op_func_node.pt_kernel_ = op_with_kernel->PtenKernel();

        (*op_func_node.pt_kernel_)(&pt_kernel_context);
      } else {
        auto kernels_iter = all_op_kernels.find(op->Type());
        PADDLE_ENFORCE_NE(
            kernels_iter, all_op_kernels.end(),
            platform::errors::Unavailable(
                "There are no kernels which are registered in the %s operator.",
                op->Type()));
        OpKernelMap& kernels = kernels_iter->second;

        auto kernel_iter = kernels.find(expected_kernel_key);
        PADDLE_ENFORCE_NE(
            kernel_iter, kernels.end(),
            platform::errors::NotFound(
                "Operator (%s) does not have kernel for %s.", op->Type(),
                KernelTypeToString(expected_kernel_key)));
        // TODO(zhiqiu): add fallback logic
        op_func_node.kernel_func_ = OpKernelComputeFunc(kernel_iter->second);
        op_func_node.kernel_func_(exec_ctx);
      }

      // post-process grad_op.outputs if need cast complex grad into real grad.
      // NOTE(Aurelius84): insert a transfer_dtype_op inplacely to cast it.
      if (framework::IsComplexType(expected_kernel_key.data_type_)) {
        interpreter::HandleComplexGradToRealGrad(
            op_func_node, place, outputs_names, &runtime_context.outputs,
            var_scope, vec_func_list, local_scope);
      }
    }

    vec_func_list->emplace_back(op_func_node);
    // gc---------------------------------------------------------------------------
    auto iter = unused_var_map.find(op);
    if (iter == unused_var_map.end()) {
      continue;
    }

    auto& delete_vars = iter->second;
    std::deque<std::shared_ptr<memory::Allocation>>* garbages =
        new std::deque<std::shared_ptr<memory::Allocation>>();

    for (auto& var_name : delete_vars) {
      auto* var = var_scope->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }

      VLOG(6) << "Erase variable " << var_name;
      if (var->IsType<LoDTensor>()) {
        garbages->emplace_back(
            var->GetMutable<LoDTensor>()->MoveMemoryHolder());
      } else if (var->IsType<pten::SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<pten::SelectedRows>()
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

    VLOG(3) << "run " << op->Type() << " done.";
  }
}

void add_fetch(const std::vector<std::string>& fetch_names,
               framework::BlockDesc* block) {
  auto* fetch_holder = block->Var(kFetchVarName);
  fetch_holder->SetType(proto::VarType::FETCH_LIST);
  fetch_holder->SetPersistable(true);

  int i = 0;
  for (auto& fetch_name : fetch_names) {
    // append fetch op
    auto* op = block->AppendOp();
    op->SetType("fetch_v2");
    op->SetInput("X", {fetch_name});
    op->SetOutput("Out", {kFetchVarName});
    op->SetAttr("col", {static_cast<int>(i)});
    op->CheckAttrs();
    i++;
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

void update_var_min_rw_op(const std::map<int, std::set<int>>& op2dependences,
                          std::map<int, std::list<int>>* var2min_rw_op,
                          int cur_op, int rw_var) {
  // rw_var is inputs or outputs of cur_op
  // this function update the var2min_rw_op set .
  if (var2min_rw_op->find(rw_var) == var2min_rw_op->end()) {
    (*var2min_rw_op)[rw_var] = std::list<int>();
  }
  for (auto dep_op : op2dependences.at(cur_op)) {
    var2min_rw_op->at(rw_var).remove(dep_op);
  }
  var2min_rw_op->at(rw_var).push_back(cur_op);
}

std::map<int, std::list<int>> get_downstream_map(
    const std::map<int, std::set<int>>& op2dependences) {
  // op2dependences is op -> it's dependences. we want to get op -> [ops] map,
  // where ops is the next instruction of op.
  std::map<int, std::list<int>> result;
  for (auto& item : op2dependences) {
    int op = item.first;
    for (auto dep_op : item.second) {
      if (result.find(dep_op) == result.end())
        result[dep_op] = std::list<int>();
      result[dep_op].push_back(op);
    }
  }
  return std::move(result);
}

std::map<int, std::list<int>> build_op_downstream_map(
    const std::vector<Instruction>& vec_instruction) {
  auto var2min_rw_op = std::map<
      int, std::list<int>>();  // # map from variable id to read / write op id.
  auto var2recent_write_op =
      std::map<int, int>();  // # map from variable to recent write op.
  auto op2dependences =
      std::map<int, std::set<int>>();  //# map from op to the dependence list,
                                       // op must run after the dependence.
  std::set<int>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    op2dependences[op_idx] = std::set<int>();
  }

  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         vec_instruction[op_idx].Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item :
         vec_instruction[op_idx].Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item :
         vec_instruction[op_idx].Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        update_var_min_rw_op(op2dependences, &var2min_rw_op, op_idx, var);
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         vec_instruction[op_idx].Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          update_var_min_rw_op(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }
  return std::move(get_downstream_map(op2dependences));
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
