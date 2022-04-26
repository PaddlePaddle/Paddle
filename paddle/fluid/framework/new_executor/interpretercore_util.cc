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
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

PADDLE_DEFINE_EXPORTED_bool(
    new_executor_sequential_run, false,
    "Enable sequential execution for standalone executor, used for debug");

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace framework {
namespace interpreter {

constexpr size_t kPrepareWorkQueueIdx = 2;

void AsyncWorkQueue::AddTask(const OpFuncType& op_func_type,
                             std::function<void()> fn) {
  VLOG(4) << "Add task: " << static_cast<size_t>(op_func_type) << " ";
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

void AsyncWorkQueue::PrepareAtomicDeps(
    const std::vector<size_t>& dependecy_count) {
  VLOG(4) << "PrepareAtomicDeps";
  atomic_deps_ =
      queue_group_->AddAwaitableTask(kPrepareWorkQueueIdx, [&dependecy_count] {
        auto op_deps = std::make_unique<std::vector<std::atomic<size_t>>>(
            dependecy_count.size());
        for (size_t i = 0; i < dependecy_count.size(); ++i) {
          (*op_deps)[i] = dependecy_count[i];
        }
        VLOG(4) << "AtomicDeps:" << op_deps.get() << " " << op_deps->size();
        return op_deps;
      });
}

void AsyncWorkQueue::PrepareAtomicVarRef(
    const std::vector<VariableMetaInfo>& vec_meta_info) {
  VLOG(4) << "PrepareAtomicVarRef";
  atomic_var_ref_ =
      queue_group_->AddAwaitableTask(kPrepareWorkQueueIdx, [&vec_meta_info] {
        auto var_ref = std::make_unique<std::vector<std::atomic<size_t>>>(
            vec_meta_info.size());
        for (size_t i = 0; i < vec_meta_info.size(); ++i) {
          (*var_ref)[i] = vec_meta_info[i].var_ref_count_;
        }
        VLOG(4) << "AtomicVarRef:" << var_ref.get() << " " << var_ref->size();
        return var_ref;
      });
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
    VLOG(4) << ops[op_idx].get()->Type() << " " << name;
  }
  VLOG(4) << "gc map size:" << result.size();
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
      // NOTE(zhiqiu): if var exists in scope and the type is right,
      // InitializeVariable will not create a new variable.
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

#ifdef PADDLE_WITH_MKLDNN
    if (FLAGS_use_mkldnn) {
      if (op->HasAttr("use_mkldnn")) {
        VLOG(4) << "Set use_mkldnn=True for " << op_base->Type();
        op_base->SetAttr("use_mkldnn", true);
      }
    }
#endif

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

#ifdef PADDLE_WITH_MKLDNN
  platform::RegisterModelLayout(ops_unique, place);
#endif

  // its elements will be moved to vec_func_list
  std::vector<std::shared_ptr<OperatorBase>> ops;
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
    VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

    if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
      // op is not a operatorwithkernel, so direcly run OperatorBase::Run()
      deal_operator_base(place, var_scope, ops[i], &op_func_node, local_scope);
      VLOG(4) << "End run " << place << " "
              << op_func_node.operator_base_->DebugStringEx(local_scope);
    } else {
      auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
          static_cast<const framework::OperatorWithKernel*>(op));
      // construct RuntimeContext and analysis KernelType
      RuntimeContext runtime_context({}, {});
      runtime_context.inputs.swap(ins_map);
      runtime_context.outputs.swap(outs_map);

      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(place);
      Scope scope;
      Scope* runtime_scope = &scope;
      // NOTE(Ruibiao): We do not encourage directly using scope in OP kernel.
      // But some OPs do have such behavior (e.g., cinn_launch OP). Here special
      // treatment for them.
      if (op_with_kernel->Type() == "cinn_launch") {
        VLOG(6) << "OP(" << op_with_kernel->Type() << ") use scope in kernel, "
                                                      "so pass a real scope to "
                                                      "ExecutionContext";
        runtime_scope = local_scope;
      }

      auto expected_kernel_key = op_with_kernel->GetExpectedKernelType(
          ExecutionContext(*op, *runtime_scope, *dev_ctx, runtime_context));
      op_with_kernel->ResetKernelType(new OpKernelType(expected_kernel_key));

      // change device by the device_guard()
      apply_device_guard(op, place, &expected_kernel_key);
      VLOG(3) << "expected_kernel_key : " << expected_kernel_key;

      // step 3. apply data transforms and insert data transfer ops
      VariableValueMap& ins_map_temp = runtime_context.inputs;
      VariableValueMap& outs_map_temp = runtime_context.outputs;

      // NOTE(zhiqiu): op_func_node->operator_base_ maybe changed in
      // ApplyDataTransform
      ApplyDataTransform(expected_kernel_key, place, &ins_map_temp,
                         &outs_map_temp, var_scope, &op_func_node,
                         vec_func_list, use_local_scope);
      op_with_kernel = const_cast<framework::OperatorWithKernel*>(
          static_cast<const framework::OperatorWithKernel*>(
              op_func_node.operator_base_.get()));

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

      // see OperatorWithKernel::RunImpl in operator.cc for why
      if (!(op->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        InterpretercoreInferShapeContext infer_shape_ctx(*op, runtime_context);
        // TODO(Aurelius84): In case of control flow ops, they are NOT
        // inheritted from OperatorWithKernel.
        op_with_kernel->Info().infer_shape_(&infer_shape_ctx);
      }

      auto exec_ctx = ExecutionContext(*op_with_kernel, *runtime_scope,
                                       *dev_ctx, runtime_context);

      auto run_phi_kernel = false;
      if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
              op_with_kernel->Type())) {
        auto pt_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
        auto pt_kernel_name = op_with_kernel->PhiKernelSignature()->name;

        if (op_with_kernel->PhiKernel()->IsValid()) {
          run_phi_kernel = true;
        } else {
          auto kernels_iter = all_op_kernels.find(op_with_kernel->Type());
          if (kernels_iter == all_op_kernels.end() ||
              kernels_iter->second.find(expected_kernel_key) ==
                  kernels_iter->second.end()) {
            auto pt_cpu_kernel_key = FallBackToCpu(
                expected_kernel_key, pt_kernel_key, *op_with_kernel);
            op_with_kernel->ResetPhiKernel(
                new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                    pt_kernel_name, pt_cpu_kernel_key)));
            if (op_with_kernel->PhiKernel()->IsValid()) {
              VLOG(6) << "Static mode PrepareImpl - kernel name: "
                      << pt_kernel_name
                      << " | kernel key: " << pt_cpu_kernel_key
                      << " | kernel: " << *(op_with_kernel->PhiKernel());
              run_phi_kernel = true;
            }
          }
        }
      }
      VLOG(3) << op_with_kernel->Type()
              << " : expected_kernel_key : " << expected_kernel_key;
      if (run_phi_kernel) {
        phi::KernelContext pt_kernel_context;
        op_with_kernel->BuildPhiKernelContext(runtime_context, dev_ctx,
                                              &pt_kernel_context);
        op_func_node.pt_kernel_ = op_with_kernel->PhiKernel();
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
      if (!op_func_node.inplace_back_map.empty()) {
        auto& m = op_func_node.inplace_back_map;
        // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in operator.cc
        for (auto& p : m) {
          auto* transformed_tensor =
              GetMutableLoDTensorOrSelectedRowsValueFromVar(
                  var_scope->Var(p.first));
          auto* original_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
              var_scope->Var(p.second));
          original_tensor->ShareDataWith(*transformed_tensor);
          VLOG(4) << "Transfer inplace variable back form "
                  << var_scope->GetNameById(p.first) << " to "
                  << var_scope->GetNameById(p.second);
        }
      }
    }

    VLOG(4) << "End run " << place << " "
            << op_func_node.operator_base_->DebugStringEx(local_scope);

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
      } else if (var->IsType<phi::SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
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
    const std::map<int, std::set<int>>& op2dependences,
    std::vector<std::vector<bool>>* op_happens_before) {
  // step1: convert op2dependences to downstream_map directly
  // op2dependences is op -> it's dependences.
  // we want to get op -> [next ops] map,
  // where ops is the next instruction of op.
  std::map<int, std::list<int>> downstream;
  for (auto& item : op2dependences) {
    int op = item.first;
    for (auto dep_op : item.second) {
      if (downstream.find(dep_op) == downstream.end())
        downstream[dep_op] = std::list<int>();
      downstream[dep_op].push_back(op);
    }
  }

  auto downstream_map_to_str = [&]() -> std::string {
    std::ostringstream oss;
    for (auto pair : downstream) {
      oss << pair.first << " -> ";
      std::copy(pair.second.begin(), pair.second.end(),
                std::ostream_iterator<int>(oss, " "));
      oss << std::endl;
    }
    return oss.str();
  };

  auto downstream_map_count = [&]() -> size_t {
    size_t count = 0;
    for (auto pair : downstream) {
      count += pair.second.size();
    }
    return count;
  };

  VLOG(6) << "downstream count: " << downstream_map_count();
  VLOG(6) << "downstream_map: " << std::endl << downstream_map_to_str();

  // step2: remove unneccessary downstream ops
  // for example, a->b->c
  // a: b, c
  // b: c
  // =>
  // a: b
  // b: c

  // NOTE(zhiqiu): the size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  auto op_num = op2dependences.size();
  // happens_before[i][j] means i should be executed before j
  op_happens_before->resize(op_num);
  for (size_t i = 0; i < op_num; ++i) {
    (*op_happens_before)[i].resize(op_num);
    std::fill((*op_happens_before)[i].begin(), (*op_happens_before)[i].end(),
              false);
  }

  // bfs to get all next ops
  auto bfs = [&](size_t op_idx) {
    std::queue<size_t> q;
    std::vector<bool> visited(op_num, false);
    q.push(op_idx);
    while (!q.empty()) {
      size_t op = q.front();
      q.pop();
      visited[op] = true;
      if (!downstream.count(op)) {
        continue;
      }
      for (auto next : downstream[op]) {
        if (!visited[next]) {
          PADDLE_ENFORCE_EQ((*op_happens_before)[next][op_idx], false,
                            paddle::platform::errors::AlreadyExists(
                                "There exists circle in graph, expected "
                                "%d->%d, but already got %d->%d",
                                op_idx, next, next, op_idx));
          (*op_happens_before)[op_idx][next] = true;
          VLOG(8) << "happens before: " << op_idx << " " << next;
          q.push(next);
        }
      }
    }
  };

  for (size_t i = 0; i < op_num; ++i) {
    bfs(i);
  }

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  for (size_t i = 0; i < op_num; ++i) {
    std::list<int> minumum_nexts;
    for (size_t item : downstream[i]) {
      bool not_after_any = true;
      // find the op that is not executed after any
      for (size_t other_item : downstream[i]) {
        if ((*op_happens_before)[other_item][item]) {
          VLOG(8) << "happens_before: " << other_item << "->" << item
                  << ", so skip " << item;
          not_after_any = false;
          break;
        }
      }
      if (not_after_any) {
        VLOG(8) << "downstream op of " << i << ": " << item;
        minumum_nexts.push_back(item);
      }
    }
    downstream[i] = minumum_nexts;
  }
  VLOG(6) << "downstream count: " << downstream_map_count();
  VLOG(6) << "downstream_map: " << std::endl << downstream_map_to_str();

  return downstream;
}

std::map<int, std::list<int>> build_op_downstream_map(
    const std::vector<Instruction>& vec_instruction,
    std::vector<std::vector<bool>>* op_happens_before) {
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

    // NOTE(zhiqiu): The inplace op with `transfer` also changes
    // original output after that so add original output as well
    // original: a->op->a
    // after: a->data_transfer->a'->op->a'->transfer_back->a
    // which means op writes a and a'
    if (!vec_instruction[op_idx].InplaceBackMap().empty()) {
      auto& m = vec_instruction[op_idx].InplaceBackMap();
      for (auto& p : m) {
        auto var = p.second;
        var2recent_write_op[var] = op_idx;
        // var in input list and in output list, so remove it.
        if (remove_duplicate.count(var) == 0) {
          update_var_min_rw_op(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // add dependences for random op, make sure that the random op is scheduled
  // sequentially
  const std::set<std::string> random_op_set = {
      "bernoulli", "poisson", "multinomial", "gaussian_random",
      "truncated_gaussian_random", "uniform_random", "randint", "randperm",
      "exponential",
      "sampling_id"
      "dropout",
      "class_center_sample",
  };

  int dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    if (random_op_set.count(vec_instruction[op_idx].OpBase()->Type())) {
      if (dependence_op_idx != -1) {
        op2dependences[op_idx].insert(dependence_op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }

  // add dependency for communication op
  auto is_comm_op = [](std::string op) -> bool {
    const std::set<std::string> special_comm_op_set = {
        "send", "recv", "send_v2", "recv_v2",
    };
    const std::string communication_op_prefix = "c_";
    if (op.find(communication_op_prefix) != std::string::npos ||
        special_comm_op_set.count(op)) {
      return true;
    }
    return false;
  };

  dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    if (is_comm_op(vec_instruction[op_idx].OpBase()->Type())) {
      if (dependence_op_idx != -1) {
        op2dependences[op_idx].insert(dependence_op_idx);
        VLOG(4) << "Add depend from "
                << vec_instruction[dependence_op_idx].OpBase()->Type() << " to "
                << vec_instruction[op_idx].OpBase()->Type();
      }
      dependence_op_idx = op_idx;
    }
  }

  // TODO(zhiqiu): there still some cases not handled
  // add dependency for c_sync_comm_stream

  // in program, we can add only one c_sync_comm_stream to sync all
  // communication ops.
  // c_allreduce_sum(a)
  // c_allreduce_sum(b)
  // c_allreduce_sum(c)
  // c_sync_comm_stream(a)
  const std::string kSyncComm = "c_sync_comm_stream";
  dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    if (vec_instruction[op_idx].OpBase()->Type() == kSyncComm) {
      dependence_op_idx = op_idx;
    } else {
      if (dependence_op_idx != -1) {
        VLOG(4) << "Add depend from "
                << vec_instruction[dependence_op_idx].OpBase()->Type() << " to "
                << vec_instruction[op_idx].OpBase()->Type();
        op2dependences[op_idx].insert(dependence_op_idx);
      }
    }
  }

  // add dependency for coalesce_tensor
  const std::string kCoalesceTensor = "coalesce_tensor";
  for (size_t op_idx = 0; op_idx < vec_instruction.size(); ++op_idx) {
    if (vec_instruction[op_idx].OpBase()->Type() == kCoalesceTensor) {
      VLOG(4) << "Add depend for " << kCoalesceTensor << " " << op_idx;
      auto fused_out = vec_instruction[op_idx].Outputs().at("FusedOutput")[0];
      auto outputs = vec_instruction[op_idx].Outputs().at("Output");

      auto is_read = [](const Instruction& inst, int var_id) -> bool {
        for (auto pair : inst.Inputs()) {
          for (auto item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      auto is_write = [](const Instruction& inst, int var_id) -> bool {
        for (auto pair : inst.Outputs()) {
          for (auto item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      // find first op that reads fused_out
      auto first_read_fused_out_op = -1;
      for (auto j = op_idx + 1; j < vec_instruction.size(); ++j) {
        if (is_read(vec_instruction[j], fused_out)) {
          first_read_fused_out_op = j;
          break;
        }
      }

      if (UNLIKELY(first_read_fused_out_op == -1)) {
        VLOG(4) << "No op read FusedOutput";
        continue;
      }

      // find ops that write 'outputs' between (op_index,
      // first_read_fused_out_op)
      // add depend: them->first_read_fused_out_op
      for (auto j = op_idx + 1;
           j < static_cast<size_t>(first_read_fused_out_op); ++j) {
        for (auto var_id : outputs) {
          if (is_write(vec_instruction[j], var_id)) {
            op2dependences[first_read_fused_out_op].insert(j);
            VLOG(4) << j << " -> " << first_read_fused_out_op;
            VLOG(4)
                << "Add depend from " << vec_instruction[j].OpBase()->Type()
                << " to "
                << vec_instruction[first_read_fused_out_op].OpBase()->Type();
          }
        }
      }

      // find first op read 'outputs' between (first_read_fused_out_op, end)
      // add depned:  first_read_fused_out_op -> first op that reads 'outputs'

      // special case for consecutive communication ops, for example,
      // FusedOutput = c_sync_calc_stream(FusedOutput)
      // FusedOutput= c_allreduce_sum(FusedOutput)
      // FusedOutput = c_sync_comm_stream(FusedOutput)
      // we should take the last one to add depned instead of
      // 'first_read_fused_out_op'
      size_t target = first_read_fused_out_op;
      for (size_t j = first_read_fused_out_op + 1; j < vec_instruction.size();
           ++j) {
        if (j == target + 1 &&
            is_comm_op(vec_instruction[target].OpBase()->Type()) &&
            is_comm_op(vec_instruction[j].OpBase()->Type())) {
          VLOG(4) << "Found consecutive communication ops, "
                  << vec_instruction[target].OpBase()->Type() << " -> "
                  << vec_instruction[j].OpBase()->Type();
          target = j;
          continue;
        }

        for (auto var_id : outputs) {
          if (is_read(vec_instruction[j], var_id)) {
            op2dependences[j].insert(target);
            VLOG(4) << target << " -> " << j;
            VLOG(4) << "Add depend from "
                    << vec_instruction[target].OpBase()->Type() << " to "
                    << vec_instruction[j].OpBase()->Type();
          }
        }
      }
    }
  }
  for (auto pair : op2dependences) {
    std::ostringstream oss;
    oss << pair.first << " Depends on " << pair.second.size() << " ops: ";
    std::copy(pair.second.begin(), pair.second.end(),
              std::ostream_iterator<int>(oss, " "));
    VLOG(10) << oss.str();
  }
  return get_downstream_map(op2dependences, op_happens_before);
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
