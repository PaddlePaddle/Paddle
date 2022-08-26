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

#include "paddle/fluid/framework/details/nan_inf_utils.h"
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
    new_executor_serial_run,
    false,
    "Enable serial execution for standalone executor, used for debug.");

DECLARE_bool(use_mkldnn);
DECLARE_bool(check_nan_inf);

namespace paddle {
namespace framework {
namespace interpreter {

using VariableIdMap = std::map<std::string, std::vector<int>>;
constexpr size_t kPrepareWorkQueueIdx = 2;
const char blocking_queue_prefix[] = "lod_tensor_blocking_queue";

const std::vector<WorkQueueOptions> ConstructWorkQueueOptions(
    size_t host_num_threads, size_t device_num_threads, EventsWaiter* waiter) {
  std::vector<WorkQueueOptions> group_options;
  // for execute host Kernel
  group_options.emplace_back(/*name*/ "HostTasks",
                             /*num_threads*/ host_num_threads,
                             /*allow_spinning*/ true,
                             /*always_spinning*/ false,
                             /*track_task*/ false,
                             /*detached*/ true,
                             /*events_waiter*/ waiter);
  // for launch device Kernel
  group_options.emplace_back(/*name*/ "DeviceKernelLaunch",
                             /*num_threads*/ device_num_threads,
                             /*allow_spinning*/ true,
                             /*always_spinning*/ true,
                             /*track_task*/ false,
                             /*detached*/ true,
                             /*events_waiter*/ waiter);
  // for prepare deps and others
  group_options.emplace_back(/*name*/ "Prepare",
                             /*num_threads*/ 1,
                             /*allow_spinning*/ true,
                             /*always_spinning*/ false,
                             /*track_task*/ false,
                             /*detached*/ true,
                             /*events_waiter*/ waiter);
  return group_options;
}

AsyncWorkQueue::AsyncWorkQueue(size_t host_num_threads,
                               size_t device_num_threads,
                               EventsWaiter* waiter)
    : host_num_thread_(host_num_threads) {
  queue_group_ = CreateWorkQueueGroup(
      ConstructWorkQueueOptions(host_num_threads, device_num_threads, waiter));
}

void AsyncWorkQueue::AddTask(const OpFuncType& op_func_type,
                             std::function<void()> fn) {
  VLOG(4) << "Add task: " << static_cast<size_t>(op_func_type) << " ";
  // NOTE(zhiqiu): use the second queue of size of, so only one thread is used.
  if (FLAGS_new_executor_serial_run) {
    queue_group_->AddTask(static_cast<size_t>(OpFuncType::kQueueAsync),
                          std::move(fn));
  } else {
    queue_group_->AddTask(static_cast<size_t>(op_func_type), std::move(fn));
  }
}

std::future<std::unique_ptr<AtomicVectorSizeT>>
AsyncWorkQueue::PrepareAtomicDeps(const std::vector<size_t>& dependecy_count) {
  VLOG(4) << "PrepareAtomicDeps";
  return queue_group_->AddAwaitableTask(
      kPrepareWorkQueueIdx, interpreter::PrepareAtomicDeps, dependecy_count);
}

std::future<std::unique_ptr<AtomicVectorSizeT>>
AsyncWorkQueue::PrepareAtomicVarRef(
    const std::vector<VariableMetaInfo>& vec_meta_info) {
  VLOG(4) << "PrepareAtomicVarRef";
  return queue_group_->AddAwaitableTask(
      kPrepareWorkQueueIdx, interpreter::PrepareAtomicVarRef, vec_meta_info);
}

std::unique_ptr<AtomicVectorSizeT> PrepareAtomicDeps(
    const std::vector<size_t>& dependecy_count) {
  VLOG(4) << "PrepareAtomicDeps";

  auto op_deps = std::make_unique<AtomicVectorSizeT>(dependecy_count.size());
  for (size_t i = 0; i < dependecy_count.size(); ++i) {
    (*op_deps)[i] = dependecy_count[i];
  }
  VLOG(4) << "AtomicDeps:" << op_deps.get() << " " << op_deps->size();
  return op_deps;
}

std::unique_ptr<AtomicVectorSizeT> PrepareAtomicVarRef(
    const std::vector<VariableMetaInfo>& vec_meta_info) {
  VLOG(4) << "PrepareAtomicVarRef";
  auto var_ref = std::make_unique<AtomicVectorSizeT>(vec_meta_info.size());
  for (size_t i = 0; i < vec_meta_info.size(); ++i) {
    (*var_ref)[i] = vec_meta_info[i].var_ref_count_;
  }
  VLOG(4) << "AtomicVarRef:" << var_ref.get() << " " << var_ref->size();
  return var_ref;
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
                          VariableScope* var_scope,
                          bool use_local_scope) {
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
    var_scope->AddVar(var_name, var_desc);
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
    const VariableNameMap& var_name_map,
    VariableScope* var_scope,
    Scope* local_scope,
    bool enforce_exist = true) {
  VariableValueMap name2var;
  VariableIdMap name2id;
  for (auto& item : var_name_map) {
    std::vector<Variable*> vars;
    std::vector<int> ids;
    vars.reserve(item.second.size());

    for (auto& var_name : item.second) {
      if (!var_scope->HasVar(var_name)) {
        // Hot fix for variables used in dataloader, like
        // 'lod_tensor_blocking_queue_0' These variables may be created in
        // scope, and it is not existed as variable in program.
        if (var_name.find(blocking_queue_prefix) != std::string::npos &&
            local_scope->FindVar(var_name)) {
          var_scope->AddVar(var_name, nullptr);
        } else if (!enforce_exist) {
          // skip the non-exist variable: such as recurrent_grad
          VLOG(4) << var_name << " don't exist in variable scope, skip it!";
          continue;
        }
      }
      auto* var = local_scope->FindVar(var_name);
      auto var_id = var_scope->VarId(var_name);
      vars.push_back(var);
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
               platform::is_gpu_place(place)) {
      // when the Op that does not have GPUKernel is assigned to GPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      if (op_base->SupportGPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no CUDA implementation. It will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else if (op_device.find("npu") != std::string::npos &&
               platform::is_npu_place(place)) {
      // when the Op that does not have NPUKernel is assigned to NPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      if (op_base->SupportNPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no NPU implementation. It will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else if (op_device.find("xpu") != std::string::npos &&
               platform::is_xpu_place(place)) {
      // when the Op that does not have XPUKernel is assigned to XPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      if (op_base->SupportXPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no XPU implementation. It will be assigned to CPUPlace.";
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
                        OpFuncNode* op_func_node,
                        Scope* local_scope) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  // input, output is prepared. set the other attributes.
  op_func_node->operator_base_ = op_base;
  if (IsSupportedHetePlace(place)) {
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
                        const std::set<std::string>& skip_gc_vars,
                        std::vector<OpFuncNode>* vec_func_list,
                        VariableScope* var_scope,
                        bool use_local_scope) {
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();
  std::vector<std::unique_ptr<OperatorBase>>
      ops_unique;  // its elements will be moved to vec_func_list
  bool flag_log_is_printed = false;
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

    // Print new executor log if grad op is used.
    // It's only for test and will be removed later.
    if (!flag_log_is_printed && op->Type().find("_grad") != std::string::npos) {
      VLOG(0) << "Standalone Executor is Used.";
      flag_log_is_printed = true;
    }

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
        build_variable_map(inputs_names, var_scope, local_scope, enforce_exist);

    VariableValueMap outs_map;
    VariableIdMap outs_name2id;
    std::tie(outs_map, outs_name2id) = build_variable_map(
        outputs_names, var_scope, local_scope, enforce_exist);

    // step 1: build OpFuncNode
    OpFuncNode op_func_node;
    op_func_node.operator_base_ = ops[i];
    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;
    VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

#ifdef PADDLE_WITH_ASCEND_CL
    // NOTE(wangxi): nan/inf cannot be detected on NPU by checking the variable
    // values, but only through special `float_status` to checks whether
    // the operation is overflow. More about `float_status`, see:
    // https://gitee.com/ascend/modelzoo/issues/I3NF8V?from=project-issue
    if (FLAGS_check_nan_inf) {
      framework::details::NPUAllocAndClearFloatStatus(*op, *local_scope, place);
    }
#endif

    if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
      // op is not a operatorwithkernel, so direcly run OperatorBase::Run()
      deal_operator_base(place, var_scope, ops[i], &op_func_node, local_scope);
    } else {
      auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
          static_cast<const framework::OperatorWithKernel*>(op));
      // construct RuntimeContext and analysis KernelType
      RuntimeContext runtime_context({}, {});
      runtime_context.inputs.swap(ins_map);
      runtime_context.outputs.swap(outs_map);

      Scope scope, *runtime_scope = &scope;
      // NOTE(Ruibiao): We do not encourage directly using scope in OP kernel.
      // But some OPs do have such behavior (e.g., cinn_launch OP). Here special
      // treatment for them.
      if (op_with_kernel->Type() == "cinn_launch") {
        VLOG(6) << "OP(" << op_with_kernel->Type()
                << ") use scope in kernel, "
                   "so pass a real scope to "
                   "ExecutionContext";
        runtime_scope = local_scope;
      }

      auto& pool = platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(place);
      auto exec_ctx = ExecutionContext(
          *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
      auto expected_kernel_key =
          op_with_kernel->GetExpectedKernelType(exec_ctx);
      // change device by the device_guard()
      apply_device_guard(op, place, &expected_kernel_key);
      VLOG(4) << "expected_kernel_key : " << expected_kernel_key;

      // step 2. select op kernel
      auto run_phi_kernel = false;
      if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
              op_with_kernel->Type())) {
        auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
        auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;

        if (op_with_kernel->PhiKernel()->IsValid()) {
          run_phi_kernel = true;
        } else {
          if (!op_with_kernel->SupportsKernelType(expected_kernel_key)) {
            auto phi_cpu_kernel_key = FallBackToCpu(
                expected_kernel_key, phi_kernel_key, *op_with_kernel);
            op_with_kernel->ResetPhiKernel(
                new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                    phi_kernel_name, phi_cpu_kernel_key)));
            if (op_with_kernel->PhiKernel()->IsValid()) {
              VLOG(6) << "Static mode PrepareImpl - kernel name: "
                      << phi_kernel_name
                      << " | kernel key: " << phi_cpu_kernel_key
                      << " | kernel: " << *(op_with_kernel->PhiKernel());
              op_with_kernel->ResetKernelType(new OpKernelType(
                  TransPhiKernelKeyToOpKernelType(phi_cpu_kernel_key)));
              run_phi_kernel = true;
            }
          }
        }
      }
      if (!run_phi_kernel) {
        op_with_kernel->ChooseKernel(exec_ctx);
        op_func_node.kernel_func_ = *op_with_kernel->kernel_func();
      } else {
        op_func_node.phi_kernel_ = op_with_kernel->PhiKernel();
      }
      auto kernel_type = *(op_with_kernel->kernel_type());
      if (kernel_type.place_ != dev_ctx->GetPlace()) {
        dev_ctx = pool.Get(kernel_type.place_);
      }
      op_func_node.dev_ctx_ = dev_ctx;
      if (IsSupportedHetePlace(kernel_type.place_)) {
        op_func_node.type_ = OpFuncType::kQueueAsync;
      } else if (platform::is_cpu_place(kernel_type.place_)) {
        op_func_node.type_ = OpFuncType::kQueueSync;
      } else {
        PADDLE_THROW(platform::errors::Fatal("Unsupported current place %s",
                                             kernel_type.place_));
      }
      VLOG(3) << op_with_kernel->Type()
              << " : finally selected kernel_key: " << kernel_type;

      // step 3. data transform
      VariableValueMap& ins_map_temp = runtime_context.inputs;
      VariableValueMap& outs_map_temp = runtime_context.outputs;
      ApplyDataTransform(kernel_type,
                         place,
                         &ins_map_temp,
                         &outs_map_temp,
                         var_scope,
                         &op_func_node,
                         vec_func_list,
                         use_local_scope);

      // step 4. infershape, see OperatorWithKernel::RunImpl in operator.cc for
      // why.
      if (!(op->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        InterpretercoreInferShapeContext infer_shape_ctx(*op, runtime_context);
        // TODO(Aurelius84): In case of control flow ops, they are NOT
        // inheritted from OperatorWithKernel.
        op_with_kernel->Info().infer_shape_(&infer_shape_ctx);
      }

      // step 5. run kernel
      if (run_phi_kernel) {
        phi::KernelContext phi_kernel_context;
        op_with_kernel->BuildPhiKernelContext(
            runtime_context, dev_ctx, &phi_kernel_context);
        (*op_func_node.phi_kernel_)(&phi_kernel_context);
      } else {
        // the place of exec_ctx maybe has changed.
        op_func_node.kernel_func_(ExecutionContext(
            *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context));
      }

      // post-process grad_op.outputs if need cast complex grad into real
      // grad.
      // NOTE(Aurelius84): insert a transfer_dtype_op inplacely to cast it.
      if (framework::IsComplexType(kernel_type.data_type_)) {
        interpreter::HandleComplexGradToRealGrad(op_func_node,
                                                 place,
                                                 outputs_names,
                                                 &runtime_context.outputs,
                                                 var_scope,
                                                 vec_func_list,
                                                 local_scope);
      }
      if (!op_func_node.inplace_back_map.empty()) {
        auto& m = op_func_node.inplace_back_map;
        // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in
        // operator.cc
        for (auto& p : m) {
          auto* transformed_tensor =
              GetMutableLoDTensorOrSelectedRowsValueFromVar(
                  local_scope->FindVar(var_scope->GetNameById(p.first)));
          auto* original_tensor = GetMutableLoDTensorOrSelectedRowsValueFromVar(
              local_scope->FindVar(var_scope->GetNameById(p.second)));
          original_tensor->ShareDataWith(*transformed_tensor);
          VLOG(4) << "Transfer inplace variable back form "
                  << var_scope->GetNameById(p.first) << " to "
                  << var_scope->GetNameById(p.second);
        }
      }

      // for debug nan/inf
      if (FLAGS_check_nan_inf) {
        VLOG(4) << "Check nan/inf";
        framework::details::CheckOpHasNanOrInf(*op, *runtime_scope, place);
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
      auto* var = local_scope->FindVar(var_name);
      if (var == nullptr || skip_gc_vars.find(var_name) != skip_gc_vars.end()) {
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
            framework::ToTypeName(var->Type()),
            var_name));
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
  std::merge(
      first.begin(), first.end(), second.begin(), second.end(), out.begin());

  std::vector<size_t>::iterator it;
  it = std::unique(out.begin(), out.end());

  out.resize(std::distance(out.begin(), it));

  return out;
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
