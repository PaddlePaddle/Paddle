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

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"

#include <algorithm>

#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/data_transfer.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"
#include "paddle/fluid/memory/stats.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/pylayer_op_helper.h"
#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif
PADDLE_DEFINE_EXPORTED_bool(
    new_executor_log_memory_stats,
    false,
    "Log memory stats after each op runs, just used for debug.");

PHI_DECLARE_bool(use_mkldnn);
PHI_DECLARE_bool(check_nan_inf);

namespace paddle {
namespace framework {
namespace interpreter {

using VariableIdMap = std::map<std::string, std::vector<int>>;

// NOTE(Ruibiao): SingleStreamGuard make some multi-strem op (i.e.,
// c_allreduce_sum) run in single stream. It is dedicated to BuildOpFuncList
// which run kernel without stream synchronization.
class SingleStreamGuard {
 public:
  explicit SingleStreamGuard(std::shared_ptr<OperatorBase>& op) : op_(op) {
    if (op_->Type() == "c_allreduce_sum" &&
        op_->Attr<bool>("use_calc_stream") == false) {
      VLOG(6) << "Set c_allredce_sum's attr use_calc_stream to true";
      op_->SetAttr("use_calc_stream", true);
      is_changed = true;
    }
  }

  ~SingleStreamGuard() {  // NOLINT
    if (!is_changed) {
      return;
    }

    if (op_->Type() == "c_allreduce_sum") {
      op_->SetAttr("use_calc_stream", false);
      VLOG(6) << "Set c_allredce_sum's attr use_calc_stream to false";
    }
  }

  DISABLE_COPY_AND_ASSIGN(SingleStreamGuard);

 private:
  bool is_changed{false};
  std::shared_ptr<OperatorBase> op_;
};

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
  // queue_idx=0 : kCpuSync or kGpuSync
  // queue_idx=1 : kGPUAsync
  queue_group_->AddTask(op_func_type == OpFuncType::kGpuAsync, std::move(fn));
}

bool IsCommunicationOp(const OperatorBase* op) {
  const std::string& op_name = op->Type();
  const std::set<std::string> special_comm_op_set = {
      "send",
      "recv",
      "send_v2",
      "recv_v2",
  };
  const std::string communication_op_prefix = "c_";
  if (op_name.find(communication_op_prefix) != std::string::npos ||
      special_comm_op_set.count(op_name)) {
    return true;
  }
  if (op->HasAttr("ring_id")) {
    return true;
  }
  return false;
}

bool IsCommunicationOp(const Instruction& instr) {
  if (!instr.OpBaseValid()) {
    return false;
  }
  return IsCommunicationOp(instr.OpBase());
}

bool IsCpuOp(const Instruction& instr) {
  return platform::is_cpu_place(instr.DeviceContext().GetPlace());
}

bool IsCpuOp(Instruction* instr) {
  return platform::is_cpu_place(instr->DeviceContext().GetPlace());
}

bool IsCpuOp(const paddle::framework::InstructionBase& instr) {
  return platform::is_cpu_place(instr.DeviceContext().GetPlace());
}

bool IsCpuOp(paddle::framework::InstructionBase* instr) {
  return platform::is_cpu_place(instr->DeviceContext().GetPlace());
}

bool IsGradOp(const std::string& op_name) {
  return paddle::string::ends_with(op_name, "_grad");
}

bool IsSupportedHeterPlace(const phi::Place& place) {
  return platform::is_gpu_place(place) || platform::is_xpu_place(place) ||
         platform::is_ipu_place(place) || platform::is_custom_place(place);
}

bool IsMemcpyD2H(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyD2H;
}

bool IsMemcpyH2D(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyH2D;
}

bool IsMemcpyH2D(Instruction* instr) {
  return instr->OpBase()->Type() == kMemcpyH2D;
}

bool IsMemcpyH2D(paddle::framework::InstructionBase* instr) {
  return instr->Name() == "pd_op.memcpy_h2d";
}

bool IsMemcpyOp(const Instruction& instr) {
  return IsMemcpyD2H(instr) || IsMemcpyH2D(instr);
}

void AddFetch(const std::vector<std::string>& fetch_names,
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
GetUnusedVars(const BlockDesc& block,
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
    auto op = ops[op_idx].get();
    result[op].emplace_back(name);
    VLOG(4) << op->Type() << " " << name;
  }
  VLOG(4) << "gc map size:" << result.size();
  return result;
}

OpFuncType AnalyseOpFuncType(const OpFuncNode& op_func_node,
                             const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(IsSupportedHeterPlace(place),
                    true,
                    phi::errors::Fatal("Unsupported current place %s", place));

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  std::shared_ptr<OperatorBase> op = op_func_node.operator_base_;
  if (op->Type() == kCoalesceTensor &&
      (!platform::is_xpu_place(place) ||
       op->Attr<bool>("persist_output") == false) &&
      op->Attr<bool>("set_constant") == false &&
      op->Attr<bool>("copy_data") == false) {
    return OpFuncType::kGpuSync;
  }

  // for memcpy explicitly called by user
  if (platform::is_gpu_place(place) && op->Type() == interpreter::kMemcpyD2H) {
    return OpFuncType::kGpuSync;
  }

  if (op->Type() == "shape") {
    return OpFuncType::kGpuSync;
  }
  return OpFuncType::kGpuAsync;
}

void CreateAllOps(const framework::BlockDesc& block,
                  std::vector<std::unique_ptr<OperatorBase>>* ops) {
  for (auto& op : block.AllOps()) {
    auto op_type = op->Type();
    VLOG(8) << "CreateOp from : " << op_type;

    auto& info = OpInfoMap::Instance().Get(op_type);

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();

    AttributeMap op_attr_map = op->GetAttrMap();
    AttributeMap op_runtime_attr_map = op->GetRuntimeAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }

    const auto& extra_attr_checkers =
        operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(op_type);
    for (const auto& checker : extra_attr_checkers) {
      checker(&op_runtime_attr_map, false);
    }

    auto op_base =
        info.Creator()(op_type, inputs_names, outputs_names, op_attr_map);
    op_base->SetRuntimeAttributeMap(op_runtime_attr_map);

#ifdef PADDLE_WITH_DNNL
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

std::tuple<VariableValueMap, VariableIdMap> BuildVariableMap(
    const VariableNameMap& var_name_map,
    VariableScope* var_scope,
    Scope* local_scope,
    bool find_var_recursively = false,
    bool allow_var_not_in_scope = false) {
  VariableValueMap name2var;
  VariableIdMap name2id;
  for (auto& item : var_name_map) {
    std::vector<Variable*> vars;
    std::vector<int> ids;
    vars.reserve(item.second.size());

    for (auto& var_name : item.second) {
      auto* var = local_scope->FindVar(var_name);

      if (!var_scope->HasVar(var_name)) {
        if (find_var_recursively && var) {
          VLOG(3) << "Add " << var_name << " to var_scope";
          var_scope->AddVar(var_name, nullptr);
        } else if (allow_var_not_in_scope) {
          VLOG(4) << var_name << " don't exist in variable scope, skip it!";
          continue;
        }
      }
      auto var_id = var_scope->VarId(var_name);
      vars.push_back(var);
      ids.push_back(var_id);
    }
    name2var[item.first] = std::move(vars);
    name2id[item.first] = std::move(ids);
  }
  return std::make_tuple(name2var, name2id);
}

void ApplyDeviceGuard(const OperatorBase* op_base,
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
    } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
      bool is_custom_device_op = false;
      for (auto dev_type : device_types) {
        if (op_device.find(dev_type) != std::string::npos) {
          is_custom_device_op = true;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(
          is_custom_device_op,
          true,
          phi::errors::Unimplemented(
              "Unsupported current device %s with Paddle CustomDevice ",
              op_device));
#else
      VLOG(1) << string::Sprintf(
          "Cannot use get_all_custom_device_type because you have installed"
          "CPU/GPU version PaddlePaddle.\n"
          "If you want to use get_all_custom_device_type, please try to "
          "install CustomDevice version "
          "PaddlePaddle by: pip install paddlepaddle\n");
#endif
      if (op_base->SupportCustomDevice()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = platform::CPUPlace();
        LOG_FIRST_N(WARNING, 1) << "Op(" << op_base->Type()
                                << ") has no Custom Place implementation. It "
                                   "will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else {
      PADDLE_THROW(
          platform::errors::Fatal("Unsupported current place %s", op_device));
    }
  }
}

platform::DeviceContext* ConstructDeviceContext(const OperatorBase* op,
                                                const platform::Place& place) {
  auto& pool = platform::DeviceContextPool::Instance();
  auto* default_dev_ctx = pool.Get(place);

  // Replace the default_dev_ctx according to dst_place_type for memcpy op if
  // needed.

  // NOTE(liudongxue01):
  //  Please apply the following logic in other Executor/Interpreter modules
  //  likewise.
  //

  // NOTE(liudongxue01):
  // The following code aims to fixup the memcpy kernel which does not handle
  // some rare case. The case is:
  //  1. The default place in the current execution context is not CUDAPlace,
  //  such as CPUPlace,
  //  2. The dst_place_type is 1 which means CUDAPlace,
  //  3. The expected result place is CUDAPlace but the actual result is
  //  CPUPlace.
  // When the default place is CPUPlace, we call the tensor.cuda() would
  // simply hit such case.
  //
  // Q: Why we do not add such logic in the memcpy kernel?
  // A: (1) To fixup the memcpy kernel, we need to construct a CUDAPlace() and
  //    corresponding DeviceContext instance which used by the phi::Copy(...)
  //    api to perform the real memcpy action. (2) We should not access the
  //    singleton of the DeviceContextPool object in the PHI framework which
  //    is designed as a standalone module and all context data should passed
  //    into the kernel API through arguments. (3) So we have no way to
  //    construct a CUDAPlace() in the memcpy kernel and then pass it
  //     to the phi::Copy(...) api.
  if (!platform::is_gpu_place(place)) {
    const auto& op_type = op->Type();
    if (op_type == "memcpy") {
      int dst_place_type = op->Attr<int>("dst_place_type");
      if (dst_place_type == 1) {  // 1 : CUDAPlace
        auto dev_ctx = pool.Get(paddle::DefaultGPUPlace());
        VLOG(4) << "Change the device context for memcpy OP: ("
                << default_dev_ctx->type_info().name() << ") -> ("
                << dev_ctx->type_info().name() << ")";
        return dev_ctx;
      }
    }
  }

  return default_dev_ctx;
}

void HandleOperatorBase(const platform::Place& place,
                        std::shared_ptr<OperatorBase> op,
                        OpFuncNode* op_func_node,
                        Scope* scope,
                        bool static_build) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  // input, output is prepared. set the other attributes.
  op_func_node->operator_base_ = op;
  op_func_node->type_ = AnalyseOpFuncType(*op_func_node, place);
  op_func_node->kernel_func_ = nullptr;
  if (static_build) {
    if (OperatorBasesMustRunInStaticBuild.count(op->Type())) {
      op->Run(*scope, place);
    }
    FakeInitializeOutputsForOperatorBase(*op, place, scope);
  } else {
    op->Run(*scope, place);  // Run without data transformer.
  }

  op_func_node->dev_ctx_ = dev_ctx;
}

void BuildOpFuncList(const platform::Place& place,
                     const framework::BlockDesc& block,
                     const std::set<std::string>& skip_gc_vars,
                     std::vector<OpFuncNode>* vec_func_list,
                     VariableScope* var_scope,
                     const ExecutionConfig& execution_config,
                     bool use_local_scope,
                     bool static_build) {
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();
  std::vector<std::unique_ptr<OperatorBase>>
      ops_unique;  // its elements will be moved to vec_func_list
  // Step 1: create all ops for current block.
  CreateAllOps(block, &ops_unique);

  VLOG(1) << "Static build: " << static_build;

  if (!execution_config.used_for_jit) {
    // If gc is enabled and block size > 1
    const ProgramDesc& main_program = *block.Program();
    operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
        main_program, block.ID(), ops_unique);
    operators::PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
        main_program, block.ID(), ops_unique);
    operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
        main_program, block.ID(), ops_unique);
    operators::PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
        main_program, block.ID(), ops_unique);
  }

#ifdef PADDLE_WITH_DNNL
  platform::RegisterModelLayout(ops_unique, place);
#endif
  // its elements will be moved to vec_func_list
  std::vector<std::shared_ptr<OperatorBase>> ops;
  for (auto& op_unique : ops_unique) {
    ops.emplace_back(std::move(op_unique));
  }
  auto unused_var_map = GetUnusedVars(block, ops);

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  bool flag_log_is_printed = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto op = ops[i].get();
    const std::string& op_type = op->Type();

    VLOG(6) << "Build OpFuncNode from : " << op_type;

    // Print new executor log if grad op is used.
    // It's only for test and will be removed later.
    if (!flag_log_is_printed && op_type.find("_grad") != std::string::npos) {
      LOG_FIRST_N(INFO, 1) << "Standalone Executor is Used.";
      flag_log_is_printed = true;
    }

    // Hot fix for variables used in dataloader, like
    // 'lod_tensor_blocking_queue_0'. These variables may be created in scope,
    // and it is not existed as variable in program.
    const std::set<std::string> ops_with_var_not_in_program = {
        "create_py_reader"};
    const std::set<std::string> ops_with_var_not_in_scope = {
        "conditional_block",
        "conditional_block_grad",
        "pylayer",
        "pylayer_grad"
        "recurrent_grad",
        "rnn_memory_helper",
        "rnn_memory_helper_grad",
        "while",
        "while_grad"};
    bool allow_var_not_in_program = ops_with_var_not_in_program.count(op_type);
    bool allow_var_not_in_scope = ops_with_var_not_in_scope.count(op_type);

    // ops in the control flow block may not find its inputs or outputs
    // in VariableScope of the sub-block, so we need search it in parent scope.

    framework::VariableNameMap& input_name_map = op->Inputs();
    VariableValueMap ins_map;
    VariableIdMap ins_name2id;
    std::tie(ins_map, ins_name2id) = BuildVariableMap(
        input_name_map,
        var_scope,
        local_scope,
        execution_config.used_for_control_flow_op || allow_var_not_in_program,
        allow_var_not_in_scope);

    framework::VariableNameMap& output_name_map = op->Outputs();
    VariableValueMap outs_map;
    VariableIdMap outs_name2id;
    std::tie(outs_map, outs_name2id) =
        BuildVariableMap(output_name_map,
                         var_scope,
                         local_scope,
                         execution_config.used_for_control_flow_op,
                         allow_var_not_in_scope);

    // step 1: build OpFuncNode
    OpFuncNode op_func_node;
    op_func_node.operator_base_ = ops[i];
    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;

    const OperatorDistAttr* dist_attr =
        block.Op(static_cast<int>(i))->DistAttr();
    if (dist_attr) {
      if (dist_attr->execution_stream() !=
          distributed::auto_parallel::kDefault) {
        op_func_node.execution_stream_ = dist_attr->execution_stream();
      }
      op_func_node.stream_priority_ = dist_attr->stream_priority();
      op_func_node.scheduling_priority_ = dist_attr->scheduling_priority();
      // set mannual event information
      op_func_node.force_record_event_ = dist_attr->force_record_event();
      op_func_node.events_to_wait_ = dist_attr->events_to_wait();
      op_func_node.event_to_record_ = dist_attr->event_to_record();
    } else {
      if (interpreter::IsCommunicationOp(op)) {
        // NOTE(Ruibiao): Dispatching computation before communication improves
        // multi-stream overlap when the time cost of communication less than
        // that of the calculation (e.g., ResNet50_bs128_pure_fp16 N4C32
        // training).
        op_func_node.scheduling_priority_ = 1;
      }
    }

    VLOG(6) << op_type
            << " : [execution_stream, stream_priority, scheduling_priority] = ["
            << op_func_node.execution_stream_ << ", "
            << op_func_node.stream_priority_ << ", "
            << op_func_node.scheduling_priority_ << "]";

    SingleStreamGuard single_stream_guard(ops[i]);

    VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

    try {
      if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
        VLOG(4) << "HandleOperatorBase";
        // op is not a operatorwithkernel, so direcly run OperatorBase::Run()
        HandleOperatorBase(
            place, ops[i], &op_func_node, local_scope, static_build);
        vec_func_list->emplace_back(op_func_node);
      } else {
        VLOG(4) << "OP is not null";
        auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
            static_cast<const framework::OperatorWithKernel*>(op));
        VLOG(4) << "get op_with_kernel";
        // construct RuntimeContext and analysis KernelType
        RuntimeContext runtime_context({}, {});
        runtime_context.inputs.swap(ins_map);
        runtime_context.outputs.swap(outs_map);
        VLOG(4) << "get RuntimeContext";

        Scope scope, *runtime_scope = &scope;
        // NOTE(Ruibiao): We do not encourage directly using scope in OP kernel.
        // But some OPs do have such behavior (e.g., cinn_launch OP). Here
        // special treatment for them.
        if (op_with_kernel->Type() == "cinn_launch" ||
            op_with_kernel->Type() == "cinn_instruction_run") {
          VLOG(6) << "OP(" << op_with_kernel->Type()
                  << ") use scope in kernel, "
                     "so pass a real scope to "
                     "ExecutionContext";
          runtime_scope = local_scope;
        }

        // construct the device context
        auto* dev_ctx = ConstructDeviceContext(op, place);

        SetDeviceCommContext(op, dev_ctx);
        auto exec_ctx = ExecutionContext(
            *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
        auto expected_kernel_key = framework::TransPhiKernelKeyToOpKernelType(
            op_with_kernel->GetExpectedKernelType(exec_ctx));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        if (op_with_kernel->CanCUDNNBeUsed(exec_ctx,
                                           expected_kernel_key.data_type_)) {
          expected_kernel_key.library_type_ = framework::LibraryType::kCUDNN;
        }
#endif
        VLOG(4) << "expected_kernel_key : " << expected_kernel_key;
        // change device by the device_guard()
        ApplyDeviceGuard(op, place, &expected_kernel_key);
        if (platform::places_are_same_class(exec_ctx.GetPlace(),
                                            expected_kernel_key.place_)) {
          expected_kernel_key.place_ = exec_ctx.GetPlace();
        }

        // step 2. select op kernel
        auto run_phi_kernel = false;
        if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                op_with_kernel->Type())) {
          auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
          auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;
          bool in_custom_back_list = false;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
          in_custom_back_list =
              phi::backends::custom_device::is_in_custom_black_list(
                  phi_kernel_name);
#endif
          if (op_with_kernel->PhiKernel()->IsValid() && !in_custom_back_list) {
            run_phi_kernel = true;
          } else {
            if ((!op_with_kernel->SupportsKernelType(expected_kernel_key,
                                                     exec_ctx)) ||
                in_custom_back_list) {
              std::string info = in_custom_back_list ? "fluid in black list "
                                                     : "fluid missing ";
              VLOG(3) << info << phi_kernel_key
                      << " kernel: " << phi_kernel_name;
              auto phi_cpu_kernel_key =
                  FallBackToCpu(phi_kernel_key, *op_with_kernel);
              op_with_kernel->ResetPhiKernel(
                  new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                      phi_kernel_name, phi_cpu_kernel_key)));
              if (op_with_kernel->PhiKernel()->IsValid()) {
                VLOG(6) << "Static graph mode PrepareImpl - kernel name: "
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

        VLOG(4) << "if run phi kernel? : " << run_phi_kernel;
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
        op_func_node.type_ =
            AnalyseOpFuncType(op_func_node, kernel_type.place_);

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
                           use_local_scope,
                           static_build);
        VLOG(4) << "apply data transform done. ";

        // step 4. infershape
        if (!static_build) {
          VLOG(4) << "infer shape";
          // see kAllKernelsMustComputeRuntimeShape in operator.h for why
          if (!(op->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
                op->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
            RuntimeInferShapeContext infer_shape_ctx(*op, runtime_context);
            // TODO(Aurelius84): In case of control flow ops, they are NOT
            // inheritted from OperatorWithKernel.
            op_with_kernel->Info().infer_shape_(&infer_shape_ctx);
          }
        }

        // step 5. run kernel
        if (run_phi_kernel &&
            op_func_node.phi_kernel_->GetKernelRegisteredType() ==
                phi::KernelRegisteredType::FUNCTION) {
          VLOG(6) << op_type << " run function kernel";
          if (static_build) {
            FakeInitializeOutputsForFunctionKernel(
                *op,
                *(op_func_node.phi_kernel_),
                *(op_with_kernel->PhiKernelSignature()),
                runtime_context,
                *dev_ctx);
          } else {
            phi::KernelContext phi_kernel_context;
            op_with_kernel->BuildPhiKernelContext(
                runtime_context, dev_ctx, &phi_kernel_context);
            (*op_func_node.phi_kernel_)(&phi_kernel_context);
          }
        } else if (run_phi_kernel &&
                   op_func_node.phi_kernel_->GetKernelRegisteredType() ==
                       phi::KernelRegisteredType::STRUCTURE) {
          VLOG(6) << op_type << " run structure kernel";
          ExecutionContext execution_context(
              *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
          if (static_build) {
            FakeInitializeOutputsForStructureKernel(kernel_type,
                                                    &execution_context);
          } else {
            (*op_func_node.phi_kernel_)(&execution_context);
          }
        } else {
          VLOG(6) << op_type << " run fluid kernel";
          // the place of exec_ctx maybe has changed.
          ExecutionContext execution_context(
              *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
          if (static_build) {
            FakeInitializeOutputsForStructureKernel(kernel_type,
                                                    &execution_context);
          } else {
            op_func_node.kernel_func_(execution_context);
          }
        }

        // for debug nan/inf

        vec_func_list->emplace_back(op_func_node);

        if (!op_func_node.inplace_back_map.empty()) {
          auto& m = op_func_node.inplace_back_map;
          // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in
          // operator.cc
          for (auto& p : m) {
            auto* transformed_tensor =
                GetMutableLoDTensorOrSelectedRowsValueFromVar(
                    local_scope->FindVar(var_scope->GetNameById(p.first)));
            auto* original_tensor =
                GetMutableLoDTensorOrSelectedRowsValueFromVar(
                    local_scope->FindVar(var_scope->GetNameById(p.second)));

            // avoid overwriting valid data
            if (static_build && original_tensor->initialized()) {
              const phi::Place& target_place = transformed_tensor->place();
              platform::DeviceContext* dev_ctx_for_copy;
              if (target_place.GetType() != AllocationType::CPU) {
                dev_ctx_for_copy = pool.Get(target_place);
              } else {
                dev_ctx_for_copy = pool.Get(original_tensor->place());
              }

              phi::Copy(*dev_ctx_for_copy,
                        *original_tensor,
                        target_place,
                        /*blocking=*/true,
                        original_tensor);
              original_tensor->set_type(transformed_tensor->dtype());
              original_tensor->set_layout(transformed_tensor->layout());
            } else {
              original_tensor->ShareDataWith(*transformed_tensor);
            }
            VLOG(4) << "Transfer inplace variable back form "
                    << var_scope->GetNameById(p.first) << " to "
                    << var_scope->GetNameById(p.second);
          }
        }

        // post-process grad_op.outputs if need cast complex grad into real
        // grad.
        // NOTE(Aurelius84): insert a transfer_dtype_op inplacely to cast it.
        if (IsGradOp(op_type) &&
            framework::IsComplexType(kernel_type.data_type_)) {
          interpreter::HandleComplexGradToRealGrad(op_func_node,
                                                   place,
                                                   output_name_map,
                                                   &runtime_context.outputs,
                                                   var_scope,
                                                   vec_func_list,
                                                   local_scope,
                                                   static_build);
        }
      }
    } catch (platform::EnforceNotMet& ex) {
      framework::InsertCallStackInfo(op_type, op->Attrs(), &ex);
      throw std::move(ex);
    } catch (platform::EOFException&) {
      std::rethrow_exception(std::current_exception());
    } catch (std::exception& ex) {
      LOG(WARNING) << op_type << " raises an exception "
                   << platform::demangle(typeid(ex).name()) << ", "
                   << ex.what();
      std::rethrow_exception(std::current_exception());
    } catch (...) {
      LOG(WARNING) << op_type << " raises an unknown exception";
      std::rethrow_exception(std::current_exception());
    }

    if (FLAGS_check_nan_inf) {
      VLOG(4) << "Check nan/inf";
      try {
        framework::details::CheckOpHasNanOrInf(*op, *local_scope, place);
      } catch (...) {
        const std::vector<std::string>* callstack = nullptr;
        auto attrs = op->Attrs();
        auto iter =
            attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
        if (iter != attrs.end()) {
          callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
          if (callstack->empty()) callstack = nullptr;
        }
        std::ostringstream sout;
        if (callstack) {
          if (FLAGS_call_stack_level > 1) {
            sout << "\n\n  Compile Traceback (most recent call last):";
          } else {
            sout << "In user code:\n";
          }
          for (auto& line : *callstack) {
            sout << "\n  " << line;
          }
        }
        std::cout << sout.str() << std::endl;
        std::rethrow_exception(std::current_exception());
      }
    }

    VLOG(4) << "End run " << place << " "
            << op_func_node.operator_base_->DebugStringEx(local_scope);

    if (!static_build) {
      // gc---------------------------------------------
      auto iter = unused_var_map.find(op);
      if (iter == unused_var_map.end()) {
        interpreter::LogDeviceMemoryStats(place);
        continue;
      }

      auto& delete_vars = iter->second;
      std::deque<std::shared_ptr<memory::Allocation>>* garbages =
          new std::deque<std::shared_ptr<memory::Allocation>>();

      for (auto& var_name : delete_vars) {
        auto* var = local_scope->FindVar(var_name);
        if (var == nullptr ||
            skip_gc_vars.find(var_name) != skip_gc_vars.end()) {
          continue;
        }

        VLOG(6) << "Erase variable " << var_name;
        if (var->IsType<phi::DenseTensor>()) {
          garbages->emplace_back(
              var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
        }
      }
      delete garbages;  // free mem

      interpreter::LogDeviceMemoryStats(place);
    }
  }

  // in the unused_var_map, we did not record the variable who are created in
  // ApplyDataTransform. So we erase these variables here.
  std::deque<std::shared_ptr<memory::Allocation>>* garbages =
      new std::deque<std::shared_ptr<memory::Allocation>>();
  for (const auto& transferred_var : var_scope->DataTransferAddedVars()) {
    const auto& var_name = transferred_var.first;
    auto* var = local_scope->FindVar(var_name);
    if (var == nullptr) continue;
    VLOG(6) << "Erase variable " << var_name;
    if (var->IsType<phi::DenseTensor>()) {
      garbages->emplace_back(
          var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
    }
  }
  delete garbages;
}

void BuildOpFuncList(
    const platform::Place& place,
    pir::Block* block,
    std::vector<OpFuncNode>* vec_func_list,
    framework::Scope* scope,
    framework::Scope* local_scope,
    const std::unordered_map<pir::Value, std::string>& value_2_name_map,
    const ExecutionConfig& execution_config) {
  vec_func_list->reserve(block->size());
  pir::IrContext* ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  for (auto op : *block) {
    OpFuncNode op_func_node;
    auto attr_map = op->attributes();

    auto op_name =
        attr_map.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
    op_func_node.phi_op_name_ = op_name;

    if (GetSpecialOpNames().count(op_name)) {
      VLOG(6) << "skip process " << op_name;
      continue;
    }

    pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

    auto impl =
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();

    op_func_node.infer_meta_interface_ =
        op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();

    VLOG(6) << "op name" << op_func_node.phi_op_name_;
    dialect::OpYamlInfoParser op_yaml_info_parser(impl->get_op_info_());
    if (op_func_node.infer_meta_interface_) {
      pir::BuildPhiContext<
          phi::InferMetaContext,
          phi::MetaTensor,
          phi::MetaTensor,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          false>(op,
                 value_2_name_map,
                 scope,
                 local_scope,
                 op_yaml_info_parser,
                 &(op_func_node.infer_meta_context_));
    }

    auto kernel_name =
        attr_map.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
    auto kernel_key = attr_map.at("kernel_key")
                          .dyn_cast<paddle::dialect::KernelAttribute>()
                          .data();

    VLOG(6) << "finish process infer meta context";
    auto t1 = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        kernel_name, kernel_key);
    op_func_node.phi_kernel_ = new phi::Kernel(t1.kernel);

    PADDLE_ENFORCE_EQ(op_func_node.phi_kernel_->IsValid(),
                      true,
                      "not found kernel for [%s]",
                      kernel_name);

    pir::BuildPhiContext<phi::KernelContext,
                         const phi::TensorBase*,
                         phi::TensorBase*,
                         paddle::small_vector<const phi::TensorBase*>,
                         paddle::small_vector<phi::TensorBase*>,
                         true>(op,
                               value_2_name_map,
                               scope,
                               local_scope,
                               op_yaml_info_parser,
                               &(op_func_node.kernel_context_));

    VLOG(6) << "finish process kernel context";
    op_func_node.kernel_context_.SetDeviceContext(
        phi::DeviceContextPool::Instance().Get(
            phi::TransToPhiPlace(kernel_key.backend())));
    op_func_node.dev_ctx_ = phi::DeviceContextPool::Instance().Get(
        phi::TransToPhiPlace(kernel_key.backend()));

    vec_func_list->emplace_back(op_func_node);
  }
}

void BuildVariableScope(const framework::BlockDesc& block,
                        const ExecutionConfig& execution_config,
                        VariableScope* var_scope) {
  VLOG(3) << "Creating Variables";
  auto inner_scope = var_scope->GetMutableScope();

  // NOTE(zhiqiu): if create_local_scope_ is true, the persistable is
  // created in var_scope.scope_ , and other scope is created in local scope.
  Scope* local_scope = execution_config.create_local_scope
                           ? var_scope->GetMutableLocalScope()
                           : var_scope->GetMutableScope();

  for (auto& var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    // TODO(xiongkun): user may create a variable with name that exists before.
    // under such circumstances, we should raise a error. Currently we can't
    // get the var_desc of startup_program, so leave it later.
    if (var_name == framework::kEmptyVarName) {
      continue;
    }

    if (var_desc->Persistable() ||
        execution_config.force_root_scope_vars.count(var_name)) {
      // In principle, we should put all trainable parameters in global scope,
      // which means the root of the scope tree. Some cases like quantization
      // will look up these parameters in global scope.
      const Scope* ancestor_scope = inner_scope;
      while (ancestor_scope->parent()) {
        ancestor_scope = ancestor_scope->parent();
      }
      auto* ptr = const_cast<Scope*>(ancestor_scope)->Var(var_name);

      // NOTE(zhiqiu): if var exists in scope and the type is right,
      // InitializeVariable will not create a new variable.
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " global, which pointer is "
              << ptr << " type is " << static_cast<int>(var_desc->GetType());
    } else {
      auto* ptr = local_scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " locally, which pointer is "
              << ptr << " type is " << static_cast<int>(var_desc->GetType());
    }
    var_scope->AddVar(var_name, var_desc);
  }
}

void LogDeviceMemoryStats(const platform::Place& place) {
  if (FLAGS_new_executor_log_memory_stats && platform::is_gpu_place(place)) {
    VLOG(0) << "memory_allocated: "
            << static_cast<double>(memory::DeviceMemoryStatCurrentValue(
                   "Allocated", place.device)) /
                   1024 / 1024
            << " MB";
    VLOG(0) << "max_memory_allocated: "
            << static_cast<double>(memory::DeviceMemoryStatPeakValue(
                   "Allocated", place.device)) /
                   1024 / 1024
            << " MB";
  }
}

void SetDeviceCommContext(framework::OperatorBase* operator_base,
                          platform::DeviceContext* dev_ctx) {
  if (operator_base->HasAttr("ring_id")) {
    int ring_id = operator_base->Attr<int>("ring_id");
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(std::to_string(ring_id))) {
      auto comm_context = comm_context_manager.Get(std::to_string(ring_id));
      if (!dev_ctx->GetCommContext()) {
        dev_ctx->SetCommContext(comm_context);
      }
    } else {
      VLOG(3) << "op: " << operator_base->Type() << ", ring_id: " << ring_id
              << ", get comm_context failed!";
    }
  }
}

void SetDeviceCommContext(pir::Operation* op,
                          platform::DeviceContext* dev_ctx) {
  auto op_attributes = op->attributes();
  if (op_attributes.count("ring_id") != 0) {
    int ring_id =
        op_attributes.at("ring_id").dyn_cast<pir::Int32Attribute>().data();
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(std::to_string(ring_id))) {
      auto comm_context = comm_context_manager.Get(std::to_string(ring_id));
      if (!dev_ctx->GetCommContext()) {
        dev_ctx->SetCommContext(comm_context);
      }
    } else {
      VLOG(3) << "op: "
              << op_attributes.at("op_name")
                     .dyn_cast<pir::StrAttribute>()
                     .AsString()
              << ", ring_id: " << ring_id << ", get comm_context failed!";
    }
  }
}

std::unordered_set<std::string> GetSpecialOpNames() {
  return {
      "builtin.combine",
      "builtin.slice",
      "pd_op.feed",
      "builtin.set_parameter",
      "builtin.get_parameter",
      "pd_op.data",
      "pd_op.shadow_output",
  };
}

void BuildId2VarName(const std::map<std::string, int>& var_name_2_id,
                     std::unordered_map<int, std::string>* id_2_var_name) {
  PADDLE_ENFORCE_NOT_NULL(
      id_2_var_name,
      phi::errors::InvalidArgument("id2_var_name can not be null"));
  for (auto [var_name, id] : var_name_2_id) {
    id_2_var_name->insert({id, var_name});
  }
}

const paddle::framework::Variable GetVariableByName(
    const std::string& var_name,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  for (auto kv : variable_2_var_name) {
    if (kv.second == var_name) {
      return *kv.first;
    }
  }
  return nullptr;
}

void PrintValuesAndVariables (
    const pir::Block& block,
    const std::unordered_map<pir::Value, std::string>* value_2_var_name,
    const std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name) {
  for (const auto& op : block) {
    std::stringstream ss;
    op->Print(ss);

    // 1. output string
    std::string ret_value_str = "Value: (";
    std::string ret_variable_str = "Variable: (";
    if (!op->results().empty()) {
      for (auto& out_value : op->results()) {
        auto& var_name = (*value_2_var_name).at(out_value);
        const paddle::framework::Variable out_variable =
            GetVariableByName(var_name, *variable_2_var_name);
        ret_value_str += (var_name + "[" + std::to_string(&out_value) + "], ");
        ret_variable_str +=
            (var_name + "[" + std::to_string(&out_variable) + "], ");
      }
      ret_value_str = ret_value_str.substr(0, ret_value_str.size() - 2);
      ret_variable_str =
          ret_variable_str.substr(0, ret_variable_str.size() - 2);
    }
    ret_value_str += ") = ";
    ret_variable_str += ") = ";

    // 2. op name
    auto op_name = op->attributes()
                       .at("op_name")
                       .dyn_cast<::pir::StrAttribute>()
                       .AsString();
    ret_value_str += op_name;
    ret_variable_str += op_name;

    // 3. input string
    ret_str += "(";
    if (!op->operands().empty()) {
      for (auto& input : op->operands()) {
        ::pir::Value in_value = input.source();
        auto& var_name = (*value_2_var_name).at(in_value);
        const paddle::framework::Variable in_variable =
            GetVariableByName(var_name, *variable_2_var_name);
        ret_value_str += (var_name + "[" + std::to_string(&in_value) + "], ");
        ret_variable_str +=
            (var_name + "[" + std::to_string(&in_variable) + "], ");
      }
      ret_value_str = ret_value_str.substr(0, ret_value_str.size() - 2);
      ret_variable_str =
          ret_variable_str.substr(0, ret_variable_str.size() - 2);
    }
    ret_value_str += ")";
    ret_variable_str += ")";
  }
  VLOG(8) << ret_value_str;
  VLOG(8) << ret_variable_str;
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
