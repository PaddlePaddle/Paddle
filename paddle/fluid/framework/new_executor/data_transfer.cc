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

#include "paddle/fluid/framework/new_executor/data_transfer.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/phi/backends/onednn/onednn_context.h"
#endif

namespace paddle {
namespace framework {
namespace interpreter {

bool DataTranferHelper::apply(const OpKernelType& kernel_type_for_var,
                              const OpKernelType& expected_kernel_key,
                              const std::string& var_name,
                              std::string* new_var_name,
                              std::vector<OpFuncNode>* op_func_nodes,
                              bool use_local_scope,
                              bool is_fetch_v2) {
  bool is_transferred = false;
  auto* src_var_name = &var_name;

  // 1. layout transform
  if (need_layout_transform(kernel_type_for_var, expected_kernel_key)) {
    auto op = TransferLayout(*src_var_name,
                             new_var_name,
                             kernel_type_for_var.data_layout_,
                             expected_kernel_key.data_layout_,
                             var_scope_,
                             scope_,
                             is_fetch_v2);
    if (op) {
      RunAndConstructOpFuncNode(
          op, *src_var_name, *new_var_name, op_func_nodes);
    }
    // update src_var_name
    src_var_name = new_var_name;
    is_transferred = true;
  }
  // 2. dype transform
  if (need_dtype_transform(kernel_type_for_var, expected_kernel_key)) {
    auto op = TransferDtype(*src_var_name,
                            new_var_name,
                            kernel_type_for_var.data_type_,
                            expected_kernel_key.data_type_,
                            var_scope_,
                            scope_);
    if (op) {
      RunAndConstructOpFuncNode(
          op, *src_var_name, *new_var_name, op_func_nodes);
    }
    // update src_var_name
    src_var_name = new_var_name;
    is_transferred = true;
  }
  // 3. device transform
  if (need_device_transform(kernel_type_for_var, expected_kernel_key)) {
    auto src_place = kernel_type_for_var.place_;
    auto dst_place = expected_kernel_key.place_;

    auto op = TransferDevice(
        *src_var_name, new_var_name, src_place, dst_place, var_scope_, scope_);
    if (op) {
      RunAndConstructOpFuncNode(
          op, *src_var_name, *new_var_name, op_func_nodes);
    }
    is_transferred = true;
  }
  return is_transferred;
}

void DataTranferHelper::RunAndConstructShareNode(
    const std::string& src_var_name,
    const std::string& dst_var_name,
    std::vector<OpFuncNode>* op_func_nodes) {
  VariableNameMap in_name_map = {{"X", {src_var_name}}};
  VariableNameMap out_name_map = {{"Out", {dst_var_name}}};
  AttributeMap attr_map;

  std::string op_type("share_data");
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf(
      "Insert %s with %s -> %s.", op_type, src_var_name, dst_var_name);

  RunAndConstructOpFuncNode(op, src_var_name, dst_var_name, op_func_nodes);
}

void DataTranferHelper::RunAndConstructOpFuncNode(
    const std::shared_ptr<OperatorBase>& op,
    const std::string& var_name,
    const std::string& new_var_name,
    std::vector<OpFuncNode>* new_op_func_nodes) {
  auto& op_type = op->Type();

  // 1. Construct RuntimeContext
  RuntimeContext runtime_context({}, {});
  runtime_context.inputs["X"] = {scope_->FindVar(var_name)};
  runtime_context.outputs["Out"] = {scope_->Var(new_var_name)};
  InterpretercoreInferShapeContext infer_shape_ctx(*op, runtime_context);
  op.get()->Info().infer_shape_(&infer_shape_ctx);

  // 2. choose kernel

  // prepare a ptr to OperatorWithKernel
  OperatorBase* op_ptr = op.get();
  if (dynamic_cast<framework::OperatorWithKernel*>(op_ptr) == nullptr) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "%s should be OperatorWithKernel type.", op_ptr->Type()));
  }
  auto op_with_kernel = static_cast<framework::OperatorWithKernel*>(op_ptr);

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place_);
  auto exec_ctx = ExecutionContext(*op, Scope(), *dev_ctx, runtime_context);
  auto expected_kernel_key = op_with_kernel->GetExpectedKernelType(exec_ctx);

  VLOG(6) << "expected_kernel_key " << expected_kernel_key << "\n";
  VLOG(6) << "op_with_kernel Type() " << op_with_kernel->Type() << "\n";

  bool run_phi_kernel = false;

  // check if phi kernel exists
  if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
          op_with_kernel->Type())) {
    auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
    auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;
    VLOG(6) << "phi_kernel_key " << phi_kernel_key << "\n";
    VLOG(6) << "phi_kernel_name " << phi_kernel_name << "\n";

    if (op_with_kernel->PhiKernel()->IsValid()) {
      run_phi_kernel = true;
    }

    // For data transfer ops, they should not fallback to cpu.
    // Though they're device-independent operations,
    // their implementations are device-related.
    // For example, consider changing the layout of a gpu tensor
    // while the gpu kernel of transfer_layout op does not exist.
    // To use the cpu kernel, you must insert memcpy_d2h/mepcpy_h2d op
    // in addition. But such operation should not be done here.
    // Maybe in future we will support this.
  }

  // 3. Execute transfer op and construct OpFuncNode
  OpFuncNode new_op_func_node;
  new_op_func_node.input_index["X"] = {var_scope_->VarId(var_name)};
  new_op_func_node.output_index["Out"] = {var_scope_->VarId(new_var_name)};

  if (!run_phi_kernel) {
    op_with_kernel->ChooseKernel(exec_ctx);
    new_op_func_node.kernel_func_ = *op_with_kernel->kernel_func();
    new_op_func_node.kernel_func_(exec_ctx);
  } else {
    new_op_func_node.phi_kernel_ = op_with_kernel->PhiKernel();
    phi::KernelContext phi_kernel_context;
    op_with_kernel->BuildPhiKernelContext(
        runtime_context, dev_ctx, &phi_kernel_context);
    (*new_op_func_node.phi_kernel_)(&phi_kernel_context);
  }

  // NOTE(winter-wang): in npu device, D2H kernel is asynchronous. need to
  // explicit synchronization.
#ifdef PADDLE_WITH_ASCEND_CL
  if (op_type == kMemcpyD2H && platform::is_npu_place(dev_ctx->GetPlace())) {
    dev_ctx->Wait();
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (op_type == kMemcpyD2H && platform::is_custom_place(dev_ctx->GetPlace())) {
    dev_ctx->Wait();
  }
#endif
  // NOTE(Aurelius84): data_transform_op is expensive operation, so we tag them
  // as kQueueSync and execute them in thread pool.
  new_op_func_node.type_ = OpFuncType::kQueueSync;
  new_op_func_node.dev_ctx_ = dev_ctx;
  new_op_func_node.operator_base_ = op;
  VLOG(3) << "Run " << op_type << " done.";

  new_op_func_nodes->emplace_back(std::move(new_op_func_node));
}

// Var is initialized && var contains tensor && tensor is initialized
bool IsTensorOfVarInitialized(Variable* var) {
  if (var->IsInitialized()) {
    if (var->IsType<LoDTensor>() || var->IsType<phi::SelectedRows>()) {
      return GetLoDTensorOrSelectedRowsValueFromVar(*var)->IsInitialized();
    } else if (var->IsType<LoDTensorArray>()) {
      return static_cast<const phi::DenseTensor*>(
                 &(var->Get<LoDTensorArray>()[0]))
          ->IsInitialized();
    }
  }
  return false;
}

std::shared_ptr<OperatorBase> TransferLayout(const std::string& var_name,
                                             std::string* new_var_name,
                                             DataLayout in_layout,
                                             DataLayout out_layout,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope,
                                             bool is_fetch_v2) {
#ifdef PADDLE_WITH_MKLDNN

  // NOTE(zhiqiu): hot fix, follow the same logic in DataCopy() in fetch_op.cc
  if (in_layout == framework::DataLayout::kMKLDNN &&
      var_name == framework::GradVarName("Filter") && is_fetch_v2) {
    VLOG(4) << "Match special case(Filter && fetch_v2) " << var_name;
    out_layout = framework::DataLayout::kNCHW;
  }

  if (in_layout == framework::DataLayout::ONEDNN &&
      out_layout != framework::DataLayout::ONEDNN) {
    auto target_layout = phi::OneDNNContext::tls().get_cur_paddle_data_layout();
    VLOG(4) << "TransDataLayoutFromOneDNN: " << in_layout << "->"
            << target_layout;

    if (out_layout == DataLayout::kNCHW &&
        var_name == framework::GradVarName("Filter")) {
      VLOG(4) << "Match special case(Filter) " << var_name;
      target_layout = out_layout;
    }
    out_layout = target_layout;
  }
#endif

  // 1. Generate new_var_name and Initialize it
  *new_var_name = var_name + "_layout_" +
                  std::to_string(static_cast<int>(in_layout)) + "_" +
                  std::to_string(static_cast<int>(out_layout));

  if (var_scope->HasVar(*new_var_name) &&
      IsTensorOfVarInitialized(local_scope->FindVar(*new_var_name))) {
    // already has same var
    VLOG(4) << "Use cached variable: " << *new_var_name;
    return nullptr;
  }

  auto* ptr = local_scope->Var(*new_var_name);
  auto var_type = local_scope->FindVar(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->MutableDataTransferAddedVars().push_back(
      std::make_pair(*new_var_name, var_type));
  var_scope->AddVar(*new_var_name, nullptr);

  // 2. Construct VariableNameMap
  VariableNameMap in_name_map = {{"X", {var_name}}};
  VariableNameMap out_name_map = {{"Out", {*new_var_name}}};
  AttributeMap attr_map = {{"src_layout", static_cast<int>(in_layout)},
                           {"dst_layout", static_cast<int>(out_layout)}};

  // 3. Create transfer_layout_op
  std::string op_type("transfer_layout");
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s for variable %s(%s) -> %s(%s).",
                             op_type,
                             var_name,
                             in_layout,
                             *new_var_name,
                             out_layout);
  return op;
}

std::shared_ptr<OperatorBase> TransferDtype(const std::string& var_name,
                                            std::string* new_var_name,
                                            proto::VarType::Type in_dtype,
                                            proto::VarType::Type out_dtype,
                                            framework::VariableScope* var_scope,
                                            framework::Scope* local_scope) {
  // 1. Generate new_var_name and Initialize it
  *new_var_name = var_name + "_dtype_" +
                  std::to_string(static_cast<int>(in_dtype)) + "_" +
                  std::to_string(static_cast<int>(out_dtype));
  if (var_scope->HasVar(*new_var_name) &&
      IsTensorOfVarInitialized(local_scope->FindVar(*new_var_name))) {
    // already has same var
    VLOG(4) << "Use cached variable: " << *new_var_name;
    return nullptr;
  }

  auto* ptr = local_scope->Var(*new_var_name);
  auto var_type = local_scope->FindVar(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->MutableDataTransferAddedVars().push_back(
      std::make_pair(*new_var_name, var_type));
  var_scope->AddVar(*new_var_name, nullptr);

  // 2. Construct VariableNameMap
  VariableNameMap in_name_map = {{"X", {var_name}}};
  VariableNameMap out_name_map = {{"Out", {*new_var_name}}};
  AttributeMap attr_map;
  attr_map["in_dtype"] = static_cast<int>(in_dtype);
  attr_map["out_dtype"] = static_cast<int>(out_dtype);
  // NOTE(Aurelius84): In whice case use_mkldnn = true?
  attr_map["use_mkldnn"] = false;

  // 3. Create transfer_dtype_op
  std::string op_type("transfer_dtype");
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).",
                             op_type,
                             var_name,
                             DataTypeToString(in_dtype),
                             *new_var_name,
                             DataTypeToString(out_dtype));
  return op;
}

std::shared_ptr<OperatorBase> TransferDevice(const std::string& var_name,
                                             std::string* new_var_name,
                                             const platform::Place& src_place,
                                             const platform::Place& dst_place,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope) {
  // 1. Generate new_var_name and Initialize it
  *new_var_name = var_name + "_device_" + src_place.DebugString() + "_" +
                  dst_place.DebugString();

  if (var_scope->HasVar(*new_var_name) &&
      IsTensorOfVarInitialized(local_scope->FindVar(*new_var_name))) {
    // already has same var
    VLOG(4) << "Use cached variable: " << *new_var_name;
    return nullptr;
  }

  auto* ptr = local_scope->Var(*new_var_name);
  auto var_type = local_scope->FindVar(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->MutableDataTransferAddedVars().push_back(
      std::make_pair(*new_var_name, var_type));
  var_scope->AddVar(*new_var_name, nullptr);

  // 2. Construct VariableNameMap
  VariableNameMap in_name_map = {{"X", {var_name}}};
  VariableNameMap out_name_map = {{"Out", {*new_var_name}}};

  // 3. Create memcpy_d2h_op or memcpy_h2d_op
  std::string op_type;
  AttributeMap attr_map;
  PADDLE_ENFORCE_EQ(platform::is_same_place(src_place, dst_place),
                    false,
                    platform::errors::PreconditionNotMet(
                        "Required src_place shall be different with dst_place, "
                        "but received same place: %s",
                        src_place));
  if (IsSupportedHetePlace(dst_place)) {
    op_type = kMemcpyH2D;
    int dst_place_type = platform::is_gpu_place(dst_place)      ? 0
                         : platform::is_npu_place(dst_place)    ? 1
                         : platform::is_ipu_place(dst_place)    ? 3
                         : platform::is_xpu_place(dst_place)    ? 2
                         : platform::is_custom_place(dst_place) ? 6
                                                                : -1;
    attr_map = {{"dst_place_type", dst_place_type}};
  } else if (IsSupportedHetePlace(src_place)) {
    op_type = kMemcpyD2H;
    int dst_place_type = platform::is_cpu_place(dst_place)           ? 0
                         : platform::is_cuda_pinned_place(dst_place) ? 1
                                                                     : -1;
    attr_map = {{"dst_place_type", dst_place_type}};
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Not support Memcpy typ : %s -> %s", src_place, dst_place));
  }

  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).",
                             op_type,
                             var_name,
                             src_place,
                             *new_var_name,
                             dst_place);
  return op;
}

void ApplyDataTransform(const OpKernelType& expected_kernel_key,
                        const platform::Place& place,
                        VariableValueMap* ins_map_temp,
                        VariableValueMap* outs_map_temp,
                        VariableScope* var_scope,
                        OpFuncNode* op_func_node,
                        std::vector<OpFuncNode>* new_op_func_nodes,
                        bool use_local_scope) {
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();

  auto op_base = op_func_node->operator_base_.get();
  PADDLE_ENFORCE_NOT_NULL(op_base,
                          platform::errors::PreconditionNotMet(
                              "op_base is null, please pass a valid "
                              "op_base in apply_data_transform."));

  VariableNameMap new_ins(op_base->Inputs());
  VariableNameMap new_outs(op_base->Outputs());
  // record the no need transform variable index.
  std::unordered_set<int> no_data_transform_index;

  const std::unordered_set<std::string>* no_buffer_ins = nullptr;
  auto& no_buffer_inferer = op_base->Info().NoNeedBufferVarsInferer();
  if (no_buffer_inferer) {
    no_buffer_ins = &(no_buffer_inferer(
        op_base->Inputs(), op_base->Outputs(), op_base->Attrs()));
    if (no_buffer_ins->empty()) {
      no_buffer_ins = nullptr;
    }
  }

  bool transfered = false;
  DataTranferHelper data_transfer_helper(place, var_scope, local_scope);
  for (auto& var_name_item : *ins_map_temp) {
    bool should_skip_input =
        no_buffer_ins && no_buffer_ins->count(var_name_item.first) > 0;

    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto var = var_name_item.second[i];
      auto var_name = new_ins[var_name_item.first].at(i);
      const phi::DenseTensor* tensor_in;
      std::string new_var_name;
      bool is_transferred = false;

      if (var->IsType<LoDTensor>() || var->IsType<phi::SelectedRows>()) {
        tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      } else if (var->IsType<LoDTensorArray>()) {
        if (var->Get<LoDTensorArray>().size() == 0) {
          continue;
        }
        tensor_in = static_cast<const phi::DenseTensor*>(
            &(var->Get<LoDTensorArray>()[0]));
      } else {
        continue;
      }
      // special case
      if (!tensor_in->IsInitialized()) {
        if (should_skip_input == true) {
#ifdef PADDLE_WITH_MKLDNN
          // Var without buffer may be needed
          // for some situation like InferShape().
          // In this situation We cannot skip Var analysis, as
          // MKL-DNN shape of Var may differ from kNHWC Var
          // In such situation corressponding resized Var
          // has to be created and registered
          if ((tensor_in->layout() == DataLayout::kMKLDNN) &&
              (var->IsType<LoDTensor>() == true) &&
              (expected_kernel_key.data_layout_ != DataLayout::kMKLDNN) &&
              (paddle::platform::MKLDNNDeviceContext::tls()
                   .get_cur_paddle_data_layout() == DataLayout::kNHWC)) {
            VLOG(7) << "Created reshaped dummy input based on MKL-DNN "
                       "phi::DenseTensor , "
                       "but kNHWC layout"
                    << var_name_item.first << " in Operator "
                    << op_base->Type();
            auto op = TransferLayout(var_name,
                                     &new_var_name,
                                     tensor_in->layout(),
                                     DataLayout::kNHWC,
                                     var_scope,
                                     local_scope,
                                     op_base->Type() == "fetch_v2");
            if (op) {
              data_transfer_helper.RunAndConstructOpFuncNode(
                  op, var_name, new_var_name, new_op_func_nodes);
            }
            is_transferred = true;
          } else {
            VLOG(7) << "Skip scanning input " << var_name_item.first
                    << " in Operator " << op_base->Type();
          }
#endif
        } else {
          continue;
        }
      } else {
        auto kernel_type_for_var =
            static_cast<const framework::OperatorWithKernel*>(op_base)
                ->GetKernelTypeForVar(
                    var_name_item.first, *tensor_in, expected_kernel_key);
        // apply data transform
        is_transferred =
            data_transfer_helper.apply(kernel_type_for_var,
                                       expected_kernel_key,
                                       var_name,
                                       &new_var_name,
                                       new_op_func_nodes,
                                       use_local_scope,
                                       op_base->Type() == "fetch_v2");
      }

      if (is_transferred) {
        transfered = true;
        // update RuntimeContext.inputs and original op_func_node inputs
        op_func_node->input_index[var_name_item.first][i] =
            var_scope->VarId(new_var_name);
        var_name_item.second[i] = local_scope->FindVar(new_var_name);
        new_ins[var_name_item.first][i] = new_var_name;
        for (auto& pair : new_outs) {
          for (size_t j = 0; j < pair.second.size(); ++j) {
            VLOG(4) << pair.second[j] << " " << var_name;
            if (pair.second[j] == var_name) {
              VLOG(4) << "Found inplace between input(" << var_name_item.first
                      << ") and output(" << pair.first
                      << "), the variable name is " << var_name;
              (*outs_map_temp)[pair.first][j] =
                  local_scope->FindVar(new_var_name);
              new_outs[pair.first][j] = new_var_name;
              op_func_node
                  ->inplace_back_map[var_scope->GetIdByName(new_var_name)] =
                  var_scope->GetIdByName(var_name);
              op_func_node->output_index[pair.first][j] =
                  var_scope->VarId(new_var_name);
            }
          }
        }
        // NOTE(Aurelius84): avoid deepcopy twice if we already insert data
        // transfer op.
        if (op_base->Type() == "fetch_v2") {
          op_base->SetAttr("deepcopy", false);
        }
      } else {
        // record no need data transformer input var_id
        VLOG(3) << op_base->Type()
                << " found no data_transform var: " << var_name
                << " with id: " << var_scope->VarId(var_name);
        no_data_transform_index.emplace(var_scope->VarId(var_name));
      }
    }
  }

  if (transfered) {
    // NOTE(zhiqiu): UPDATE the corresponding OeratorBase to make it consistent
    // with instruction.
    op_base->Inputs() = new_ins;
    op_base->Outputs() = new_outs;
  }
  op_func_node->no_data_transform_index = std::move(no_data_transform_index);
}

void HandleComplexGradToRealGrad(const OpFuncNode& op_func_node,
                                 const platform::Place& place,
                                 const VariableNameMap& out_names,
                                 VariableValueMap* out_vars,
                                 VariableScope* var_scope,
                                 std::vector<OpFuncNode>* op_func_nodes,
                                 framework::Scope* local_scope) {
  DataTranferHelper data_transfer_helper(place, var_scope, local_scope);
  for (auto& var_name_item : out_names) {
    std::vector<Variable*>& vars = out_vars->at(var_name_item.first);
    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      // 1. find grad_var & check whether is complex tensor
      auto var_name = var_name_item.second[i];
      auto orig_var_name = framework::GradOriginalVarName(var_name);
      // only focus on gradient var
      if (var_name == orig_var_name) {
        VLOG(3) << "skip " << var_name << " with same name as "
                << orig_var_name;
        continue;
      }
      auto* grad_var = vars[i];
      // skip nullptr var
      if (grad_var == nullptr) {
        VLOG(3) << "skip grad_var with nullptr";
        continue;
      }
      // don't process LoDTensorArray temporarily,
      // add support if necessary for complex number calculations in the future
      if (!framework::VarIsTensor(*grad_var)) {
        VLOG(3) << "skip grad_var with LoDTensorArray type";
        continue;
      }
      auto* grad_tensor =
          framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(grad_var);
      // skip nullptr tensor
      if (grad_tensor == nullptr || !grad_tensor->IsInitialized()) {
        VLOG(3) << "skip with grad_tensor not IsInitialized";
        continue;
      }
      // only focus on complex dtype now
      auto src_type = framework::TransToProtoVarType(grad_tensor->dtype());
      if (!framework::IsComplexType(src_type)) {
        VLOG(3) << "skip grad_tensor with not complexType";
        continue;
      }

      // 2. find forward var & check whether need to cast
      auto* var = local_scope->FindVar(orig_var_name);
      // if forward var not exists, do nothing
      if (var == nullptr) {
        VLOG(3) << "skip " << orig_var_name << " with not found in var_scope";
        continue;
      }
      if (!framework::VarIsTensor(*var)) {
        VLOG(3) << "skip " << orig_var_name << " with LoDTensorArray.";
        continue;
      }
      const auto* tensor =
          framework::GetLoDTensorOrSelectedRowsValueFromVar(*var);
      PADDLE_ENFORCE_NOT_NULL(
          tensor,
          platform::errors::Unavailable(
              "Forward tensor is nullptr when handle complex data to real."));
      // only need record type, the allocation may have been released
      auto dst_type = framework::TransToProtoVarType(tensor->dtype());
      // only focus on real dtype and need casting
      if (framework::IsComplexType(dst_type)) {
        continue;
      }

      // 3. cast complex grad to real grad inplacely
      VLOG(3) << "Transform " << framework::DataTypeToString(src_type)
              << " var `" << var_name << "` to "
              << framework::DataTypeToString(dst_type)
              << " real var in static graph.";

      // NOTE(Aurelius84): Consider to define a complex2real op to deal this
      // case.
      std::string new_var_name;
      auto op = TransferDtype(
          var_name, &new_var_name, src_type, dst_type, var_scope, local_scope);
      data_transfer_helper.RunAndConstructOpFuncNode(
          op, var_name, new_var_name, op_func_nodes);
      data_transfer_helper.RunAndConstructShareNode(
          new_var_name, var_name, op_func_nodes);
    }
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
