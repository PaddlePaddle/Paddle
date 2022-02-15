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

namespace paddle {
namespace framework {
namespace interpreter {

bool DataTranferHelper::apply(const OpKernelType& kernel_type_for_var,
                              const OpKernelType& expected_kernel_key,
                              const std::string& var_name,
                              std::string* new_var_name,
                              std::vector<OpFuncNode>* op_func_nodes,
                              bool use_local_scope) {
  bool is_transferred = false;
  auto* src_var_name = &var_name;

  Scope* local_scope = use_local_scope ? var_scope_->GetMutableLocalScope()
                                       : var_scope_->GetMutableScope();

  // 1. layout transform
  if (need_layout_transform(kernel_type_for_var, expected_kernel_key)) {
    auto op = TransferLayout(
        *src_var_name, new_var_name, kernel_type_for_var.data_layout_,
        expected_kernel_key.data_layout_, var_scope_, local_scope);
    RunAndConstructOpFuncNode(op, *src_var_name, *new_var_name, op_func_nodes);
    // update src_var_name
    src_var_name = new_var_name;
    is_transferred = true;
  }
  // 2. dype transform
  if (need_dtype_transform(kernel_type_for_var, expected_kernel_key)) {
    auto op = TransferDtype(
        *src_var_name, new_var_name, kernel_type_for_var.data_type_,
        expected_kernel_key.data_type_, var_scope_, local_scope);
    RunAndConstructOpFuncNode(op, *src_var_name, *new_var_name, op_func_nodes);
    // update src_var_name
    src_var_name = new_var_name;
    is_transferred = true;
  }
  // 3. device transform
  if (need_device_transform(kernel_type_for_var, expected_kernel_key)) {
    auto src_place = kernel_type_for_var.place_;
    auto dst_place = expected_kernel_key.place_;
    auto op = TransferDevice(*src_var_name, new_var_name, src_place, dst_place,
                             var_scope_, local_scope);
    RunAndConstructOpFuncNode(op, *src_var_name, *new_var_name, op_func_nodes);
    is_transferred = true;
  }
  return is_transferred;
}

void DataTranferHelper::RunAndConstructShareNode(
    const std::string& src_var_name, const std::string& dst_var_name,
    std::vector<OpFuncNode>* op_func_nodes) {
  VariableNameMap in_name_map = {{"X", {src_var_name}}};
  VariableNameMap out_name_map = {{"Out", {dst_var_name}}};
  AttributeMap attr_map;

  std::string op_type("share_data");
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s with %s -> %s.", op_type, src_var_name,
                             dst_var_name);

  RunAndConstructOpFuncNode(op, src_var_name, dst_var_name, op_func_nodes);
}

void DataTranferHelper::RunAndConstructOpFuncNode(
    const std::shared_ptr<OperatorBase>& op, const std::string& var_name,
    const std::string& new_var_name,
    std::vector<OpFuncNode>* new_op_func_nodes) {
  auto& op_type = op->Type();

  // 1. Construct RuntimeContext
  RuntimeContext runtime_context({}, {});
  runtime_context.inputs["X"] = {var_scope_->Var(var_name)};
  runtime_context.outputs["Out"] = {var_scope_->Var(new_var_name)};
  InterpretercoreInferShapeContext infer_shape_ctx(*op, runtime_context);

  // 2. Execute infer shape and choose kernel
  auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
  op.get()->Info().infer_shape_(&infer_shape_ctx);
  auto kernels_iter = all_op_kernels.find(op_type);
  PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                    platform::errors::Unavailable(
                        "There are no kernels which are registered in "
                        "the %s operator.",
                        op_type));
  OpKernelMap& kernels = kernels_iter->second;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place_);
  Scope scope;
  auto exec_ctx = ExecutionContext(*op, scope, *dev_ctx, runtime_context);
  auto expected_kernel_key =
      dynamic_cast<const framework::OperatorWithKernel*>(op.get())
          ->GetExpectedKernelType(exec_ctx);
  auto kernel_iter = kernels.find(expected_kernel_key);

  // 3. Execute transfer op and construct OpFuncNode
  OpFuncNode new_op_func_node;
  new_op_func_node.input_index["X"] = {var_scope_->VarId(var_name)};
  new_op_func_node.output_index["Out"] = {var_scope_->VarId(new_var_name)};
  new_op_func_node.kernel_func_ = OpKernelComputeFunc(kernel_iter->second);
  new_op_func_node.kernel_func_(exec_ctx);
  // NOTE(Aurelius84): data_transform_op is expensive operation, so we tag them
  // as kQueueSync and execute them in thread pool.
  new_op_func_node.type_ = OpFuncType::kQueueSync;
  new_op_func_node.dev_ctx_ = dev_ctx;
  new_op_func_node.operator_base_ = op;
  VLOG(3) << "Run " << op_type << " done.";

  new_op_func_nodes->emplace_back(std::move(new_op_func_node));
}

std::shared_ptr<OperatorBase> TransferLayout(const std::string& var_name,
                                             std::string* new_var_name,
                                             DataLayout in_layout,
                                             DataLayout out_layout,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope) {
  // 1. Generate new_var_name and Initialize it
  *new_var_name =
      var_name + "_layout_" + std::to_string(var_scope->VarSize() + 1);
  auto* ptr = local_scope->Var(*new_var_name);

  auto var_type = var_scope->Var(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->SetVarDesc(*new_var_name, nullptr);

  // 2. Construct VariableNameMap
  VariableNameMap in_name_map = {{"X", {var_name}}};
  VariableNameMap out_name_map = {{"Out", {*new_var_name}}};
  AttributeMap attr_map = {{"dst_layout", static_cast<int>(out_layout)}};

  // 3. Create transfer_layout_op
  std::string op_type("transfer_layout");
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s(%s) with %s -> %s(%s).", op_type,
                             var_name, in_layout, *new_var_name, out_layout);
  return op;
}

std::shared_ptr<OperatorBase> TransferDtype(const std::string& var_name,
                                            std::string* new_var_name,
                                            proto::VarType::Type in_dtype,
                                            proto::VarType::Type out_dtype,
                                            VariableScope* var_scope,
                                            framework::Scope* local_scope) {
  // 1. Generate new_var_name and Initialize it
  *new_var_name =
      var_name + "_dtype_" + std::to_string(var_scope->VarSize() + 1);
  auto* ptr = local_scope->Var(*new_var_name);

  auto var_type = var_scope->Var(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));

  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->SetVarDesc(*new_var_name, nullptr);

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

  VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).", op_type,
                             var_name, DataTypeToString(in_dtype),
                             *new_var_name, DataTypeToString(out_dtype));
  return op;
}

std::shared_ptr<OperatorBase> TransferDevice(const std::string& var_name,
                                             std::string* new_var_name,
                                             const platform::Place& src_place,
                                             const platform::Place& dst_place,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope) {
  // 1. Generate new_var_name and Initialize it
  *new_var_name =
      var_name + "_device_" + std::to_string(var_scope->VarSize() + 1);
  auto* ptr = local_scope->Var(*new_var_name);

  auto var_type = var_scope->Var(var_name)->Type();
  InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
  VLOG(3) << "Create Variable " << *new_var_name
          << " locally, which pointer is " << ptr << "Variable Type "
          << var_type;
  var_scope->SetVarDesc(*new_var_name, nullptr);

  // 2. Construct VariableNameMap
  VariableNameMap in_name_map = {{"X", {var_name}}};
  VariableNameMap out_name_map = {{"Out", {*new_var_name}}};
  int dst_place_type = platform::is_cpu_place(dst_place)
                           ? 0
                           : platform::is_gpu_place(dst_place) ? 1 : -1;
  AttributeMap attr_map = {{"dst_place_type", dst_place_type}};

  // 3. Create memcpy_d2h_op or memcpy_h2d_op
  std::string op_type = get_memcpy_type(src_place, dst_place);
  auto& op_info = OpInfoMap::Instance().Get(op_type);
  auto op = std::shared_ptr<OperatorBase>(
      op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

  VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).", op_type,
                             var_name, src_place, *new_var_name, dst_place);
  return op;
}

void ApplyDataTransform(const OpKernelType& expected_kernel_key,
                        const platform::Place& place,
                        VariableValueMap* ins_map_temp,
                        VariableScope* var_scope, OpFuncNode* op_func_node,
                        std::vector<OpFuncNode>* new_op_func_nodes,
                        bool use_local_scope) {
  auto op_base = op_func_node->operator_base_.get();
  PADDLE_ENFORCE_NOT_NULL(op_base, platform::errors::PreconditionNotMet(
                                       "op_base is null, please pass a valid "
                                       "op_base in apply_data_transform."));

  VariableNameMap new_ins(op_base->Inputs());
  // record the no need transform variable index.
  std::unordered_set<int> no_data_transform_index;

  DataTranferHelper data_transfer_helper(place, var_scope);
  for (auto& var_name_item : *ins_map_temp) {
    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto var = var_name_item.second[i];
      auto& var_name = new_ins[var_name_item.first].at(i);
      const Tensor* tensor_in;
      if (var->IsType<LoDTensor>() || var->IsType<pten::SelectedRows>()) {
        tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      } else if (var->IsType<LoDTensorArray>()) {
        tensor_in =
            static_cast<const Tensor*>(&(var->Get<LoDTensorArray>()[0]));
      } else {
        continue;
      }
      if (!tensor_in->IsInitialized()) {
        continue;
      }
      auto kernel_type_for_var =
          static_cast<const framework::OperatorWithKernel*>(op_base)
              ->GetKernelTypeForVar(var_name_item.first, *tensor_in,
                                    expected_kernel_key);
      // apply data transform
      std::string new_var_name;
      bool is_transferred = data_transfer_helper.apply(
          kernel_type_for_var, expected_kernel_key, var_name, &new_var_name,
          new_op_func_nodes, use_local_scope);

      if (is_transferred) {
        // update RuntimeContext.inputs and original op_func_node inputs
        op_func_node->input_index[var_name_item.first][i] =
            var_scope->VarId(new_var_name);
        var_name_item.second[i] = var_scope->Var(new_var_name);
        new_ins[var_name_item.first][i] = new_var_name;
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

  // NOTE(zhiqiu): UPDATE the corresponding OeratorBase to make it consistent
  // with instruction. (hot fix, it is not good design here)
  op_func_node->operator_base_ =
      std::shared_ptr<OperatorBase>(framework::OpRegistry::CreateOp(
          op_base->Type(), new_ins, op_base->Outputs(), op_base->Attrs()));
  op_func_node->no_data_transform_index = std::move(no_data_transform_index);
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

void HandleComplexGradToRealGrad(const OpFuncNode& op_func_node,
                                 const platform::Place& place,
                                 const VariableNameMap& out_names,
                                 VariableValueMap* out_vars,
                                 VariableScope* var_scope,
                                 std::vector<OpFuncNode>* op_func_nodes,
                                 framework::Scope* local_scope) {
  DataTranferHelper data_transfer_helper(place, var_scope);
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
      auto* var = var_scope->FindVar(orig_var_name);
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
      auto op = TransferDtype(var_name, &new_var_name, src_type, dst_type,
                              var_scope, local_scope);
      data_transfer_helper.RunAndConstructOpFuncNode(op, var_name, new_var_name,
                                                     op_func_nodes);
      data_transfer_helper.RunAndConstructShareNode(new_var_name, var_name,
                                                    op_func_nodes);
    }
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
