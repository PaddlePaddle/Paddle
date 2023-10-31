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

#include "paddle/fluid/pir/transforms/dtype_transfer_utils.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/op_info.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/core/type.h"

namespace pir {
phi::Kernel* GetKernel(pir::Operation* op, const phi::KernelKey& kernel_key) {
  auto& op_attributes = op->attributes();
  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  auto phi_kernel = new phi::Kernel(kernel_result.kernel);
  return phi_kernel;
}

bool NeedTransformDataType(const phi::DataType& l, const phi::DataType& r) {
  return l != phi::DataType::ALL_DTYPE && r != phi::DataType::ALL_DTYPE &&
         l != r;
}

const phi::DataType GetKernelTypeforVar(pir::Operation* op,
                                        const std::string& var_name,
                                        const phi::DataType& tensor_dtype,
                                        phi::KernelKey* expected_kernel_key) {
  auto phi_kernel = GetKernel(op, *expected_kernel_key);

  bool has_infer_varkernel_fn =
      phi_kernel && phi_kernel->get_kerneltype_forvar_fn_ != nullptr;

  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op->name());

  auto get_kernel_type_for_var_interface =
      op_info.GetInterfaceImpl<paddle::dialect::GetKernelTypeForVarInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      get_kernel_type_for_var_interface,
      phi::errors::NotFound(
          "can not find GetKernelTypeForVarInterface from [%s]", op->name()));

  phi::DataType kernel_dtype_for_var =
      get_kernel_type_for_var_interface->get_kernel_type_for_var_(
          var_name, tensor_dtype, (*expected_kernel_key).dtype());

  //   AttributeMap pir_attrs = op->attributes();
  //   phi::AttributeMap infer_attrs{};

  //   phi::GetKernelTypeForVarContext infer_varkernel_context =
  //       BuildGetKernelTypeForVarContext(expected_kernel_key,
  //                                         pir_attrs,
  //                                         &infer_attrs,
  //                                         has_infer_varkernel_fn);
  //     if (has_infer_varkernel_fn) {
  //       VLOG(2) << "use infer_varkernel_fn to get kernel key for var";
  //       infer_varkernel_context.SetVarName(const_cast<std::string*>(&var_name));
  //       infer_varkernel_context.SetDenseTensor(const_cast<phi::DenseTensor*>(tensor_in));
  //       kernel_type_for_var =
  //           phi_kernel->get_kerneltype_forvar_fn_(&infer_varkernel_context);
  //     }
  return kernel_dtype_for_var;
}

void AddDtypeTransferOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Operation* kernel_op,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  //   phi::KernelKey dtype_transfer_key{
  //         phi::Backend::GPU,
  //         phi::DataLayout::ANY,
  //         expected_kernel_key};
  //   std::unordered_map<std::string, pir::Attribute> attr_map{
  //     {"op_name", pir::StrAttribute::get(ctx, "pd_op.transfer_dtype")},
  //     {"kernel_name", pir::StrAttribute::get(ctx, "transfer_dtype")},
  //     {"kernel_key", dialect::KernelAttribute::get(ctx, shadow_key)}};

  pir::OpInfo phi_kernel_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::PhiKernelOp::name());
  //   pir::Operation* dtype_transfer_kernel_op =
  //       pir::Operation::Create({kernel_op->result(0)},
  //                              attr_map,
  //                              {expected_kernel_key.dtype()},
  //                              phi_kernel_op_info);

  //   block->push_back(dtype_transfer_op);

  //   (*map_op_pair)[op_item] = op;
  //   if (op_item->num_results() > 0) {
  //     for (size_t i = 0; i < shadow_op->num_results(); ++i) {
  //       (*map_value_pair)[op_item->result(i)] =
  //       dtype_transfer_kernel_op->result(i);
  //     }
  //   }
}

// std::shared_ptr<Operation> TransferDtype(const std::string& var_name,
//                                             std::string* new_var_name,
//                                             phi::DataType in_dtype,
//                                             phi::DataType out_dtype,
//                                             framework::VariableScope*
//                                             var_scope, framework::Scope*
//                                             local_scope) {
//   // 1. Generate new_var_name and Initialize it
//   *new_var_name = var_name + "_dtype_" +
//                   std::to_string(static_cast<int>(in_dtype)) + "_" +
//                   std::to_string(static_cast<int>(out_dtype));
//   if (var_scope->HasVar(*new_var_name) &&
//       IsTensorOfVarInitialized(local_scope->FindVar(*new_var_name))) {
//     // already has same var
//     VLOG(4) << "Use cached variable: " << *new_var_name;
//     return nullptr;
//   }

//   auto* ptr = local_scope->Var(*new_var_name);
//   auto var_type = local_scope->FindVar(var_name)->Type();
//   InitializeVariable(ptr, static_cast<proto::VarType::Type>(var_type));
//   VLOG(3) << "Create Variable " << *new_var_name
//           << " locally, which pointer is " << ptr << "Variable Type "
//           << var_type;
//   var_scope->MutableDataTransferAddedVars().emplace_back(*new_var_name,
//                                                          var_type);
//   var_scope->AddVar(*new_var_name, nullptr);

//   // 2. Construct VariableNameMap
//   VariableNameMap in_name_map = {{"X", {var_name}}};
//   VariableNameMap out_name_map = {{"Out", {*new_var_name}}};
//   AttributeMap attr_map;
//   attr_map["in_dtype"] = static_cast<int>(in_dtype);
//   attr_map["out_dtype"] = static_cast<int>(out_dtype);
//   // NOTE(Aurelius84): In whice case use_mkldnn = true?
//   attr_map["use_mkldnn"] = false;

//   // 3. Create transfer_dtype_op
//   std::string op_type("transfer_dtype");
//   auto& op_info = OpInfoMap::Instance().Get(op_type);
//   auto op = std::shared_ptr<OperatorBase>(
//       op_info.Creator()(op_type, in_name_map, out_name_map, attr_map));

//   VLOG(3) << string::Sprintf("Insert %s with %s(%s) -> %s(%s).",
//                              op_type,
//                              var_name,
//                              DataTypeToString(in_dtype),
//                              *new_var_name,
//                              DataTypeToString(out_dtype));
//   return op;
// }

// phi::GetKernelTypeForVarContext BuildGetKernelTypeForVarContext(
//     const phi::KernelKey &kernel_key,
//     const AttributeMap &pir_attrs,
//     phi::AttributeMap *phi_attrs,
//     bool has_infer_varkernel_fn) {

//   if (has_infer_varkernel_fn) {
//     for (auto &attr : pir_attrs) {
//       if(attr.second->second.isa<StrAttribute>()) {
//           (*phi_attrs)[attr.first] = PADDLE_GET_CONST(std::string,
//           attr.second);
//       }
//       else{
//           VLOG(6) << "GetKernelTypeForVarContext currently only use "
//                      "std::string. You add other type if need.";
//       }
//     }
//   }
//   return phi::GetKernelTypeForVarContext(&kernel_key, phi_attrs);
// }

}  // namespace pir
