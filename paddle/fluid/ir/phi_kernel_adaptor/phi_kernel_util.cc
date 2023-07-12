// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/phi/core/enforce.h"

#include "glog/logging.h"

namespace ir {

paddle::framework::Variable* CreateVar(ir::Value value,
                                       const std::string& name,
                                       paddle::framework::Scope* scope,
                                       paddle::framework::Scope* local_scope) {
  Operation* def_op = value.GetDefiningOp();
  bool is_persisable = false;
  if (def_op->attributes().count("is_persisable")) {
    is_persisable = def_op->attributes()
                        .at("is_persisable")
                        .dyn_cast<ir::BoolAttribute>()
                        .data();
  }
  if (is_persisable) {
    VLOG(6) << "Create var: " << name << " in scope " << scope->root();
    return const_cast<paddle::framework::Scope*>(scope->root())->Var(name);
  } else {
    VLOG(6) << "Create var: " << name << " in scope " << local_scope;
    return local_scope->Var(name);
  }
}

void CheckInputVars(
    ir::Operation* op,
    const std::string& op_name,
    const std::unordered_map<ir::Value, std::string>& name_map) {
  size_t input_num = op->num_operands();
  if (input_num > 0) {
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand(i);
      if (value) {
        PADDLE_ENFORCE_NE(
            name_map.find(value),
            name_map.end(),
            phi::errors::PreconditionNotMet(
                "input should in name map, [%d] 'th input of [%s] op",
                i,
                op_name));
      }
    }
  }
}

void BuildValue(ir::Value value,
                paddle::framework::Scope* scope,
                paddle::framework::Scope* local_scope,
                std::unordered_map<ir::Value, std::string>* name_map,
                int& count) {  // NOLINT
  auto inner_local_scope = local_scope != nullptr ? local_scope : scope;
  std::string name;
  if (name_map->find(value) != name_map->end()) {
    name = name_map->at(value);
  } else {
    name = "inner_var_" + std::to_string(count++);
    name_map->emplace(value, name);
  }
  auto var = CreateVar(value, name, scope, inner_local_scope);
  // Only support DenseTensor or Vector<DenseTensor>
  if (!value.type()) {
    var->GetMutable<phi::DenseTensor>();
  } else if (value.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    var->GetMutable<phi::DenseTensor>();
  } else if (value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    var->GetMutable<phi::SelectedRows>();
  } else if (value.type().isa<ir::VectorType>()) {
    auto tensor_array = var->GetMutable<paddle::framework::TensorRefArray>();
    for (size_t i = 0; i < value.type().dyn_cast<ir::VectorType>().size();
         i++) {
      PADDLE_ENFORCE(value.type()
                         .dyn_cast<ir::VectorType>()[i]
                         .isa<paddle::dialect::AllocatedDenseTensorType>(),
                     paddle::platform::errors::Fatal(
                         "Element of VectorType output only support "
                         "DenseTensorType"));
      std::string name_i = "inner_var_" + std::to_string(count++);
      auto var_i = CreateVar(value, name_i, scope, inner_local_scope);
      tensor_array->emplace_back(var_i->GetMutable<phi::DenseTensor>());
    }
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Output only support DenseTensorType or VectorType"));
  }
}

void HandleForSpecialOp(ir::Operation* op,
                        paddle::framework::Scope* scope,
                        paddle::framework::Scope* local_scope,
                        std::unordered_map<ir::Value, std::string>* name_map,
                        int& count) {  // NOLINT
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
  }
  size_t input_num = op->num_operands();

  if (op_name == "pd.fetch") {
    // fetch is a very special op, with no output
    VLOG(6) << "Handle for pd.fetch:";
    auto var = scope->Var("fetch");
    VLOG(6) << "Create var: fetch in scope " << scope;
    auto fetch_list = var->GetMutable<paddle::framework::FetchList>();
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    fetch_list->resize(index + 1);
  }

  if (op_name == "pd.feed") {
    VLOG(6) << "Handle for pd.feed:";
    auto value = op->result(0);
    std::string name = "inner_var_" + std::to_string(count++);
    name_map->emplace(value, name);
    auto var = CreateVar(value, name, scope, local_scope);
    // TODO(phlrain): need to update here, support StringTensor
    auto out_tensor = var->GetMutable<phi::DenseTensor>();

    auto feed_var = scope->Var("feed");
    VLOG(6) << "Create var: feed in scope " << scope;
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    auto feed_list = feed_var->Get<paddle::framework::FeedList>();
    auto& in_tensor = (PADDLE_GET(phi::DenseTensor, feed_list.at(index)));
    out_tensor->ShareDataWith(in_tensor);
  }

  if (op_name == "builtin.combine") {
    VLOG(6) << "Handle for builtin.combine:";
    auto out_value = op->result(0);
    std::string name;
    if (name_map->find(out_value) != name_map->end()) {
      name = name_map->at(out_value);
    } else {
      name = "inner_var_" + std::to_string(count++);
      name_map->emplace(out_value, name);
    }

    auto var = CreateVar(out_value, name, scope, local_scope);
    auto tensor_array = var->GetMutable<paddle::framework::TensorRefArray>();
    // clear tensor array
    tensor_array->clear();

    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand(i);

      PADDLE_ENFORCE_EQ(
          name_map->count(value),
          true,
          phi::errors::PreconditionNotMet("can not found input of combine op"));
      tensor_array->emplace_back(
          &(CreateVar(value, name_map->at(value), scope, local_scope)
                ->Get<phi::DenseTensor>()));
    }
  }

  if (op_name == "builtin.set_parameter") {
    VLOG(6) << "Handle for builtin.set_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();

    auto in_ptr = op->operand(0);
    // change opreand name to param_name

    auto orig_name = name_map->at(in_ptr);
    if (scope->FindVar(param_name) == nullptr) {
      scope->Rename(orig_name, param_name);
    }
    (*name_map)[in_ptr] = param_name;
  }

  if (op_name == "builtin.get_parameter") {
    VLOG(6) << "Handle for builtin.get_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();
    auto out_ptr = op->result(0);
    name_map->emplace(out_ptr, param_name);
  }
}

void HandleForInplaceOp(ir::Operation* op,
                        paddle::framework::Scope* scope,
                        paddle::framework::Scope* local_scope,
                        std::unordered_map<ir::Value, std::string>* name_map,
                        int& count) {  // NOLINT
  if (op->num_results() < 1) return;
  ir::IrContext* ctx = ir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
  }
  VLOG(4) << "HandleForInplaceOp: " << op_name;
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_());

  for (size_t i = 0; i < op->num_results(); ++i) {
    ir::Value value = op->result(i);
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      std::string inplace_name = yaml_parser.InplaceName(value_name);
      ir::Value inplace_value =
          op->operand(yaml_parser.InputName2Id().at(inplace_name));
      std::string var_name = name_map->at(inplace_value);
      VLOG(4) << "inplace: " << value_name << " -> " << inplace_name
              << " (var: " << var_name << ")";
      name_map->emplace(value, var_name);
    } else {
      BuildValue(value, scope, local_scope, name_map, count);
    }
  }
}

void BuildScope(const ir::Block& block,
                paddle::framework::Scope* scope,
                paddle::framework::Scope* local_scope,
                std::unordered_map<ir::Value, std::string>* name_map) {
  VLOG(4) << "***** [before build] scope: ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(scope->root()));
  // NOTE(zhiqiu): if use local_scope (local_scope != nullptr), the persistable
  // is created in scope , and other is created in local_scope.
  auto inner_local_scope = local_scope != nullptr ? local_scope : scope;
  VLOG(6) << "Build: scope [" << scope << "] inner_local_scope ["
          << inner_local_scope << "]";

  // int count = name_map->size();
  int count = inner_local_scope->Size();
  for (auto it = block.begin(); it != block.end(); ++it) {
    ir::Operation* op = *it;

    std::string op_name = op->name();
    if (op->attributes().count("op_name")) {
      op_name =
          op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
    }

    VLOG(4) << "BuildScope for :" << op_name;

    if (op_name == "pd.feed" || op_name == "pd.fetch" ||
        op_name == "builtin.combine" || op_name == "builtin.set_parameter" ||
        op_name == "builtin.get_parameter") {
      VLOG(4) << "HandleForSpecialOp: " << op_name;
      HandleForSpecialOp(op, scope, inner_local_scope, name_map, count);
      continue;
    }

    CheckInputVars(op, op_name, *name_map);

    if (op->num_results() < 1) continue;
    if (op->attributes().count("is_inplace") != 0 &&
        op->attributes()
            .at("is_inplace")
            .dyn_cast<ir::BoolAttribute>()
            .data()) {
      HandleForInplaceOp(op, scope, inner_local_scope, name_map, count);
      continue;
    } else {
      for (size_t i = 0; i < op->num_results(); ++i) {
        BuildValue(op->result(i), scope, local_scope, name_map, count);
      }
    }
  }

  VLOG(4) << "***** [after build] scope: ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(scope->root()));
}

}  // namespace ir
