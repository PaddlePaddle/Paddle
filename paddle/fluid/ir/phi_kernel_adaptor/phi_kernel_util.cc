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

#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/utils.h"
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
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_parser.h"
#include "paddle/fluid/ir/dialect/paddle_kernel_dialect/ir/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/paddle_kernel_dialect/ir/kernel_type.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/phi/core/enforce.h"

#include "glog/logging.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/operator.h"

namespace ir {

void AddNewData(ir::Value value,
                std::string name,
                paddle::framework::Variable* var,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id,
                std::vector<paddle::framework::Variable*>* variable_list) {
  if (value_2_var_name->count(value) == 0) {
    value_2_var_name->emplace(value, name);
  }

  variable_2_var_name->emplace(var, name);
  if (var_name_2_id->count(name) == 0) {
    auto id = var_name_2_id->size();
    var_name_2_id->emplace(name, id);
    variable_list->push_back(var);
  }
  PADDLE_ENFORCE_EQ(
      variable_list->size(),
      var_name_2_id->size(),
      paddle::platform::errors::InvalidArgument(
          "The size of variable_list and var_name_2_id map should be equal"));
}

void RenameData(ir::Value value,
                std::string new_name,
                std::string orig_name,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id) {
  (*value_2_var_name)[value] = new_name;

  for (auto kv : (*value_2_var_name)) {
    if (kv.second == orig_name) {
      (*value_2_var_name)[kv.first] = new_name;
    }
  }

  for (auto kv : (*variable_2_var_name)) {
    if (kv.second == orig_name) {
      (*variable_2_var_name)[kv.first] = new_name;
    }
  }

  for (auto kv : *(var_name_2_id)) {
    if (kv.first == orig_name) {
      var_name_2_id->emplace(new_name, kv.second);
    }
  }
  var_name_2_id->erase(orig_name);
}

using VariableNameMap =
    std::unordered_map<const paddle::framework::Variable*, std::string>;

paddle::framework::Variable* CreateVar(
    ir::Value value,
    paddle::framework::Scope* inner_scope,
    const std::string& var_name_prefix,
    bool force_persisable,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  Operation* def_op = value.GetDefiningOp();
  bool is_persisable = false;
  if (def_op->name() == "builtin.set_parameter") {
    is_persisable = true;
  }

  paddle::framework::Variable* var = nullptr;

  std::string name = var_name_prefix + "_inner_var_" +
                     std::to_string(variable_2_var_name->size());

  if (force_persisable || is_persisable) {
    VLOG(6) << "Create var: " << name << " in scope " << inner_scope->root();
    var = const_cast<paddle::framework::Scope*>(inner_scope->root())->Var(name);
  } else {
    VLOG(6) << "Create var: " << name << " in scope " << inner_scope;
    var = inner_scope->Var(name);
  }
  AddNewData(value,
             name,
             var,
             value_2_var_name,
             variable_2_var_name,
             var_name_2_id,
             variable_list);
  return var;
}

void CheckInputVars(
    ir::Operation* op,
    const std::string& op_name,
    const std::unordered_map<ir::Value, std::string>& value_2_var_name) {
  size_t input_num = op->num_operands();
  if (input_num > 0) {
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand_source(i);
      if (value) {
        PADDLE_ENFORCE_NE(
            value_2_var_name.find(value),
            value_2_var_name.end(),
            phi::errors::PreconditionNotMet(
                "input should in name map, [%d] 'th input of [%s] op",
                i,
                op_name));
      }
    }
  }
}

void BuildValue(ir::Value value,
                paddle::framework::Scope* inner_scope,
                const std::string& var_name_prefix,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id,
                std::vector<paddle::framework::Variable*>* variable_list) {
  paddle::framework::Variable* var = nullptr;
  if (value_2_var_name->find(value) != value_2_var_name->end()) {
    var = inner_scope->FindVar(value_2_var_name->at(value));
  } else {
    var = CreateVar(value,
                    inner_scope,
                    var_name_prefix,
                    false,
                    value_2_var_name,
                    variable_2_var_name,
                    var_name_2_id,
                    variable_list);
  }
  // Only support DenseTensor or Vector<DenseTensor>
  if (!value.type()) {
    var->GetMutable<phi::DenseTensor>();
  } else if (value.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    var->GetMutable<phi::DenseTensor>();
  } else if (value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    var->GetMutable<phi::SelectedRows>();
  } else if (value.type().isa<ir::VectorType>()) {
    auto tensor_array = var->GetMutable<paddle::framework::VariableRefArray>();
    for (size_t i = 0; i < value.type().dyn_cast<ir::VectorType>().size();
         i++) {
      PADDLE_ENFORCE(value.type()
                         .dyn_cast<ir::VectorType>()[i]
                         .isa<paddle::dialect::AllocatedDenseTensorType>(),
                     paddle::platform::errors::Fatal(
                         "Element of VectorType output only support "
                         "DenseTensorType"));
      auto var_i = CreateVar(value,
                             inner_scope,
                             var_name_prefix,
                             false,
                             value_2_var_name,
                             variable_2_var_name,
                             var_name_2_id,
                             variable_list);

      var_i->GetMutable<phi::DenseTensor>();
      tensor_array->emplace_back(var_i);
    }
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Output only support DenseTensorType or VectorType"));
  }
}

void HandleForSpecialOp(
    ir::Operation* op,
    paddle::framework::Scope* inner_scope,
    const std::string& var_name_prefix,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().AsString();
  }

  if (op_name == "pd.fetch") {
    // fetch is a very special op, with no output
    auto fetch_src_name =
        op->attributes().at("name").dyn_cast<ir::StrAttribute>().AsString();

    auto fetch_var_name = fetch_src_name + "@fetch";
    auto* var = const_cast<paddle::framework::Scope*>(inner_scope->root())
                    ->Var(fetch_var_name);
    var->GetMutable<phi::DenseTensor>();
    auto value = op->result(0);

    AddNewData(value,
               fetch_var_name,
               var,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id,
               variable_list);
  }

  if (op_name == "pd.feed") {
    auto value = op->result(0);
    VLOG(6) << "link feed output to feed in variable" << inner_scope;

    std::string name =
        op->attributes().at("name").dyn_cast<ir::StrAttribute>().AsString();
    paddle::framework::Variable* var = inner_scope->FindVar(name);
    PADDLE_ENFORCE(var,
                   paddle::platform::errors::InvalidArgument(
                       "The variable %s shoud exist", name));

    AddNewData(value,
               name,
               var,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id,
               variable_list);
  }

  if (op_name == "pd.data") {
    VLOG(6) << "Handle for pd.data";
    auto var_name =
        op->attributes().at("name").dyn_cast<ir::StrAttribute>().AsString();

    auto value = op->result(0);

    paddle::framework::Variable* var = inner_scope->FindVar(var_name);
    PADDLE_ENFORCE(var,
                   paddle::platform::errors::InvalidArgument(
                       "The variable %s shoud exist", var_name));

    AddNewData(value,
               var_name,
               var,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id,
               variable_list);
  }

  if (op_name == "builtin.combine") {
    auto out_value = op->result(0);

    paddle::framework::Variable* var = nullptr;
    if (value_2_var_name->find(out_value) != value_2_var_name->end()) {
      var = inner_scope->FindVar(value_2_var_name->at(out_value));
    } else {
      var = CreateVar(out_value,
                      inner_scope,
                      var_name_prefix,
                      false,
                      value_2_var_name,
                      variable_2_var_name,
                      var_name_2_id,
                      variable_list);
    }

    auto tensor_array = var->GetMutable<paddle::framework::VariableRefArray>();
    // clear tensor array
    tensor_array->clear();
    size_t input_num = op->num_operands();
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand_source(i);
      PADDLE_ENFORCE_EQ(
          value_2_var_name->count(value),
          true,
          phi::errors::PreconditionNotMet("can not found input of combine op"));
      tensor_array->emplace_back(
          inner_scope->FindVar(value_2_var_name->at(value)));
    }
  }

  if (op_name == "builtin.set_parameter") {
    VLOG(6) << "Handle for builtin.set_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .AsString();

    auto value = op->operand_source(0);
    // change opreand name to param_name
    auto orig_name = value_2_var_name->at(value);

    PADDLE_ENFORCE_NE(
        param_name,
        orig_name,
        phi::errors::PreconditionNotMet(
            "SetParamer param name should not equal with var name"));

    if (inner_scope->root()->FindVar(param_name) == nullptr) {
      const_cast<paddle::framework::Scope*>(inner_scope->root())
          ->Rename(orig_name, param_name);
      VLOG(6) << "set_parameter rename var: " << orig_name << " -> "
              << param_name;
    }

    RenameData(value,
               param_name,
               orig_name,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id);
  }

  if (op_name == "pd.shadow_output") {
    VLOG(6) << "Handle for pd.shadow_ouptut";
    auto var_name =
        op->attributes().at("name").dyn_cast<ir::StrAttribute>().AsString();

    auto value = op->operand_source(0);
    // change opreand name to param_name
    auto orig_name = value_2_var_name->at(value);

    if (inner_scope->root()->FindVar(var_name) == nullptr) {
      const_cast<paddle::framework::Scope*>(inner_scope->root())
          ->Rename(orig_name, var_name);
    }
    RenameData(value,
               var_name,
               orig_name,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id);
  }

  if (op_name == "builtin.get_parameter") {
    VLOG(6) << "Handle for builtin.get_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .AsString();
    auto value = op->result(0);

    paddle::framework::Variable* var = inner_scope->FindVar(param_name);
    AddNewData(value,
               param_name,
               var,
               value_2_var_name,
               variable_2_var_name,
               var_name_2_id,
               variable_list);
  }

  if (op_name == "builtin.slice") {
    VLOG(6) << "Handle for builtin.slice";
    auto out_value = op->result(0);
    auto in_value = op->operand_source(0);
    PADDLE_ENFORCE_EQ(value_2_var_name->count(in_value),
                      true,
                      phi::errors::PreconditionNotMet(
                          "input of buildin slice not in name map"));

    int index =
        op->attributes().at("index").dyn_cast<ir::Int32Attribute>().data();
    auto in_var = inner_scope->FindVar(value_2_var_name->at(in_value));
    auto variable_array = in_var->Get<paddle::framework::VariableRefArray>();

    PADDLE_ENFORCE_EQ(
        variable_2_var_name->count(variable_array[index]),
        true,
        phi::errors::PreconditionNotMet("[%d] the variable in build slice "
                                        "input MUST in variable name map",
                                        index));

    std::string var_name = variable_2_var_name->at(variable_array[index]);
    value_2_var_name->emplace(out_value, var_name);
  }

  if (op_name == "builtin.split") {
    VLOG(6) << "Handle for builtin.split";
    auto in_value = op->operand_source(0);
    PADDLE_ENFORCE_EQ(value_2_var_name->count(in_value),
                      true,
                      phi::errors::PreconditionNotMet(
                          "input of buildin split not in name map"));

    auto in_var = inner_scope->FindVar(value_2_var_name->at(in_value));
    auto variable_array = in_var->Get<paddle::framework::VariableRefArray>();

    for (uint64_t idx = 0; idx < variable_array.size(); ++idx) {
      auto out_value = op->result(idx);
      PADDLE_ENFORCE_EQ(
          variable_2_var_name->count(variable_array[idx]),
          true,
          phi::errors::PreconditionNotMet("[%d] the variable in build split "
                                          "input MUST in variable name map",
                                          idx));
      std::string var_name = variable_2_var_name->at(variable_array[idx]);
      value_2_var_name->emplace(out_value, var_name);
    }
  }
}

void HandleForInplaceOp(
    ir::Operation* op,
    paddle::framework::Scope* inner_scope,
    const std::string& var_name_prefix,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  if (op->num_results() < 1) return;
  ir::IrContext* ctx = ir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().AsString();
  }

  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_());

  for (size_t i = 0; i < op->num_results(); ++i) {
    ir::Value value = op->result(i);
    if (value.type().storage() == nullptr) {
      continue;
    }
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string& inplace_name = yaml_parser.InplaceName(value_name);
      ir::Value inplace_value =
          op->operand_source(yaml_parser.InputName2Id().at(inplace_name));
      std::string var_name = value_2_var_name->at(inplace_value);
      VLOG(4) << "inplace: " << value_name << " -> " << inplace_name
              << " (var: " << var_name << ")";
      value_2_var_name->emplace(value, var_name);
    } else if (yaml_parser.HasView(value_name)) {
      const std::string& view_name = yaml_parser.ViewName(value_name);
      ir::Value view_value =
          op->operand_source(yaml_parser.InputName2Id().at(view_name));
      const std::string& var_name = value_2_var_name->at(view_value);
      VLOG(4) << "view: " << value_name << " -> " << view_name
              << " (var: " << var_name << ")";
      value_2_var_name->emplace(value, var_name);
    } else {
      BuildValue(value,
                 inner_scope,
                 var_name_prefix,
                 value_2_var_name,
                 variable_2_var_name,
                 var_name_2_id,
                 variable_list);
    }
  }
}

// NOTE(zhiqiu): the persistable is created in inner_scope's root, and other is
// created in inner_scope.
void BuildScope(const ir::Block& block,
                paddle::framework::Scope* inner_scope,
                const std::string& var_name_prefix,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id,
                std::vector<paddle::framework::Variable*>* variable_list) {
  VLOG(4) << "***** [before build] scope"
          << "(" << inner_scope << ") ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(inner_scope->root()));

  for (auto op : block) {
    std::string op_name = op->name();
    if (op->attributes().count("op_name")) {
      op_name = op->attributes()
                    .at("op_name")
                    .dyn_cast<ir::StrAttribute>()
                    .AsString();
    }
    VLOG(4) << "build op:" << op_name;

    if (op_name == "pd.feed" || op_name == "pd.fetch" ||
        op_name == "builtin.combine" || op_name == "builtin.set_parameter" ||
        op_name == "builtin.get_parameter" || op_name == "builtin.slice" ||
        op_name == "builtin.split" || op_name == "pd.data" ||
        op_name == "pd.shadow_output") {
      HandleForSpecialOp(op,
                         inner_scope,
                         var_name_prefix,
                         value_2_var_name,
                         variable_2_var_name,
                         var_name_2_id,
                         variable_list);
      continue;
    }

    CheckInputVars(op, op_name, *value_2_var_name);

    if (op->num_results() < 1) continue;
    if (op->attributes().count("is_inplace") != 0 &&
        op->attributes()
            .at("is_inplace")
            .dyn_cast<ir::BoolAttribute>()
            .data()) {
      HandleForInplaceOp(op,
                         inner_scope,
                         var_name_prefix,
                         value_2_var_name,
                         variable_2_var_name,
                         var_name_2_id,
                         variable_list);
      continue;
    } else {
      for (size_t i = 0; i < op->num_results(); ++i) {
        BuildValue(op->result(i),
                   inner_scope,
                   var_name_prefix,
                   value_2_var_name,
                   variable_2_var_name,
                   var_name_2_id,
                   variable_list);
      }
    }
  }

  VLOG(4) << "***** [after build] scope"
          << "(" << inner_scope << ") ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(inner_scope->root()));
}

void BuildRuntimeContext(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    paddle::framework::Scope* local_scope,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    paddle::framework::RuntimeContext* runtime_ctx) {
  paddle::framework::Scope* inner_scope =
      local_scope != nullptr ? local_scope : scope;
  VLOG(6) << "BuildPhiContext in scope[" << scope << "] inner_scope["
          << inner_scope << "]";

  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(true);

  auto& name2id = op_yaml_info.InputName2Id();

  auto pd_op_name =
      op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().AsString();
  auto fluid_op_name = pd_op_name.substr(3);  // pd_op_name start with "pd.xxx"

  if (fluid_op_name == "c_broadcast_") {
    fluid_op_name = fluid_op_name.substr(0, fluid_op_name.length() - 1);
  }

  auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();

  for (auto& name : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(name),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", name));
    auto index = op_yaml_info.InputName2Id().at(name);
    ir::Value ptr = op->operand_source(index);

    auto in_var_name = name_map.at(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << name << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    std::vector<paddle::framework::Variable*> vec_tmp = {var};
    auto legacy_attr_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);

    runtime_ctx->inputs[legacy_attr_name].push_back(var);
  }

  auto& output_name_list = op_yaml_info.OutputNames();
  for (size_t i = 0; i < output_name_list.size(); ++i) {
    auto name = output_name_list[i];
    ir::Value ptr = op->result(i);

    auto in_var_name = name_map.at(ptr);
    VLOG(6) << "ctx->EmplaceBackOutput: " << name << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));

    auto var = inner_scope->FindVar(in_var_name);

    auto type = ptr.type();
    auto legacy_arg_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>() ||
        type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
      runtime_ctx->outputs[legacy_arg_name] = {var};
    } else if (type.isa<ir::VectorType>()) {
      auto var_ref = var->Get<paddle::framework::VariableRefArray>();
      std::vector<paddle::framework::Variable*> vec_tmp;
      vec_tmp.reserve(var_ref.size());
      for (size_t k = 0; k < var_ref.size(); ++k) {
        vec_tmp.push_back(const_cast<paddle::framework::Variable*>(var_ref[k]));
      }
      runtime_ctx->outputs[legacy_arg_name] = vec_tmp;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType  and "
          "ir::vector type"));
    }
  }
}

std::shared_ptr<paddle::framework::OperatorBase> BuildOperatorBase(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name,
    const paddle::framework::Scope* scope) {
  paddle::framework::VariableNameMap in_name_map;
  paddle::framework::VariableNameMap out_name_map;
  paddle::framework::AttributeMap attr_map;
  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(true);

  auto& name2id = op_yaml_info.InputName2Id();

  auto pd_op_name =
      op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().AsString();
  auto fluid_op_name = pd_op_name.substr(3);  // pd_op_name start with "pd.xxx"

  auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();

  for (auto& name : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(name),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", name));
    auto index = op_yaml_info.InputName2Id().at(name);
    ir::Value ptr = op->operand_source(index);

    auto in_var_name = name_map.at(ptr);

    auto legacy_attr_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);
    in_name_map[legacy_attr_name].push_back(in_var_name);
  }

  // build attribute
  auto& op_attr_map = op->attributes();
  auto attr_name_list = op_yaml_info.AttrParams(true);
  for (auto& name : attr_name_list) {
    auto& val = op_attr_map.at(name);

    if (val.isa<ir::StrAttribute>()) {
      attr_map[name] = val.dyn_cast<ir::StrAttribute>().AsString();
    } else if (val.isa<ir::Int32Attribute>()) {
      attr_map[name] = val.dyn_cast<ir::Int32Attribute>().data();
    } else if (val.isa<ir::BoolAttribute>()) {
      attr_map[name] = val.dyn_cast<ir::BoolAttribute>().data();
    } else if (val.isa<ir::FloatAttribute>()) {
      attr_map[name] = val.dyn_cast<ir::FloatAttribute>().data();
    } else if (val.isa<ir::DoubleAttribute>()) {
      attr_map[name] = val.dyn_cast<ir::DoubleAttribute>().data();
    } else if (val.isa<ir::Int64Attribute>()) {
      attr_map[name] = val.dyn_cast<ir::Int64Attribute>().data();
    } else {
      std::stringstream ss;
      val.Print(ss);
      VLOG(1) << "type not support " << ss.str() << std::endl;
      PADDLE_THROW("Type[%s] in attribute map not support yet", ss.str());
    }
  }

  auto& output_name_list = op_yaml_info.OutputNames();
  for (size_t i = 0; i < output_name_list.size(); ++i) {
    auto name = output_name_list[i];
    ir::Value ptr = op->result(i);

    auto out_var_name = name_map.at(ptr);

    auto type = ptr.type();
    auto legacy_arg_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>() ||
        type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
      out_name_map[legacy_arg_name].push_back(out_var_name);
    } else if (type.isa<ir::VectorType>()) {
      auto var = scope->FindVar(out_var_name);
      auto var_ref = var->Get<paddle::framework::VariableRefArray>();
      for (size_t k = 0; k < var_ref.size(); ++k) {
        PADDLE_ENFORCE(variable_2_var_name.count(var_ref[k]),
                       "Variable MUST in variable_2_var_name map");
        out_name_map[legacy_arg_name].push_back(
            variable_2_var_name.at(var_ref[k]));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType  and "
          "ir::vector type"));
    }
  }

  auto* op_info =
      paddle::framework::OpInfoMap::Instance().GetNullable(fluid_op_name);
  if (op_info == nullptr && (fluid_op_name.back() == '_')) {
    op_info = paddle::framework::OpInfoMap::Instance().GetNullable(
        fluid_op_name.substr(0, fluid_op_name.length() - 1));
  }
  PADDLE_ENFORCE_NOT_NULL(
      op_info,
      paddle::platform::errors::NotFound("Operator (%s) is not registered.",
                                         fluid_op_name));

  auto ptr =
      op_info->Creator()(fluid_op_name, in_name_map, out_name_map, attr_map);

  std::shared_ptr<paddle::framework::OperatorBase> res(ptr);
  return res;
}

}  // namespace ir
