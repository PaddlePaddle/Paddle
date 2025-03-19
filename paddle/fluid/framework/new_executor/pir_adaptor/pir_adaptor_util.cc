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

#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

#include "glog/logging.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/vocab/string_array.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace paddle::framework {
std::shared_ptr<ValueExecutionInfo> ValueExecutionInfo::NewChild(Scope* scope) {
  std::shared_ptr<ValueExecutionInfo> info =
      std::make_shared<ValueExecutionInfo>(scope);
  info->parent_ = this;
  info->value_2_var_name_ = this->value_2_var_name_;
  info->var_2_var_name_ = this->var_2_var_name_;
  info->var_name_2_id_ = this->var_name_2_id_;
  info->id_2_var_name_ = this->id_2_var_name_;
  info->var_list_ = this->var_list_;
  return info;
}

void ValueExecutionInfo::Add(::pir::Value value, const std::string& var_name) {
  auto* var = scope_->FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(
      var, common::errors::NotFound("Cannot find %s in scope.", var_name));

  if (value_2_var_name_.count(value) == 0) {
    value_2_var_name_.emplace(value, var_name);
  }

  var_2_var_name_.emplace(var, var_name);

  if (var_name_2_id_.count(var_name) == 0) {
    auto id = var_name_2_id_.size();
    var_name_2_id_.emplace(var_name, id);
    id_2_var_name_.emplace(id, var_name);
    var_list_.push_back(var);
  }

  PADDLE_ENFORCE_EQ(
      var_list_.size(),
      var_name_2_id_.size(),
      common::errors::InvalidArgument(
          "The size of variable_list and var_name_2_id map should be equal"));
}

void ValueExecutionInfo::Rename(const std::string& new_name,
                                const std::string& orig_name) {
  for (auto kv : value_2_var_name_) {
    if (kv.second == orig_name) {
      value_2_var_name_[kv.first] = new_name;
    }
  }

  for (auto kv : var_2_var_name_) {
    if (kv.second == orig_name) {
      var_2_var_name_[kv.first] = new_name;
    }
  }

  for (auto kv : var_name_2_id_) {
    if (kv.first == orig_name) {
      var_name_2_id_.emplace(new_name, kv.second);
      id_2_var_name_[kv.second] = new_name;
    }
  }
  var_name_2_id_.erase(orig_name);
}

int ValueExecutionInfo::GetIdByName(const std::string& name) const {
  auto it = var_name_2_id_.find(name);
  if (it != var_name_2_id_.end()) {
    return it->second;
  }
  return -1;
}

std::string ValueExecutionInfo::GetNameById(int id) const {
  // NOTE(zhiqiu): do not use vec_meta_info_[id].vardesc_->Name() since
  // vec_meta_info_[id] may be nullptr,
  // typically when the target variable is not existed in the original program
  // desc, but created by interpretercore.
  // For example, created and used by d2h_copy or h2d_copy operator.
  auto it = id_2_var_name_.find(id);
  if (it != id_2_var_name_.end()) {
    return it->second;
  }
  return "";
}

Variable* ValueExecutionInfo::GetVarByValue(pir::Value value) const {
  return scope_->FindVar(GetVarName(value));
}

const std::unordered_map<::pir::Value, std::string>&
ValueExecutionInfo::GetValue2VarName() const {
  return value_2_var_name_;
}

void ValueExecutionInfo::AddValue2VarName(::pir::Value value,
                                          const std::string& var_name) {
  value_2_var_name_.emplace(value, var_name);
}

void ValueExecutionInfo::UpdateValue2VarName(::pir::Value value,
                                             const std::string& var_name) {
  value_2_var_name_[value] = var_name;
}

const std::unordered_map<const Variable*, std::string>&
ValueExecutionInfo::GetVar2VarName() const {
  return var_2_var_name_;
}

const std::map<std::string, int>& ValueExecutionInfo::GetVarName2Id() const {
  return var_name_2_id_;
}

const std::unordered_map<int, std::string>& ValueExecutionInfo::GetId2VarName()
    const {
  return id_2_var_name_;
}

const std::vector<Variable*>& ValueExecutionInfo::GetVarList() const {
  return var_list_;
}

void ValueExecutionInfo::ResetVarList(int id, Variable* var) {
  var_list_[id] = var;
}

bool ValueExecutionInfo::HasVar(const std::string& var_name) const {
  auto it = var_name_2_id_.find(var_name);
  if (it != var_name_2_id_.end()) {
    return true;
  }
  return false;
}

bool ValueExecutionInfo::HasValue(::pir::Value value) const {
  auto it = value_2_var_name_.find(value);
  if (it != value_2_var_name_.end()) {
    return true;
  }
  return false;
}

std::string ValueExecutionInfo::GetVarName(::pir::Value value) const {
  auto it = value_2_var_name_.find(value);
  if (it != value_2_var_name_.end()) {
    return it->second;
  }
  VLOG(8) << "can not find var name for value %s", value.impl();
  return "";
}

std::string ValueExecutionInfo::GetVarName(const Variable* var) const {
  auto it = var_2_var_name_.find(var);
  if (it != var_2_var_name_.end()) {
    return it->second;
  }
  return "";
}

int ValueExecutionInfo::GetVarId(::pir::Value value) const {
  auto var_name = GetVarName(value);
  auto it = var_name_2_id_.find(var_name);
  if (it != var_name_2_id_.end()) {
    return it->second;
  }
  return -1;
}

int ValueExecutionInfo::GetVarId(const Variable* var) const {
  auto var_name = GetVarName(var);
  auto it = var_name_2_id_.find(var_name);
  if (it != var_name_2_id_.end()) {
    return it->second;
  }
  return -1;
}
const std::unordered_set<std::string> SpecialOps = {
    paddle::dialect::FeedOp::name(),
    paddle::dialect::FetchOp::name(),
    pir::CombineOp::name(),
    pir::SetParameterOp::name(),
    pir::ParameterOp::name(),
    pir::ConstantOp::name(),
    pir::SliceOp::name(),
    pir::SplitOp::name(),
    paddle::dialect::DataOp::name(),
    pir::ShadowOutputOp::name(),
    paddle::dialect::IfOp::name(),
    paddle::dialect::PyLayerOp::name(),
    paddle::dialect::WhileOp::name(),
    pir::StackCreateOp::name(),
    paddle::dialect::ShareVarOp::name(),
};

Variable* CreateVar(pir::Value value,
                    const std::string& var_name_prefix,
                    bool force_persistable,
                    ValueExecutionInfo* value_exe_info) {
  pir::Operation* def_op = value.defining_op();
  bool is_persistable = false;
  if (def_op->isa<::pir::ParameterOp>()) {
    is_persistable = true;
  } else if (auto attr =
                 value.attribute<pir::BoolAttribute>(kAttrIsPersistable)) {
    is_persistable = attr.data();
  }

  Variable* var = nullptr;
  std::string name = var_name_prefix + "_inner_var_" +
                     std::to_string(value_exe_info->GetVar2VarName().size());

  if (force_persistable || is_persistable) {
    VLOG(6) << "Create var: " << name << " in scope "
            << value_exe_info->GetScope()->root();
    var = const_cast<Scope*>(value_exe_info->GetScope()->root())->Var(name);
  } else {
    VLOG(6) << "Create var: " << name << " in scope "
            << value_exe_info->GetScope();
    var = value_exe_info->GetScope()->Var(name);
  }

  value_exe_info->Add(value, name);

  return var;
}

void CheckInputVars(pir::Operation* op,
                    const std::string& op_name,
                    ValueExecutionInfo* execution_info) {
  size_t input_num = op->num_operands();
  if (input_num > 0) {
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand_source(i);
      if (IsInvalid(value)) {
        PADDLE_ENFORCE_EQ(
            execution_info->HasValue(value),
            true,
            common::errors::PreconditionNotMet(
                "input should in execution_info, [%d] 'th input of [%s] op",
                i,
                op_name));
      }
    }
  }
}

void DeepCopyVariable(const Variable* src_var,
                      Variable** dst_var,
                      ValueExecutionInfo* value_exe_info,
                      uint32_t stack_size,
                      bool is_optional,
                      std::map<const Variable*, Variable*>* src_to_dst_map) {
  if (src_to_dst_map->count(src_var)) {
    (*dst_var) = (*src_to_dst_map)[src_var];
    return;
  }

  if (src_var->IsType<phi::DenseTensor>()) {
    auto& src_tensor = src_var->Get<phi::DenseTensor>();
    auto* tmp_dst_tensor = (*dst_var)->GetMutable<phi::DenseTensor>();
    tmp_dst_tensor->set_lod(src_tensor.lod());
    // NOTE(chenxi67): why add <src_tensor.numel() == 0> ? In some case(e.g.
    // Opresult reserve_space generated by BatchNorm Op), Variable pointer is
    // initialized but the content it hold (DenseTensor for most cases) does not
    // have holder. In this case we only do set_meta but not copy Tensor.
    if (src_tensor.numel() == 0) {
      tmp_dst_tensor->set_meta(src_tensor.meta());
      if (src_tensor.IsInitialized()) {
        tmp_dst_tensor->ResetHolder(
            ::phi::memory_utils::AllocShared(src_tensor.place(), 0u));
      }
      return;
    }
    if (!src_tensor.initialized()) {
      if (is_optional) {
        (*dst_var) = nullptr;
        return;
      } else {
        PADDLE_THROW(
            common::errors::PermissionDenied("DenseTensor shouldn't be null"));
      }
    }
    framework::TensorCopy(src_tensor, src_tensor.place(), tmp_dst_tensor);
  } else if (src_var->IsType<phi::SelectedRows>()) {
    auto& src_slr = src_var->Get<phi::SelectedRows>();
    auto* tmp_dst_slr = (*dst_var)->GetMutable<phi::SelectedRows>();
    tmp_dst_slr->set_rows(src_slr.rows());
    tmp_dst_slr->set_height(src_slr.height());
    auto& src_t = src_slr.value();
    auto* dst_t = tmp_dst_slr->mutable_value();
    if (src_t.numel() == 0) {
      dst_t->set_meta(src_t.meta());
      return;
    }
    if (!src_slr.initialized()) {
      if (is_optional) {
        (*dst_var) = nullptr;
        return;
      } else {
        PADDLE_THROW(
            common::errors::PermissionDenied("SelectedRows shouldn't be null"));
      }
    }
    framework::TensorCopy(src_t, src_t.place(), dst_t);
  } else if (src_var->IsType<phi::TensorArray>()) {
    auto src_tensor_array = src_var->Get<phi::TensorArray>();
    auto* dst_tensor_array = (*dst_var)->GetMutable<phi::TensorArray>();
    if (!src_tensor_array.initialized()) {
      if (is_optional) {
        (*dst_var) = nullptr;
        return;
      } else {
        PADDLE_THROW(
            common::errors::PermissionDenied("TensorArray shouldn't be null"));
      }
    }
    dst_tensor_array->resize(src_tensor_array.size());
    for (size_t i = 0; i < src_tensor_array.size(); ++i) {
      phi::DenseTensor& tmp_dst_tensor = dst_tensor_array->at(i);
      if (src_tensor_array.at(i).numel() == 0) {
        tmp_dst_tensor.set_meta(src_tensor_array.at(i).meta());
        continue;
      }
      framework::TensorCopy(src_tensor_array.at(i),
                            src_tensor_array.at(i).place(),
                            &tmp_dst_tensor);
    }
  } else if (src_var->IsType<VariableRefArray>()) {
    auto src_ref_array = src_var->Get<VariableRefArray>();
    auto* dst_ref_array = (*dst_var)->GetMutable<VariableRefArray>();
    dst_ref_array->clear();
    for (auto src_ref_var : src_ref_array) {
      std::string new_name = "copied_" + std::to_string(stack_size) + '_' +
                             value_exe_info->GetVarName(src_ref_var);
      auto tmp_dst_var = value_exe_info->GetScope()->Var(new_name);
      DeepCopyVariable(src_ref_var,
                       &tmp_dst_var,
                       value_exe_info,
                       stack_size,
                       is_optional,
                       src_to_dst_map);
      dst_ref_array->emplace_back(tmp_dst_var);
    }

  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Output only support DenseTensorType "
        "or SelectedRowsType or TensorArrayType or VariableRefArrayType"));
  }
  (*src_to_dst_map)[src_var] = (*dst_var);
}

void BuildValue(pir::Value value,
                const std::string& var_name_prefix,
                ValueExecutionInfo* value_exe_info) {
  if (!IsInvalid(value)) {
    VLOG(8) << "Value " << value.impl()
            << " is not invalid, so skip build a variable.";
    return;
  }
  Variable* var = nullptr;
  auto& value_2_var_name = value_exe_info->GetValue2VarName();
  if (value_2_var_name.find(value) != value_2_var_name.end()) {
    var = value_exe_info->GetVarByValue(value);
  } else {
    var = CreateVar(value, var_name_prefix, false, value_exe_info);
  }
  // Only support DenseTensor or Vector<DenseTensor>
  if (!value.type() ||
      value.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    var->GetMutable<phi::DenseTensor>();
  } else if (value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    var->GetMutable<phi::SelectedRows>();
  } else if (value.type()
                 .isa<paddle::dialect::AllocatedSparseCooTensorType>()) {
    var->GetMutable<phi::SparseCooTensor>();
  } else if (value.type()
                 .isa<paddle::dialect::AllocatedSparseCsrTensorType>()) {
    var->GetMutable<phi::SparseCsrTensor>();
  } else if (value.type()
                 .isa<paddle::dialect::AllocatedDenseTensorArrayType>()) {
    var->GetMutable<phi::TensorArray>();
  } else if (value.type().isa<pir::StackType>()) {
    var->GetMutable<VariableRefArray>();
  } else if (value.type().isa<pir::VectorType>()) {
    auto tensor_array = var->GetMutable<VariableRefArray>();
    tensor_array->clear();
    for (size_t i = 0; i < value.type().dyn_cast<pir::VectorType>().size();
         i++) {
      PADDLE_ENFORCE(
          value.type()
              .dyn_cast<pir::VectorType>()[i]
              .isa<paddle::dialect::AllocatedDenseTensorType>(),
          common::errors::Fatal("Element of VectorType output only support "
                                "DenseTensorType"));
      auto var_i = CreateVar(value, var_name_prefix, false, value_exe_info);

      var_i->GetMutable<phi::DenseTensor>();
      tensor_array->emplace_back(var_i);
    }
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Output only support DenseTensorType "
        "or SelectedRowsType or VectorType or StackType or SpasrCooTensorType "
        "or SpasreCsrTensorType"));
  }
}

void HandleForSpecialOp(pir::Operation* op,
                        const std::string& var_name_prefix,
                        ValueExecutionInfo* value_exe_info,
                        const ExecutionConfig& execution_config) {
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }

  if (op_name == paddle::dialect::FetchOp::name()) {
    // fetch is a very special op, with no output
    auto fetch_src_name =
        op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString();

    auto fetch_var_name = fetch_src_name + "@fetch";
    auto* var = const_cast<Scope*>(value_exe_info->GetScope()->root())
                    ->Var(fetch_var_name);
    if (op->operand_source(0)
            .type()
            .isa<paddle::dialect::DenseTensorArrayType>()) {
      var->GetMutable<phi::TensorArray>();
    } else {
      var->GetMutable<phi::DenseTensor>();
    }
    auto value = op->result(0);

    value_exe_info->Add(value, fetch_var_name);
  } else if (op_name == paddle::dialect::FeedOp::name() ||
             op_name == paddle::dialect::DataOp::name()) {
    VLOG(6) << "Handle for " << op_name;
    auto value = op->result(0);
    VLOG(6) << "link feed output to feed in variable"
            << value_exe_info->GetScope();

    std::string name =
        op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString();
    Variable* var = value_exe_info->GetScope()->FindVar(name);
    if (var == nullptr) {
      var = value_exe_info->GetScope()->Var(name);
      auto* t = var->GetMutable<phi::DenseTensor>();
      if (op_name == paddle::dialect::DataOp::name()) {
        auto shape = op->attribute<dialect::IntArrayAttribute>("shape");
        auto dim = phi::make_ddim(shape.data().GetData());
        auto dtype = op->attribute<dialect::DataTypeAttribute>("dtype");
        if (!common::contain_unknown_dim(dim)) {
          phi::DenseTensorMeta meta(dtype.data(), dim);
          t->set_meta(meta);
        }
      }
    }
    PADDLE_ENFORCE(
        var,
        common::errors::InvalidArgument("The variable %s should exist", name));

    value_exe_info->Add(value, name);
  } else if (op->isa<pir::CombineOp>()) {
    auto out_value = op->result(0);

    Variable* var = nullptr;
    auto& value_2_var_name = value_exe_info->GetValue2VarName();
    if (value_2_var_name.find(out_value) != value_2_var_name.end()) {
      var = value_exe_info->GetVarByValue(out_value);
    } else {
      var = CreateVar(out_value, var_name_prefix, false, value_exe_info);
    }

    auto tensor_array = var->GetMutable<VariableRefArray>();
    // clear tensor array
    tensor_array->clear();
    size_t input_num = op->num_operands();
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand_source(i);
      PADDLE_ENFORCE_EQ(value_2_var_name.count(value),
                        true,
                        common::errors::PreconditionNotMet(
                            "can not found input of combine op"));
      tensor_array->emplace_back(value_exe_info->GetVarByValue(value));
    }
  } else if (op->isa<pir::SetParameterOp>()) {
    VLOG(6) << "Handle for builtin.set_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();

    auto value = op->operand_source(0);
    // change operand name to param_name
    auto orig_name = value_exe_info->GetValue2VarName().at(value);

    if (param_name == orig_name) {
      return;
    }

    if (value_exe_info->GetScope()->root()->FindVar(param_name) == nullptr) {
      const_cast<Scope*>(value_exe_info->GetScope()->root())
          ->Rename(orig_name, param_name);
      VLOG(6) << "set_parameter rename var: " << orig_name << " -> "
              << param_name;
    }

    value_exe_info->Rename(param_name, orig_name);
  } else if (op->isa<pir::ShadowOutputOp>()) {
    VLOG(6) << "Handle for builtin.shadow_output";
    auto var_name = op->attributes()
                        .at("output_name")
                        .dyn_cast<pir::StrAttribute>()
                        .AsString();

    auto value = op->operand_source(0);

    Scope* scope = const_cast<Scope*>(value_exe_info->GetScope());
    if (!execution_config.used_for_inference) {
      if (auto bool_attr =
              value.attribute<pir::BoolAttribute>(kAttrIsPersistable)) {
        if (bool_attr.data()) {
          VLOG(6) << "Handle for builtin.shadow_output persistable value:"
                  << var_name;
          scope = const_cast<Scope*>(value_exe_info->GetScope()->root());
        }
      }
    }

    // change operand name to param_name
    auto orig_name = value_exe_info->GetValue2VarName().at(value);

    if (var_name == orig_name) {
      return;
    }

    if (value_exe_info->HasVar(var_name)) {
      value_exe_info->UpdateValue2VarName(value, var_name);
    } else {
      if (scope->FindVar(var_name) != nullptr) {
        scope->EraseVars({var_name});
        VLOG(1) << "var " << var_name << " has been removed from scope";
      }
      scope->Rename(orig_name, var_name);
      VLOG(8) << "var " << orig_name << " has been renamed to " << var_name;
      value_exe_info->Rename(var_name, orig_name);
    }
  } else if (op->isa<pir::ParameterOp>()) {
    VLOG(6) << "Handle for builtin.parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();
    auto value = op->result(0);

    value_exe_info->Add(value, param_name);
  } else if (op_name == pir::ConstantOp::name()) {
    VLOG(6) << "Handle for builtin.constant:";
    if (op->isa<pir::ConstantTensorOp>()) {
      auto param_name = op->dyn_cast<pir::ConstantTensorOp>().tensor_name();
      auto value = op->result(0);
      value_exe_info->Add(value, param_name);
    }
  } else if (op->isa<pir::SliceOp>()) {
    VLOG(6) << "Handle for builtin.slice";
    auto out_value = op->result(0);
    auto in_value = op->operand_source(0);
    PADDLE_ENFORCE_EQ(value_exe_info->GetValue2VarName().count(in_value),
                      true,
                      common::errors::PreconditionNotMet(
                          "input of builtin slice not in name map"));

    int index =
        op->attributes().at("index").dyn_cast<pir::Int32Attribute>().data();
    auto in_var = value_exe_info->GetVarByValue(in_value);
    auto variable_array = in_var->Get<VariableRefArray>();

    PADDLE_ENFORCE_EQ(
        value_exe_info->GetVar2VarName().count(variable_array[index]),
        true,
        common::errors::PreconditionNotMet("[%d] the variable in build slice "
                                           "input MUST in variable name map",
                                           index));

    std::string var_name =
        value_exe_info->GetVar2VarName().at(variable_array[index]);
    value_exe_info->AddValue2VarName(out_value, var_name);
  } else if (op->isa<pir::SplitOp>()) {
    VLOG(6) << "Handle for builtin.split";
    auto in_value = op->operand_source(0);
    PADDLE_ENFORCE_EQ(value_exe_info->GetValue2VarName().count(in_value),
                      true,
                      common::errors::PreconditionNotMet(
                          "input of builtin split not in name map"));

    auto in_var = value_exe_info->GetVarByValue(in_value);
    auto variable_array = in_var->Get<VariableRefArray>();

    for (uint64_t idx = 0; idx < variable_array.size(); ++idx) {
      auto out_value = op->result(idx);
      PADDLE_ENFORCE_EQ(
          value_exe_info->GetVar2VarName().count(variable_array[idx]),
          true,
          common::errors::PreconditionNotMet("[%d] the variable in build split "
                                             "input MUST in variable name map",
                                             idx));
      std::string var_name =
          value_exe_info->GetVar2VarName().at(variable_array[idx]);
      value_exe_info->AddValue2VarName(out_value, var_name);
    }
  } else if (op->isa<paddle::dialect::IfOp>()) {
    auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
    for (size_t i = 0; i < if_op->num_results(); ++i) {
      auto if_op_out_value = if_op->result(i);
      BuildValue(if_op_out_value, var_name_prefix, value_exe_info);
    }
  } else if (op->isa<paddle::dialect::PyLayerOp>()) {
    auto pylayer_op = op->dyn_cast<paddle::dialect::PyLayerOp>();

    for (size_t i = 0; i < pylayer_op->num_results(); ++i) {
      auto pylayer_op_out_value = pylayer_op->result(i);
      BuildValue(pylayer_op_out_value, var_name_prefix, value_exe_info);
    }
  } else if (op->isa<paddle::dialect::WhileOp>()) {
    auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();

    for (size_t i = 0; i < while_op->num_results(); ++i) {
      auto while_op_out_value = while_op->result(i);
      BuildValue(while_op_out_value, var_name_prefix, value_exe_info);
    }
  } else if (op->isa<pir::StackCreateOp>()) {
    auto stack_create_op = op->dyn_cast<pir::StackCreateOp>();
    auto stack_value = stack_create_op.stack();
    std::string stack_var_name = var_name_prefix + "(stack)";
    BuildValue(stack_value, stack_var_name, value_exe_info);

    stack_var_name = value_exe_info->GetVarName(stack_value);
    auto inlet_value = stack_create_op.inlet();
    auto outlet_value = stack_create_op.outlet();
    value_exe_info->AddValue2VarName(inlet_value, stack_var_name);
    value_exe_info->AddValue2VarName(outlet_value, stack_var_name);
  } else if (op_name == "pd_op.share_var") {
    VLOG(6) << "Handle for pd_op.share_var";
    auto first_name =
        value_exe_info->GetValue2VarName().at(op->operand_source(0));
    for (size_t idx = 1u; idx < op->num_operands(); ++idx) {
      value_exe_info->UpdateValue2VarName(op->operand_source(idx), first_name);
    }
  }
}

bool IsNeedVarInplace(pir::Operation* op,
                      pir::Value value,
                      std::string op_name) {
  return (value.type().isa<paddle::dialect::DenseTensorArrayType>() ||
          op_name == "pd_op.assign_value_" || op_name == "pd_op.assign_out_" ||
          op_name == "pd_op.coalesce_tensor_");
}

// NOTE(chenxi67): Here, we only perform inplace processing for variables that
// need to be inplaced by var (mostly, whose type is TensorArray or re-Allocated
// Densetensor). For other types of variables, we only share the holder of
// DenseTensor but not the var*. The reason is that vector<DenseTensor> in
// TensorArray (or re-Allocated Densetensor) cannot be shared totally.
void HandleForInplaceVarOp(pir::Operation* op,
                           const std::string& var_name_prefix,
                           ValueExecutionInfo* value_exe_info) {
  if (op->num_results() < 1) return;
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value value = op->result(i);
    if (!IsInvalid(value)) {
      VLOG(8) << "Number " << i << " result of " << op_name
              << " is not invalid, so skip build a variable.";
      continue;
    }
    if (!IsNeedVarInplace(op, value, op_name)) {
      BuildValue(value, var_name_prefix, value_exe_info);
      continue;
    }
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string& inplace_name = yaml_parser.InplaceName(value_name);
      pir::Value inplace_value =
          op->operand_source(yaml_parser.InputName2Id().at(inplace_name));
      std::string var_name = value_exe_info->GetVarName(inplace_value);
      if (var_name != "") {
        VLOG(4) << "inplace: " << value_name << " -> " << inplace_name
                << " (var: " << var_name << ")";
        value_exe_info->AddValue2VarName(value, var_name);
      } else {
        BuildValue(value, var_name_prefix, value_exe_info);
      }
    } else if (yaml_parser.HasView(value_name)) {
      const std::string& view_name = yaml_parser.ViewName(value_name);
      pir::Value view_value =
          op->operand_source(yaml_parser.InputName2Id().at(view_name));
      // const std::string& var_name = value_2_var_name->at(view_value);
      std::string var_name = value_exe_info->GetVarName(view_value);
      if (var_name != "") {
        VLOG(4) << "view: " << value_name << " -> " << view_name
                << " (var: " << var_name << ")";
        value_exe_info->AddValue2VarName(value, var_name);
      } else {
        BuildValue(value, var_name_prefix, value_exe_info);
      }
    } else {
      BuildValue(value, var_name_prefix, value_exe_info);
    }
  }
}

// NOTE(zhiqiu): the persistable is created in inner_scope's root, and other
// is created in inner_scope.
void BuildScope(const pir::Block& block,
                const std::string& var_name_prefix,
                const ExecutionConfig& execution_config,
                ValueExecutionInfo* value_exe_info) {
  VLOG(4) << "***** [before build] scope"
          << "(" << value_exe_info->GetScope() << ") ******\n"
          << GenScopeTreeDebugInfo(
                 const_cast<Scope*>(value_exe_info->GetScope()->root()));

  VLOG(6) << "Start handle keyword blockargument!";
  for (auto& kwarg : block.kwargs()) {
    VLOG(6) << "link keyword blockargument in variable"
            << value_exe_info->GetScope();
    Variable* var = value_exe_info->GetScope()->FindVar(kwarg.first);
    PADDLE_ENFORCE(var,
                   common::errors::InvalidArgument(
                       "The variable %s should exist", kwarg.first));

    value_exe_info->Add(kwarg.second, kwarg.first);
  }
  VLOG(6) << "Finished handle keyword blockargument!";

  for (auto& op : block) {
    std::string op_name = op.name();
    if (op.attributes().count("op_name")) {
      op_name = op.attributes()
                    .at("op_name")
                    .dyn_cast<pir::StrAttribute>()
                    .AsString();
    }
    VLOG(4) << "build op:" << op_name;
    if (SpecialOps.count(op_name)) {
      HandleForSpecialOp(
          &op, var_name_prefix, value_exe_info, execution_config);
      continue;
    }

    CheckInputVars(&op, op_name, value_exe_info);

    if (op.num_results() < 1) continue;
    if (op.attributes().count("is_inplace") != 0 &&
        op.attributes()
            .at("is_inplace")
            .dyn_cast<pir::BoolAttribute>()
            .data()) {
      HandleForInplaceVarOp(&op, var_name_prefix, value_exe_info);
      continue;
    } else {
      for (size_t i = 0; i < op.num_results(); ++i) {
        BuildValue(op.result(i), var_name_prefix, value_exe_info);
      }
    }
  }

  VLOG(4) << "***** [after build] scope"
          << "(" << value_exe_info->GetScope() << ") ******\n"
          << GenScopeTreeDebugInfo(
                 const_cast<Scope*>(value_exe_info->GetScope()->root()));
}

void BuildRuntimeContext(pir::Operation* op,
                         const ValueExecutionInfo& value_exec_info,
                         const paddle::dialect::OpYamlInfoParser& op_yaml_info,
                         RuntimeContext* runtime_ctx) {
  const Scope* inner_scope = value_exec_info.GetScope();
  VLOG(6) << "BuildPhiContext in scope[" << inner_scope << "]";

  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(true);

  auto& name2id = op_yaml_info.InputName2Id();

  std::string fluid_op_name =
      phi::TransToFluidOpName(op_yaml_info.OpRuntimeInfo().kernel_func);

  auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();

  for (auto& name : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(name),
        true,
        common::errors::NotFound("param [%s] MUST in name2id map", name));
    auto index = op_yaml_info.InputName2Id().at(name);
    pir::Value ptr = op->operand_source(index);

    if (!IsInvalid(ptr)) {
      VLOG(8) << "ctx->EmplaceBackInput : an optional input " << name;
      continue;
    }

    auto legacy_attr_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);
    auto in_var_name = value_exec_info.GetVarName(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << name << "\t" << in_var_name;
    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<VariableRefArray>()) {
      for (auto single_var : var->Get<VariableRefArray>()) {
        runtime_ctx->inputs[legacy_attr_name].push_back(
            const_cast<framework::Variable*>(single_var));
      }
    } else {
      runtime_ctx->inputs[legacy_attr_name].push_back(var);
    }
  }

  auto& output_name_list = op_yaml_info.OutputNames();
  for (size_t i = 0; i < output_name_list.size(); ++i) {
    auto name = output_name_list[i];
    pir::Value ptr = op->result(i);
    auto legacy_arg_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);

    if (!IsInvalid(ptr)) {
      VLOG(8) << "ctx->EmplaceBackOutput : an optional output " << name;
      continue;
    }

    auto in_var_name = value_exec_info.GetVarName(ptr);
    VLOG(6) << "ctx->EmplaceBackOutput: " << name << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));

    auto var = inner_scope->FindVar(in_var_name);

    auto type = ptr.type();
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>() ||
        type.isa<paddle::dialect::AllocatedSelectedRowsType>() ||
        type.isa<paddle::dialect::AllocatedSparseCooTensorType>() ||
        type.isa<paddle::dialect::AllocatedSparseCsrTensorType>()) {
      runtime_ctx->outputs[legacy_arg_name] = {var};
    } else if (type.isa<pir::VectorType>()) {
      auto var_ref = var->Get<VariableRefArray>();
      std::vector<Variable*> vec_tmp;
      vec_tmp.reserve(var_ref.size());
      for (size_t k = 0; k < var_ref.size(); ++k) {
        vec_tmp.push_back(const_cast<Variable*>(var_ref[k]));
      }
      runtime_ctx->outputs[legacy_arg_name] = vec_tmp;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType, "
          "AllocatedSparseCooTensorType, AllocatedSparseCsrTensorType, and "
          "pir::vector type"));
    }
  }
}

std::shared_ptr<OperatorBase> BuildOperatorBase(
    pir::Operation* op,
    const ValueExecutionInfo& value_exec_info,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info) {
  paddle::framework::VariableNameMap in_name_map;
  paddle::framework::VariableNameMap out_name_map;
  paddle::framework::AttributeMap attr_map;
  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(true);

  auto& name2id = op_yaml_info.InputName2Id();

  std::string fluid_op_name =
      phi::TransToFluidOpName(op_yaml_info.OpRuntimeInfo().kernel_func);

  auto& op_normalizer = paddle::translator::OpNameNormalizer::instance();

  auto scope = value_exec_info.GetScope();

  // build inputs
  for (auto& name : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(name),
        true,
        common::errors::NotFound("param [%s] MUST in name2id map", name));
    auto index = op_yaml_info.InputName2Id().at(name);
    pir::Value ptr = op->operand_source(index);
    auto legacy_attr_name = op_normalizer.GetLegacyArgName(fluid_op_name, name);

    if (!IsInvalid(ptr)) {
      VLOG(8) << "Push back inputs to VariableNameMap : an optional input "
              << name;
      continue;
    }
    VLOG(6) << "Push back inputs to VariableNameMap : "
            << value_exec_info.GetVarName(ptr);
    in_name_map[legacy_attr_name].push_back(value_exec_info.GetVarName(ptr));
  }

  // build attribute
  auto& op_attr_map = op->attributes();
  auto attr_name_list = op_yaml_info.AttrParams(true);
  for (auto& name : attr_name_list) {
    auto& val = op_attr_map.at(name);
    auto legacy_arg_name = op_normalizer.GetLegacyAttrName(fluid_op_name, name);

    if (val.isa<pir::StrAttribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::StrAttribute>().AsString();
    } else if (val.isa<pir::Int32Attribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::Int32Attribute>().data();
    } else if (val.isa<pir::BoolAttribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::BoolAttribute>().data();
    } else if (val.isa<pir::FloatAttribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::FloatAttribute>().data();
    } else if (val.isa<pir::DoubleAttribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::DoubleAttribute>().data();
    } else if (val.isa<pir::Int64Attribute>()) {
      attr_map[legacy_arg_name] = val.dyn_cast<pir::Int64Attribute>().data();
    } else if (val.isa<pir::ArrayAttribute>()) {
      auto array_list = val.dyn_cast<pir::ArrayAttribute>().AsVector();
      PADDLE_ENFORCE(array_list.size() > 0,
                     common::errors::Fatal("Attribute %s is empty", name));
      if (array_list[0].isa<pir::Int32Attribute>()) {
        std::vector<int> vec_int;
        for (auto attribute : array_list) {
          vec_int.push_back(attribute.dyn_cast<pir::Int32Attribute>().data());
        }
        attr_map[legacy_arg_name] = vec_int;
      } else if (array_list[0].isa<pir::Int64Attribute>()) {
        std::vector<int64_t> vec_int64;
        for (auto attribute : array_list) {
          vec_int64.push_back(
              attribute.dyn_cast<pir::Int64Attribute>().data());  // NOLINT
        }
        attr_map[legacy_arg_name] = vec_int64;
      } else if (array_list[0].isa<pir::BoolAttribute>()) {
        std::vector<bool> vec_bool;
        for (auto attribute : array_list) {
          vec_bool.push_back(attribute.dyn_cast<pir::BoolAttribute>().data());
        }
        attr_map[legacy_arg_name] = vec_bool;
      } else if (array_list[0].isa<pir::FloatAttribute>()) {
        std::vector<float> vec_float;
        for (auto attribute : array_list) {
          vec_float.push_back(
              attribute.dyn_cast<pir::FloatAttribute>().data());  // NOLINT
        }
        attr_map[legacy_arg_name] = vec_float;
      } else if (array_list[0].isa<pir::DoubleAttribute>()) {
        std::vector<double> vec_double;
        for (auto attribute : array_list) {
          vec_double.push_back(
              attribute.dyn_cast<pir::DoubleAttribute>().data());  // NOLINT
        }
        attr_map[legacy_arg_name] = vec_double;
      } else if (array_list[0].isa<pir::StrAttribute>()) {
        std::vector<std::string> vec_string;
        for (auto attribute : array_list) {
          vec_string.push_back(
              attribute.dyn_cast<pir::StrAttribute>().AsString());  // NOLINT
        }
        attr_map[legacy_arg_name] = vec_string;
      } else {
        std::stringstream ss;
        val.Print(ss);
        VLOG(1) << "type not support " << ss.str() << std::endl;
        PADDLE_THROW("Type[%s] in attribute map not support yet", ss.str());
      }
    } else if (val.isa<paddle::dialect::DataTypeAttribute>()) {
      attr_map[legacy_arg_name] = paddle::framework::TransToProtoVarType(
          val.dyn_cast<paddle::dialect::DataTypeAttribute>().data());
    } else {
      std::stringstream ss;
      val.Print(ss);
      VLOG(1) << "type not support " << ss.str() << std::endl;
      PADDLE_THROW("Type[%s] in attribute map not support yet", ss.str());
    }
  }

  // build outputs
  auto& output_name_list = op_yaml_info.OutputNames();
  for (size_t i = 0; i < output_name_list.size(); ++i) {
    pir::Value ptr = op->result(i);
    auto legacy_arg_name =
        op_normalizer.GetLegacyArgName(fluid_op_name, output_name_list[i]);

    if (!IsInvalid(ptr)) {
      VLOG(8) << "Push back outputs to VariableNameMap : an optional output "
              << legacy_arg_name;
      continue;
    }

    if (ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>() ||
        ptr.type().isa<paddle::dialect::AllocatedSelectedRowsType>() ||
        ptr.type().isa<paddle::dialect::AllocatedSparseCooTensorType>() ||
        ptr.type().isa<paddle::dialect::AllocatedSparseCsrTensorType>()) {
      out_name_map[legacy_arg_name].push_back(value_exec_info.GetVarName(ptr));
      VLOG(6) << "Push back outputs to VariableNameMap : "
              << value_exec_info.GetVarName(ptr);
    } else if (ptr.type().isa<pir::VectorType>()) {
      auto var = scope->FindVar(value_exec_info.GetVarName(ptr));
      auto var_ref = var->Get<VariableRefArray>();
      for (size_t k = 0; k < var_ref.size(); ++k) {
        out_name_map[legacy_arg_name].push_back(
            value_exec_info.GetVarName(var_ref[k]));
        VLOG(6) << "Push back outputs to VariableNameMap : "
                << value_exec_info.GetVarName(var_ref[k]);
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "only support AllocatedDenseTensor, AllocatedSelectedRowsType, "
          "AllocatedSparseCooTensorType, AllocatedSparseCsrTensorType  and "
          "pir::vector type"));
    }
  }
  auto& op_info = OpInfoMap::Instance().Get(fluid_op_name);
  auto ptr =
      op_info.Creator()(fluid_op_name, in_name_map, out_name_map, attr_map);

  std::shared_ptr<OperatorBase> res(ptr);
  return res;
}

}  // namespace paddle::framework
