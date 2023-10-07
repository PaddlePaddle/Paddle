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

#pragma once

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"

#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/core/type_name.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {

class CondInstruction;
class ValueExecutionInfo {
 public:
  explicit ValueExecutionInfo(Scope* scope) : scope_(scope) {}

  const ValueExecutionInfo* Parent() const { return parent_; }

  Scope* GetScope() { return scope_; }

  void Add(::pir::Value value, std::string var_name);

  void Rename(pir::Value value, std::string new_name, std::string orig_name);

  int GetIdByName(const std::string& name) const;

  std::string GetNameById(int id) const;

  const std::unordered_map<::pir::Value, std::string>& GetValue2VarName()
      const {
    return value_2_var_name_;
  }

  void AddValue2VarName(::pir::Value value, const std::string& var_name) {
    value_2_var_name_.emplace(value, var_name);
  }

  const std::unordered_map<const paddle::framework::Variable*, std::string>&
  GetVar2VarName() const {
    return var_2_var_name_;
  }

  const std::map<std::string, int>& GetVarName2Id() const {
    return var_name_2_id_;
  }

  const std::unordered_map<int, std::string>& GetId2VarName() const {
    return id_2_var_name_;
  }

  const std::vector<Variable*>& GetVarList() const { return var_list_; }

  void ResetVarList(int id, Variable* var) { var_list_[id] = var; }

  friend class CondInstruction;

 private:
  std::shared_ptr<ValueExecutionInfo> NewChild(Scope* scope);

  ValueExecutionInfo* parent_{nullptr};  // not owned

  Scope* scope_{nullptr};  // not owned

  std::unordered_map<::pir::Value, std::string> value_2_var_name_;

  std::unordered_map<const paddle::framework::Variable*, std::string>
      var_2_var_name_;

  std::map<std::string, int> var_name_2_id_;

  std::unordered_map<int, std::string> id_2_var_name_;

  std::vector<Variable*> var_list_;
};

}  // namespace framework
}  // namespace paddle

namespace pir {

// NOTE(zhangbo): Some operators of Paddle support optional inputs or outputs,
// representing whether the input or output exists. In the Pir, whether the
// value itself is empty or the type it holds is empty is used to indicate
// whether the input or output exists.
inline bool IsInvalid(pir::Value value) {
  if ((!value) || (!value.type())) {
    return false;
  }
  return true;
}

void BuildScope(
    const pir::Block& block,
    const std::string& var_name_prefix,
    std::map<pir::Block*, paddle::framework::Scope*>* sub_blocks,
    paddle::framework::ValueExecutionInfo* value_exe_info = nullptr);

void BuildRuntimeContext(
    pir::Operation* op,
    const std::unordered_map<pir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    paddle::framework::Scope* local_scope,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    paddle::framework::RuntimeContext* runtime_ctx);

std::shared_ptr<paddle::framework::OperatorBase> BuildOperatorBase(
    pir::Operation* op,
    const std::unordered_map<pir::Value, std::string>& name_map,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name,
    const paddle::framework::Scope* scope);

template <typename Context,
          typename InType,
          typename OutType,
          typename InListType,
          typename OutListType,
          bool is_kernel>
void BuildPhiContext(
    pir::Operation* op,
    const std::unordered_map<pir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    paddle::framework::Scope* local_scope,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    Context* ctx) {
  paddle::framework::Scope* inner_scope =
      local_scope != nullptr ? local_scope : scope;
  VLOG(6) << "Build " << get_type_name<Context>() << " in scope[" << scope
          << "] inner_scope[" << inner_scope << "]";

  auto attr_map = op->attributes();

  // EmplaceBackInputs
  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(is_kernel);
  auto& name2id = op_yaml_info.InputName2Id();
  for (auto& t : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", t));

    pir::Value ptr = op->operand_source(op_yaml_info.InputName2Id().at(t));

    if (!IsInvalid(ptr)) {
      if (op_yaml_info.GetInputType(op_yaml_info.InputName2Id().at(t)) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        InListType optional_inputs;
        ctx->EmplaceBackInputs(optional_inputs);
      } else {
        phi::DenseTensor* temp = nullptr;
        InType optional_input(temp);
        ctx->EmplaceBackInput(optional_input);
      }
      VLOG(8) << "ctx->EmplaceBackInput : an optioanl input " << t;
      continue;
    }

    auto in_var_name = name_map.at(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
      ctx->EmplaceBackInput(InType(tensor_in));
    } else if (var->IsType<paddle::framework::VariableRefArray>()) {
      InListType inputs;
      auto& variable_array = var->Get<paddle::framework::VariableRefArray>();
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          inputs.emplace_back(InType(const_cast<phi::DenseTensor*>(
              &(variable_array[i]->Get<phi::DenseTensor>()))));
        } else if (variable_array[i]->IsType<phi::SelectedRows>()) {
          inputs.emplace_back(InType(const_cast<phi::SelectedRows*>(
              &(variable_array[i]->Get<phi::SelectedRows>()))));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Only support Vector<DenseTensor> and vector<SelectedRows> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      ctx->EmplaceBackInputs(inputs);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                              var->Type()));
    }
  }

  // EmplaceBackAttributes
  auto& vec_kernel_fn_attr_params = op_yaml_info.AttrParams(is_kernel);
  for (auto& t : vec_kernel_fn_attr_params) {
    if (name2id.count(t)) {
      // tensor attribute, get information from input
      pir::Value ptr = op->operand_source(name2id.at(t));

      auto in_var_name = name_map.at(ptr);

      auto& tensor_attr_type = op_yaml_info.TensorAttrTypeName(t);
      VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t" << in_var_name;
      if (tensor_attr_type == "paddle::dialect::IntArrayAttribute") {
        if (ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
          phi::Attribute attr = phi::TensorRef(
              &(inner_scope->FindVar(in_var_name)->Get<phi::DenseTensor>()));
          ctx->EmplaceBackAttr(attr);
        } else if (ptr.type().isa<pir::VectorType>()) {
          auto& tensor_array = inner_scope->FindVar(in_var_name)
                                   ->Get<paddle::framework::VariableRefArray>();
          if (tensor_array.size() == 1) {
            phi::Attribute attr =
                phi::TensorRef(&(tensor_array[0]->Get<phi::DenseTensor>()));
            ctx->EmplaceBackAttr(attr);
          } else {
            std::vector<phi::TensorRef> vec_ref;
            for (size_t i = 0; i < tensor_array.size(); ++i) {
              vec_ref.emplace_back(
                  phi::TensorRef(&(tensor_array[i]->Get<phi::DenseTensor>())));
            }
            ctx->EmplaceBackAttr(vec_ref);
          }
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              " [%s] only support dense tensor and vector type  ",
              tensor_attr_type));
        }
      } else if (tensor_attr_type == "paddle::dialect::ScalarAttribute") {
        phi::Attribute attr = phi::TensorRef(
            &(inner_scope->FindVar(in_var_name)->Get<phi::DenseTensor>()));

        ctx->EmplaceBackAttr(attr);
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                tensor_attr_type));
      }

      continue;
    }

    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "paddle::dialect::IntArrayAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::DataTypeAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
    } else if (attr_type_name == "pir::Int32Attribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<pir::Int32Attribute>().data());
    } else if (attr_type_name == "pir::Int64Attribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<pir::Int64Attribute>().data());
    } else if (attr_type_name == "pir::FloatAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<pir::FloatAttribute>().data());
    } else if (attr_type_name == "pir::BoolAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<pir::BoolAttribute>().data());
    } else if (attr_type_name == "pir::StrAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::StrAttribute>().AsString());
    } else if (attr_type_name ==
               "pir::ArrayAttribute<paddle::dialect::ScalarAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<phi::Scalar> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<paddle::dialect::ScalarAttribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be dialect::ScalarAttribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(array_list[i]
                                .dyn_cast<paddle::dialect::ScalarAttribute>()
                                .data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int32Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<int32_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int32Attribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be pir::Int32Attribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int32Attribute>().data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::FloatAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<float> vec_res;
      if (array_list.size() > 0) {
        if (array_list[0].isa<pir::FloatAttribute>()) {
          for (size_t i = 0; i < array_list.size(); ++i) {
            vec_res.push_back(
                array_list[i].dyn_cast<pir::FloatAttribute>().data());
          }

        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  attr_type_name));
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int64Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int64Attribute>(),
            true,
            phi::errors::PreconditionNotMet(
                "Element in array list MUST be pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int64Attribute>().data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int64Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int64Attribute>(),
            true,
            phi::errors::PreconditionNotMet(
                "Element in array list MUST be pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int64Attribute>().data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "paddle::dialect::PlaceAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::ScalarAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                              attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // EmplaceBackOutputs
  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value out_ptr = op->result(i);
    if (!IsInvalid(out_ptr)) {
      if (op_yaml_info.GetOutputType(i) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        OutListType optional_outputs;
        ctx->EmplaceBackOutputs(optional_outputs);
      } else {
        phi::DenseTensor* temp = nullptr;
        OutType optional_input(temp);
        ctx->EmplaceBackOutput(optional_input);
      }
      VLOG(8) << "ctx->EmplaceBackOutput : an optioanl output";
      continue;
    }

    if (out_ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      ctx->EmplaceBackOutput(OutType(const_cast<phi::DenseTensor*>(
          &(inner_scope->FindVar(name_map.at(out_ptr))
                ->Get<phi::DenseTensor>()))));
    } else if (out_ptr.type()
                   .isa<paddle::dialect::AllocatedSelectedRowsType>()) {
      ctx->EmplaceBackOutput(OutType(const_cast<phi::SelectedRows*>(
          &(inner_scope->FindVar(name_map.at(out_ptr))
                ->Get<phi::SelectedRows>()))));
    } else if (out_ptr.type().isa<pir::VectorType>()) {
      OutListType outputs;
      auto& variable_array = inner_scope->FindVar(name_map.at(out_ptr))
                                 ->Get<paddle::framework::VariableRefArray>();
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          outputs.emplace_back(OutType(const_cast<phi::DenseTensor*>(
              &(variable_array[i]->Get<phi::DenseTensor>()))));
        } else if (variable_array[i]->IsType<phi::SelectedRows>()) {
          outputs.emplace_back(OutType(const_cast<phi::SelectedRows*>(
              &(variable_array[i]->Get<phi::SelectedRows>()))));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Only support Vector<DenseTensor> and vector<SelectedRows> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      ctx->EmplaceBackOutputs(outputs);
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("only support DenseTensor and vector "));
    }
  }

  VLOG(6) << "Done build phi context";
}

}  // namespace pir
