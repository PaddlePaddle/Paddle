// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pir/program_method_class.h"
#include "paddle/ap/include/axpr/dim_expr_method_class.h"
#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"
#include "paddle/ap/include/paddle/pir_node.h"

namespace ap::paddle {

struct PirProgramMethodClass {
  using This = PirProgramMethodClass;
  using Self = Program;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    std::ostringstream ss;
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    pir::IrPrinter(ss).PrintProgram(self->pir_program.get());
    return ss.str();
  }

  static adt::Result<axpr::Value> Empty(const axpr::Value& self_val,
                                        const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    std::ostringstream ss;
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self->pir_program->block()->size() == 0;
  }

  static adt::Result<axpr::Value> CopyToConstProgramData(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    std::ostringstream ss;
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::vector<pir::Value> values{};
    std::unordered_map<pir::Value, int64_t> value2index{};
    std::vector<const pir::Operation*> ops{};
    std::unordered_map<const pir::Operation*, int64_t> op2index{};
    for (const auto& op : *self->pir_program->block()) {
      for (int i = 0; i < op.num_operands(); ++i) {
        if (value2index.emplace(op.operand_source(i), values.size()).second) {
          values.push_back(op.operand_source(i));
        }
      }
      op2index[&op] = ops.size();
      ops.push_back(&op);
      for (int i = 0; i < op.num_results(); ++i) {
        if (value2index.emplace(op.result(i), values.size()).second) {
          values.push_back(op.result(i));
        }
      }
    }
    axpr::AttrMap<axpr::Value> attr_map;
    ADT_LET_CONST_REF(value_data, This{}.ConvertToValues(values, op2index));
    attr_map->Set("values", value_data);
    ADT_LET_CONST_REF(op_data, This{}.ConvertToOps(ops, value2index));
    attr_map->Set("ops", op_data);
    return attr_map;
  }

  static adt::Result<axpr::Value> Clone(const axpr::Value& self_val,
                                        const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    pir::IrMapping ir_mapping;
    auto new_program = self->pir_program->Clone(ir_mapping);
    ADT_RETURN_IF_ERR(This{}.CloneSymbolicShapes(
        new_program.get(), self->pir_program.get(), ir_mapping));
    Program ap_program{new_program};
    return GetPirProgramClass().New(ap_program);
  }

  adt::Result<adt::Ok> CloneSymbolicShapes(pir::Program* new_program,
                                           pir::Program* old_program,
                                           const pir::IrMapping& ir_mapping) {
    auto* new_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(new_program);
    auto* old_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(old_program);
    for (const auto& [old_value, new_value] : ir_mapping.GetMap<pir::Value>()) {
      new_shape_analysis->SetShapeOrDataForValue(
          new_value, old_shape_analysis->GetShapeOrDataForValue(old_value));
    }
    return adt::Ok{};
  }

  adt::Result<adt::List<axpr::Value>> ConvertToOps(
      const std::vector<const pir::Operation*>& ops,
      const std::unordered_map<pir::Value, int64_t>& value2index) {
    adt::List<axpr::Value> ret;
    ret->reserve(ops.size());
    int64_t op_index = 0;
    for (const auto* op : ops) {
      ADT_LET_CONST_REF(op_data, ConvertToOpData(op_index++, op, value2index));
      ret->emplace_back(op_data);
    }
    return ret;
  }

  adt::Result<axpr::Value> ConvertToOpData(
      int64_t op_index,
      const pir::Operation* op,
      const std::unordered_map<pir::Value, int64_t>& value2index) {
    axpr::AttrMap<axpr::Value> attr_map;
    attr_map->Set("op_index", op_index);
    attr_map->Set("op_name", op->name());
    {
      adt::List<axpr::Value> input_indexes;
      input_indexes->reserve(op->num_operands());
      for (int i = 0; i < op->num_operands(); ++i) {
        const auto& index_iter = value2index.find(op->operand_source(i));
        ADT_CHECK(index_iter != value2index.end());
        input_indexes->push_back(index_iter->second);
      }
      attr_map->Set("input_value_indexes", input_indexes);
    }
    {
      adt::List<axpr::Value> output_indexes;
      output_indexes->reserve(op->num_results());
      for (int i = 0; i < op->num_results(); ++i) {
        const auto& index_iter = value2index.find(op->result(i));
        ADT_CHECK(index_iter != value2index.end());
        output_indexes->push_back(index_iter->second);
      }
      attr_map->Set("output_value_indexes", output_indexes);
    }
    {
      axpr::AttrMap<axpr::Value> op_attributes;
      for (const auto& [attr_name, attr_val] : op->attributes()) {
        op_attributes->Set(attr_name, GetPirAttributeClass().New(attr_val));
      }
      attr_map->Set("attributes", op_attributes);
    }
    return attr_map;
  }

  adt::Result<adt::List<axpr::Value>> ConvertToValues(
      const std::vector<pir::Value>& values,
      const std::unordered_map<const pir::Operation*, int64_t>& op2index) {
    adt::List<axpr::Value> ret;
    ret->reserve(values.size());
    int64_t value_index = 0;
    for (pir::Value value : values) {
      ADT_LET_CONST_REF(value_data,
                        ConvertToValueData(value_index++, value, op2index));
      ret->emplace_back(value_data);
    }
    return ret;
  }

  adt::Result<axpr::Value> ConvertToValueData(
      int64_t value_index,
      const pir::Value& value,
      const std::unordered_map<const pir::Operation*, int64_t>& op2index) {
    if (!value) return adt::Nothing{};
    axpr::AttrMap<axpr::Value> attr_map;
    attr_map->Set("value_index", value_index);
    ADT_CHECK(value.defining_op() != nullptr);
    const auto& index_iter = op2index.find(value.defining_op());
    ADT_CHECK(index_iter != op2index.end());
    attr_map->Set("defining_op_index", index_iter->second);
    attr_map->Set("type", GetPirTypeClass().New(value.type()));
    ADT_LET_CONST_REF(symbolic_shape, GetShape(value));
    attr_map->Set("symbolic_shape", symbolic_shape);
    return attr_map;
  }

  adt::Result<axpr::Value> GetShape(pir::Value value) {
    NativeIrValue ir_value{value};
    ADT_LET_CONST_REF(shape_ptr, ir_value.GetShapeDimExprsPtr());
    adt::List<axpr::Value> lst;
    lst->reserve(shape_ptr->size());
    for (const auto& dim_expr : *shape_ptr) {
      axpr::BuiltinClassInstance<axpr::Value> instance{
          axpr::GetDimExprClass<axpr::Value>(), dim_expr};
      lst->emplace_back(instance);
    }
    return lst;
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirProgramClass() {
  using Impl = PirProgramMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("PirProgram", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("empty", &Impl::Empty);
        Yield("copy_to_const_program_data", &Impl::CopyToConstProgramData);
        Yield("clone", &Impl::Clone);
      }));
  return axpr::MakeGlobalNaiveClassOps<Program>(cls);
}

}  // namespace ap::paddle
