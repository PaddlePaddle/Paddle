// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/graph/node_topo_cstr.h"
#include "paddle/ap/include/ir_match/ref_match_ctx.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::paddle {

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetNativeIrValueClass();

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetPackedIrValueClass();

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetRefIrValueClass();

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetNativeIrOpClass();

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetPackedIrOpClass();

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetRefIrOpClass();

struct NativeIrValue {
  pir::Value value;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetNativeIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const { return std::hash<pir::Value>()(value); }

  bool operator==(const NativeIrValue& other) const {
    return this->value == other.value;
  }

  graph::NativeIrValueTopoCstr node_topo_cstr() const {
    return graph::NativeIrValueTopoCstr{};
  }

  adt::Result<axpr::DataType> GetDataType() const {
    ADT_LET_CONST_REF(type, GetPhiDataType());
    return ap::axpr::GetDataTypeFromPhiDataType(type);
  }

  adt::Result<const std::vector<symbol::DimExpr>*> GetShapeDimExprsPtr() const {
    ADT_LET_CONST_REF(shape_or_data, GetShapeOrDataDimExprsPtr());
    return &shape_or_data->shape();
  }

  adt::Result<const symbol::ShapeOrDataDimExprs*> GetShapeOrDataDimExprsPtr()
      const {
    auto* op = value.defining_op();
    ADT_CHECK(op != nullptr);
    auto* program = op->GetParentProgram();
    auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(program);
    const auto& shape_or_data = shape_analysis.GetShapeOrDataForValue(value);
    using RetT = adt::Result<const symbol::ShapeOrDataDimExprs*>;
    return shape_or_data.Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> RetT {
          return &shape_or_data;
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "GetShapeDimExprsPtr only support TensorShapeOrDataDimExprs."};
        });
  }

 private:
  adt::Result<phi::DataType> GetPhiDataType() const {
    ADT_LET_CONST_REF(type, GetPirDataType());
    try {
      return ::paddle::dialect::TransToPhiDataType(type);
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<pir::Type> GetPirDataType() const {
    if (!this->value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        this->value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }
};

struct PackedIrValue {
  cinn::dialect::FusionOp fusion_op;
  bool is_output;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetPackedIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
               static_cast<pir::Operation*>(fusion_op)) ^
           is_output;
  }

  bool operator==(const PackedIrValue& other) const {
    return this->fusion_op == other.fusion_op &&
           this->is_output == other.is_output;
  }

  graph::PackedIrValueTopoCstr node_topo_cstr() const {
    return graph::PackedIrValueTopoCstr{};
  }
};

struct NativeIrOpOperand {
  pir::OpOperand op_operand;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpOperand>()(op_operand);
  }

  bool operator==(const NativeIrOpOperand& other) const {
    return this->op_operand == other.op_operand;
  }

  graph::NativeIrOpOperandTopoCstr node_topo_cstr() const {
    return graph::NativeIrOpOperandTopoCstr{this->op_operand.index()};
  }
};

struct PackedIrOpOperand {
  cinn::dialect::FusionOp fusion_op;
  std::size_t free_tensor_index;

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
               static_cast<pir::Operation*>(fusion_op)) ^
           free_tensor_index;
  }

  bool operator==(const PackedIrOpOperand& other) const {
    return this->fusion_op == other.fusion_op &&
           this->free_tensor_index == other.free_tensor_index;
  }

  graph::PackedIrOpOperandTopoCstr node_topo_cstr() const {
    return graph::PackedIrOpOperandTopoCstr{};
  }
};

struct NativeIrOp {
  pir::Operation* op;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetNativeIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const { return std::hash<pir::Operation*>()(op); }

  bool operator==(const NativeIrOp& other) const {
    return this->op == other.op;
  }

  graph::NativeIrOpTopoCstr node_topo_cstr() const {
    return graph::NativeIrOpTopoCstr{this->op->name()};
  }
};

struct PackedIrOp {
  cinn::dialect::FusionOp fusion_op;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetPackedIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
        static_cast<pir::Operation*>(fusion_op));
  }

  bool operator==(const PackedIrOp& other) const {
    return this->fusion_op == other.fusion_op;
  }

  graph::PackedIrOpTopoCstr node_topo_cstr() const {
    return graph::PackedIrOpTopoCstr{"ap_trivial_fusion_op"};
  }
};

struct NativeIrOpResult {
  pir::OpResult op_result;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpResult>()(op_result);
  }

  bool operator==(const NativeIrOpResult& other) const {
    return this->op_result == other.op_result;
  }

  graph::NativeIrOpResultTopoCstr node_topo_cstr() const {
    return graph::NativeIrOpResultTopoCstr{this->op_result.index()};
  }
};

struct PackedIrOpResult {
  pir::OpResult op_result;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpResult>()(op_result);
  }

  bool operator==(const PackedIrOpResult& other) const {
    return this->op_result == other.op_result;
  }

  graph::PackedIrOpResultTopoCstr node_topo_cstr() const {
    return graph::PackedIrOpResultTopoCstr{this->op_result.index()};
  }
};

}  // namespace ap::paddle

namespace std {

template <>
struct hash<ap::paddle::NativeIrValue> {
  std::size_t operator()(const ap::paddle::NativeIrValue& node) const {
    return node.GetHashValue();
  }
};

template <>
struct hash<ap::paddle::NativeIrOpOperand> {
  std::size_t operator()(const ap::paddle::NativeIrOpOperand& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std

namespace ap::paddle {

using RefNodeInfo = ir_match::RefNodeInfo<NativeIrValue, NativeIrOpOperand>;

struct RefIrValue {
  RefNodeInfo ref_node_info;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetRefIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrValue& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  adt::Result<NativeIrValue> GetOwnerNativeIrValue() const {
    return this->ref_node_info->ir_value;
  }

  graph::RefIrValueTopoCstr node_topo_cstr() const {
    return graph::RefIrValueTopoCstr{};
  }
};

struct RefIrOpOperand {
  RefNodeInfo ref_node_info;

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOpOperand& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpOperandTopoCstr node_topo_cstr() const {
    return graph::RefIrOpOperandTopoCstr{};
  }
};

struct RefIrOp {
  RefNodeInfo ref_node_info;

  template <typename ValueT>
  static axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetBuiltinClass() {
    return GetRefIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOp& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpTopoCstr node_topo_cstr() const {
    return graph::RefIrOpTopoCstr{};
  }
};

struct RefIrOpResult {
  RefNodeInfo ref_node_info;

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOpResult& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpResultTopoCstr node_topo_cstr() const {
    return graph::RefIrOpResultTopoCstr{};
  }
};

using PirNodeImpl = std::variant<NativeIrValue,
                                 PackedIrValue,
                                 NativeIrOpOperand,
                                 PackedIrOpOperand,
                                 RefIrOpOperand,
                                 NativeIrOp,
                                 PackedIrOp,
                                 NativeIrOpResult,
                                 PackedIrOpResult,
                                 RefIrValue,
                                 RefIrOp,
                                 RefIrOpResult>;

struct PirNode : public PirNodeImpl {
  using PirNodeImpl::PirNodeImpl;
  ADT_DEFINE_VARIANT_METHODS(PirNodeImpl);

  using dim_expr_type = ::symbol::DimExpr;
  using native_op_type = NativeIrOp;
  using packed_op_type = PackedIrOp;
  using ref_op_type = RefIrOp;
  using native_value_type = NativeIrValue;
  using packed_value_type = PackedIrValue;
  using ref_value_type = RefIrValue;
  using native_op_operand_type = NativeIrOpOperand;

  std::size_t GetHashValue() const {
    return Match([](const auto& impl) { return impl.GetHashValue(); });
  }

  graph::NodeTopoCstr node_topo_cstr() const {
    return Match([](const auto& impl) -> graph::NodeTopoCstr {
      return impl.node_topo_cstr();
    });
  }

  static adt::Result<std::string> GetOpNameFromDrrPackedOpName(
      const std::string& drr_packed_op_name) {
    if (drr_packed_op_name == "ap_trivial_fusion_op") {
      return "cinn_op.fusion";
    }
    return adt::errors::KeyError{
        std::string() + "no pir op name matched to drr packed op name: '" +
        drr_packed_op_name + "'"};
  }
};

}  // namespace ap::paddle

namespace std {

template <>
struct hash<ap::paddle::PirNode> {
  std::size_t operator()(const ap::paddle::PirNode& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std
