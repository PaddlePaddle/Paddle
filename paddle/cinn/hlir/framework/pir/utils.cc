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

#include "paddle/cinn/hlir/framework/pir/utils.h"

#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd_op.full", "fill_constant"},
    {"pd_op.sum", "reduce_sum"},
    {"pd_op.max", "reduce_max"},
    {"pd_op.add", "elementwise_add"}};

std::string CompatibleInfo::OpName(const ::pir::Operation& op) {
  std::string name = op.name();
  if (OP_NAMES.count(name)) {
    return OP_NAMES.at(name);
  }
  auto pos = name.find(".");
  if (pos == std::string::npos) {
    return name;
  }
  auto cinn_op_name = name.substr(pos + 1);
  VLOG(4) << "GetOpName: " << name << " -> " << cinn_op_name;
  return cinn_op_name;
}

std::string CompatibleInfo::ValueName(const ::pir::Value& value) {
  return CompatibleInfo::kNamePrefix +
         std::to_string(std::hash<::pir::Value>()(value));
}

std::string CompatibleInfo::OpFuncName(const ::pir::Operation& op) {
  std::string op_name = OpName(op);
  std::string func_name =
      cinn::common::Context::Global().NewName("fn_" + op_name);
  return func_name;
}

std::string CompatibleInfo::GroupOpsName(
    const std::vector<::pir::Operation*>& ops) {
  std::string name = "fn";
  for (auto* op : ops) {
    std::string op_name = OpName(*op);
    name += "_" + cinn::common::Context::Global().NewName(op_name);
  }
  return name;
}

std::vector<std::string> CompatibleInfo::InputNames(const ::pir::Operation& op,
                                                    bool allow_duplicate) {
  std::vector<std::string> names;
  std::unordered_set<std::string> repeat;
  for (int i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    std::string name = CompatibleInfo::ValueName(value);
    if (!allow_duplicate && repeat.count(name)) {
      continue;
    }
    repeat.insert(name);
    names.push_back(name);
  }
  return names;
}

std::vector<std::string> CompatibleInfo::OutputNames(::pir::Operation& op) {
  std::vector<std::string> names;
  for (int i = 0; i < op.num_results(); ++i) {
    auto value = op.result(i);
    std::string name = CompatibleInfo::ValueName(value);
    names.push_back(std::move(name));
  }
  return names;
}

std::vector<::pir::Value> CompatibleInfo::RealOperandSources(
    const ::pir::Operation& op) {
  if (OpMapper::Instance().has(op, MapperType::OPERAND)) {
    return OpMapper::Instance().RealOprandSources(op);
  } else {
    return op.operands_source();
  }
}

utils::Attribute CompatibleInfo::ConvertAttribute(
    const ::pir::Attribute& src_attr) {
  utils::Attribute dst_attr;
  if (src_attr.isa<::pir::BoolAttribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::BoolAttribute>().data();
  } else if (src_attr.isa<::pir::FloatAttribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::FloatAttribute>().data();
  } else if (src_attr.isa<::pir::Int32Attribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::Int32Attribute>().data();
  } else if (src_attr.isa<::pir::StrAttribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::StrAttribute>().AsString();
  } else if (src_attr.isa<::pir::Int64Attribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::Int64Attribute>().data();
  } else if (src_attr.isa<::pir::DoubleAttribute>()) {
    dst_attr = src_attr.dyn_cast<::pir::DoubleAttribute>().data();
  } else if (src_attr.isa<paddle::dialect::IntArrayAttribute>()) {
    auto& arr = src_attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                    .data()
                    .GetData();
    std::vector<int> val(arr.begin(), arr.end());
    dst_attr = val;
  } else if (src_attr.isa<paddle::dialect::DataTypeAttribute>()) {
    auto dtype = src_attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
    dst_attr = phi::DataTypeToString(dtype);
  } else {
    LOG(FATAL) << "unknown Attribute: " << src_attr;
  }

  return dst_attr;
}

utils::AttributeMap CompatibleInfo::ConvertAttributes(
    const ::pir::Operation& op) {
  auto& src_attrs = op.attributes();
  utils::AttributeMap dst_attrs;
  for (auto& item : src_attrs) {
    VLOG(4) << "deal with " << item.first;
    if (item.first == ::pir::kStopGradientAttrName) {
      continue;
    } else if (item.second.isa<paddle::dialect::PlaceAttribute>()) {
      auto is_cpu =
          item.second.dyn_cast<paddle::dialect::PlaceAttribute>().data() ==
          phi::CPUPlace();
      dst_attrs["force_cpu"] = is_cpu;
    } else {
      dst_attrs[item.first] = std::move(ConvertAttribute(item.second));
    }
  }

  if (OpMapper::Instance().has(op, MapperType::ATTRIBUTE)) {
    OpMapper::Instance().AppendVariantAttrs(op, dst_attrs);
  }
  VLOG(4) << "dst_attrs.size(): " << dst_attrs.size();
  return dst_attrs;
}

#define CASE_TYPE(src, dst) \
  else if (type.isa<::pir::src>()) return common::dst();

common::Type CompatibleInfo::ConvertIRType(::pir::Type type) {
  if (type.isa<::pir::BFloat16Type>()) return common::BF16();
  CASE_TYPE(Float16Type, F16)
  CASE_TYPE(Float32Type, F32)
  CASE_TYPE(Float64Type, F64)
  CASE_TYPE(Int8Type, I8)
  CASE_TYPE(UInt8Type, UI8)
  CASE_TYPE(Int16Type, I16)
  CASE_TYPE(Int32Type, I32)
  CASE_TYPE(Int64Type, I64)
  CASE_TYPE(IndexType, I32)
  CASE_TYPE(BoolType, UI1)

  LOG(FATAL) << "unknown ir::Type " << type;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
