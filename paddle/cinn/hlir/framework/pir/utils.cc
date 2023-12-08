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
#include "glog/logging.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

// Mapping PaddleDialect Op into CINN AST Compute register Op.
// All key names are also supported in CINN. For ops not in this
// list, we judge them by search it in CINN global Operator table.
const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd_op.full", "fill_constant"},
    {"pd_op.sum", "reduce_sum"},
    {"pd_op.max", "reduce_max"},
    {"pd_op.add", "elementwise_add"},
    {"pd_op.elementwise_pow", "pow"},
    {"pd_op.multiply", "elementwise_mul"},
    {"pd_op.maximum", "max"},
    {"pd_op.minimum", "min"},
    {"pd_op.split_with_num", "split"},
    {"cinn_op.reshape", "reshape"},
    {"cinn_op.scale", "scale"},
    {"cinn_op.broadcast", "broadcast_to"},
    // The following should implement OpPattern in pd_to_cinn_pass,
    // otherwise, it will be block in BuildCinnPass.
    {"cinn_op.squeeze", ""},
    {"cinn_op.unsqueeze", ""}};

// In following cases, the op is marked SupportCinn:
// 1. its name is in OP_NAMES, like pd_op.sum;
// 2. it supports AttributeTensor but has Pattern to process it.
//    Such as cinn_op.reshape, except pd_op.reshape;
// 3. otherwise, it should be registered in OpRegistry;
bool CompatibleInfo::IsSupportCinn(const ::pir::Operation& op) {
  if (OP_NAMES.find(op.name()) != OP_NAMES.end()) {
    return true;
  }
  // After PdToCinnPass, if pd_op.reshape still exists, return false.
  std::string black_op_name =
      std::string(cinn::dialect::OperatorDialect::name()) + "." + OpName(op);
  if (OP_NAMES.find(black_op_name) != OP_NAMES.end()) {
    VLOG(4) << "Found black op after PdToCinnPass, because it has Attribute "
               "Tensor: "
            << op.name();
    return false;
  }
  return OpRegistry::Global()->Find(OpName(op)) != nullptr;
}

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
  CHECK(cinn_op_name != "")
      << "Found empty cinn_op_name, maybe you should implement OpPattern for "
      << name;
  return cinn_op_name;
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

std::string CompatibleInfo::ValueName(const ::pir::Value& value) {
  size_t hash_key = std::hash<::pir::Value>()(value);
  return cinn::common::Context::Global().PrettyUniqName(
      hash_key, CompatibleInfo::kNamePrefix);
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
  } else if (src_attr.isa<::pir::ArrayAttribute>()) {
    auto attr_vec = src_attr.dyn_cast<::pir::ArrayAttribute>().AsVector();
    if (attr_vec.size() > 0) {
      if (attr_vec[0].isa<::pir::Int32Attribute>()) {
        std::vector<int> vec_int32;
        for (auto vec_element : attr_vec) {
          vec_int32.push_back(
              vec_element.dyn_cast<::pir::Int32Attribute>().data());
        }
        dst_attr = vec_int32;

      } else if (attr_vec[0].isa<::pir::Int64Attribute>()) {
        std::vector<int64_t> vec_int64;
        int index = 0;
        for (auto vec_element : attr_vec) {
          vec_int64.push_back(
              vec_element.dyn_cast<::pir::Int64Attribute>().data());
        }

        dst_attr = vec_int64;
      } else {
        LOG(FATAL)
            << "only suuport int32 and int64 attribute in ArrayAttribute";
      }
    }
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
  else if (type.isa<::pir::src>()) return cinn::common::dst();

cinn::common::Type CompatibleInfo::ConvertIRType(::pir::Type type) {
  if (type.isa<::pir::BFloat16Type>()) return cinn::common::BF16();
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

int CompatibleInfo::ShapeProduct(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

OpPatternKind CompatibleInfo::OpKind(const ::pir::Operation& op) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto op_name = CompatibleInfo::OpName(op);
  const hlir::framework::Operator* cinn_op = Operator::Get(op_name);
  CHECK(op_pattern_dict.Find(cinn_op));
  auto kind = op_pattern_dict[cinn_op];
  if (kind == hlir::framework::kBroadcast) {
    // As binary op was defined as broadcast, actually it should be
    // element-wise. See fusion_hepler_base.h for detail.
    if (op_name != "broadcast_to") {
      kind = hlir::framework::kElementWise;
    }
  }
  VLOG(4) << op_name << " OpPatternKind: " << kind;
  return kind;
}

std::vector<int> CompatibleInfo::ValueShape(const ::pir::Value& value) {
  auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize<int>(dim);
}

std::vector<int64_t> GetBroadcastAxis(const phi::DDim& in_shape,
                                      const std::vector<int64_t>& out_shape) {
  std::vector<int64_t> broadcast_axes(in_shape.size(), 0);
  auto in_shape_size = in_shape.size();
  if (in_shape_size >= 1) {
    for (int i = 1; i <= in_shape_size; ++i) {
      broadcast_axes[in_shape_size - i] = out_shape.size() - i;
    }
  }

  return broadcast_axes;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
