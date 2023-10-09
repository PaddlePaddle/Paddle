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
#include <string>
#include <unordered_map>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/utils/type_defs.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"

namespace cinn {
namespace utils {

using NewIR_AttributeMap = std::unordered_map<std::string, ::pir::Attribute>;

Attribute ConvertAttribute(const ::pir::Attribute& src_attr) {
  Attribute dst_attr;
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

AttributeMap ConvertAttributes(const NewIR_AttributeMap& src_attrs) {
  AttributeMap dst_attrs;
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
  VLOG(4) << "dst_attrs.size(): " << dst_attrs.size();
  return dst_attrs;
}

#define CASE_TYPE(src, dst) \
  else if (type.isa<::pir::src>()) return common::dst();

common::Type ConvertIRType(::pir::Type type) {
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

}  // namespace utils
}  // namespace cinn
