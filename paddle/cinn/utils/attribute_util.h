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

#include "paddle/cinn/utils/type_defs.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/phi/common/data_type.h"

namespace cinn {
namespace utils {

using NewIR_AttributeMap = std::unordered_map<std::string, ::ir::Attribute>;

Attribute ConvertAttribute(const ::ir::Attribute& src_attr) {
  Attribute dst_attr;
  if (src_attr.isa<::ir::BoolAttribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::BoolAttribute>().data();
  } else if (src_attr.isa<::ir::FloatAttribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::FloatAttribute>().data();
  } else if (src_attr.isa<::ir::Int32Attribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::Int32Attribute>().data();
  } else if (src_attr.isa<::ir::StrAttribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::StrAttribute>().AsString();
  } else if (src_attr.isa<::ir::Int64Attribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::Int64Attribute>().data();
  } else if (src_attr.isa<::ir::DoubleAttribute>()) {
    dst_attr = src_attr.dyn_cast<::ir::DoubleAttribute>().data();
  } else if (src_attr.isa<paddle::dialect::IntArrayAttribute>()) {
    auto arr = src_attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data();
    std::vector<int> val;
    for (size_t i = 0; i < arr.size(); ++i) {
      val.push_back(arr[i]);
    }
    dst_attr = val;
  } else if (src_attr.isa<paddle::dialect::DataTypeAttribute>()) {
    // TODO(Aurelius84): Need add convert logic from phi::DataType into cinn
    // String.
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
    if (!item.second.isa<paddle::dialect::PlaceAttribute>()) {
      dst_attrs[item.first] = std::move(ConvertAttribute(item.second));
    } else {
      // TODO(Aurelius84): support place attribute for special Op
      dst_attrs["force_cpu"] = false;
    }
  }
  VLOG(4) << "dst_attrs.size(): " << dst_attrs.size();
  return dst_attrs;
}

}  // namespace utils
}  // namespace cinn
