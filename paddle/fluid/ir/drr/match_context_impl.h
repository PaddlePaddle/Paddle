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

#include <memory>
#include <unordered_map>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/fluid/ir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/ir/drr/api/tensor_interface.h"
#include "paddle/fluid/ir/drr/ir_operation.h"
#include "paddle/fluid/ir/drr/ir_value.h"
#include "paddle/ir/core/builtin_attribute.h"

namespace ir {
namespace drr {

template <class T>
struct CppTypeToIrAttribute;

#define PD_SPECIALIZE_CppTypeToIrAttribute(cpp_type, ir_attr_type) \
  template <>                                                      \
  struct CppTypeToIrAttribute<cpp_type> {                          \
    using type = ir_attr_type;                                     \
  };

PD_SPECIALIZE_CppTypeToIrAttribute(bool, BoolAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int32_t, Int32Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int64_t, Int64Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(float, FloatAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::DataType,
                                   paddle::dialect::DataTypeAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::Place, paddle::dialect::PlaceAttribute);

template <typename T>
struct IrAttrTypeCast {
  static T To(const ir::Attribute& attr) {
    return attr.dyn_cast<typename CppTypeToIrAttribute<T>::type>().data();
  }
};

template <>
struct IrAttrTypeCast<std::vector<int32_t>> {
  static std::vector<int32_t> To(const ir::Attribute& attr) {
    std::vector<int32_t> result;
    auto array_attr = attr.dyn_cast<ir::ArrayAttribute>();
    for (size_t i = 0; i < array_attr.size(); i++) {
      result.push_back(array_attr.at(i).dyn_cast<ir::Int32Attribute>().data());
    }
    return result;
  }
};

template <>
struct IrAttrTypeCast<std::vector<int64_t>> {
  static std::vector<int64_t> To(const ir::Attribute& attr) {
    std::vector<int64_t> result;
    if (attr.dyn_cast<ir::ArrayAttribute>()) {
      auto array_attr = attr.dyn_cast<ir::ArrayAttribute>();
      for (size_t i = 0; i < array_attr.size(); i++) {
        result.push_back(
            array_attr.at(i).dyn_cast<ir::Int64Attribute>().data());
      }
    } else if (attr.dyn_cast<paddle::dialect::IntArrayAttribute>()) {
      result =
          attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "Dynamic cast failed for IR attribute vector<int64_t>"));
    }
    return result;
  }
};

class MatchContextImpl final {
 public:
  MatchContextImpl() = default;
  ~MatchContextImpl() = default;

  const TensorInterface& Tensor(const std::string& tensor_name) const {
    return *tensor_map_.at(tensor_name);
  }

  const IrOperation& Operation(const OpCall* op_call) const {
    return *operation_map_.at(op_call);
  }

  template <typename T>
  T Attr(const std::string& attr_name) const {
    return IrAttrTypeCast<T>::To(GetIrAttr(attr_name));
  }

  const IrValue& GetIrValue(const std::string& tensor_name) const {
    return *tensor_map_.at(tensor_name);
  }

  ir::Attribute GetIrAttr(const std::string& attr_name) const {
    return attr_map_.at(attr_name);
  }

  const std::unordered_map<const OpCall*, std::shared_ptr<IrOperation>>&
  operation_map() const {
    return operation_map_;
  }

  const std::unordered_map<std::string, ir::Attribute>& attr_map() const {
    return attr_map_;
  }

  void BindIrValue(const std::string& value_name,
                   const std::shared_ptr<IrValue>& value) {
    tensor_map_.emplace(value_name, value);
  }

  void BindIrOperation(const OpCall* op_call,
                       const std::shared_ptr<IrOperation>& op) {
    operation_map_.emplace(op_call, op);
    const auto& attrs = op_call->attributes();
    for (const auto& kv : attrs) {
      std::visit(
          [&](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                         NormalAttribute>) {
              BindIrAttr(arg.name(), op->get()->attribute(kv.first));
            }
          },
          kv.second);
    }
  }

 private:
  void BindIrAttr(const std::string& attr_name, ir::Attribute attr) {
    attr_map_.emplace(attr_name, attr);
  }

  std::unordered_map<std::string, std::shared_ptr<IrValue>> tensor_map_;
  std::unordered_map<const OpCall*, std::shared_ptr<IrOperation>>
      operation_map_;
  std::unordered_map<std::string, ir::Attribute> attr_map_;
};

}  // namespace drr
}  // namespace ir
