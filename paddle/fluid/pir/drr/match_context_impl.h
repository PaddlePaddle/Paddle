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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/api/tensor_interface.h"
#include "paddle/fluid/pir/drr/attr_type_uilts.h"
#include "paddle/fluid/pir/drr/ir_operation.h"
#include "paddle/fluid/pir/drr/ir_value.h"
#include "paddle/pir/core/builtin_attribute.h"

namespace pir {
namespace drr {

class MatchContextImpl final {
 public:
  MatchContextImpl() = default;
  ~MatchContextImpl() = default;

  const TensorInterface& Tensor(const std::string& tensor_name) const {
    IR_ENFORCE(tensor_map_.count(tensor_name),
               "Drr tensor [%s] must exists in pattern graph.",
               tensor_name);
    return *tensor_map_.at(tensor_name);
  }

  const IrOperation& Operation(const OpCall* op_call) const {
    IR_ENFORCE(operation_map_.count(op_call),
               "Drr operation [%s] must exists in pattern graph.",
               op_call->name());
    return *operation_map_.at(op_call);
  }

  template <typename T>
  T Attr(const std::string& attr_name) const {
    return IrAttrTypeCast<T>::To(GetIrAttr(attr_name));
  }

  const IrValue& GetIrValue(const std::string& tensor_name) const {
    auto iter = tensor_map_.find(tensor_name);
    PADDLE_ENFORCE_NE(
        iter,
        tensor_map_.end(),
        phi::errors::OutOfRange(
            "the drr tensor(%s) is not found in the map to ir value.",
            tensor_name));
    return *iter->second;
  }

  pir::Attribute GetIrAttr(const std::string& attr_name) const {
    auto iter = attr_map_.find(attr_name);
    PADDLE_ENFORCE_NE(
        iter,
        attr_map_.end(),
        phi::errors::OutOfRange(
            "the drr attr(%s) is not found in the map to ir attribute.",
            attr_name));
    return iter->second;
  }

  const std::unordered_map<const OpCall*, std::shared_ptr<IrOperation>>&
  operation_map() const {
    return operation_map_;
  }

  const std::unordered_map<std::string, pir::Attribute>& attr_map() const {
    return attr_map_;
  }

  const std::unordered_map<std::string, std::shared_ptr<IrValue>>& tensor_map()
      const {
    return tensor_map_;
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
  void BindIrAttr(const std::string& attr_name, pir::Attribute attr) {
    attr_map_.emplace(attr_name, attr);
  }

  std::unordered_map<std::string, std::shared_ptr<IrValue>> tensor_map_;
  std::unordered_map<const OpCall*, std::shared_ptr<IrOperation>>
      operation_map_;
  std::unordered_map<std::string, pir::Attribute> attr_map_;
};

}  // namespace drr
}  // namespace pir
