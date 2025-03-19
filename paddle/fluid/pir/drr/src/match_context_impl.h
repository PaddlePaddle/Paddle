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

#include "glog/logging.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/attr_type_uilts.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace drr {

class MatchContextImpl final {
 public:
  MatchContextImpl() = default;
  ~MatchContextImpl() = default;

  const pir::Value& Tensor(const std::string& tensor_name) const {
    PADDLE_ENFORCE_NE(
        tensor_map_.count(tensor_name),
        0,
        common::errors::NotFound("Not found tensor. The drr tensor [%s] must "
                                 "exist in pattern graph to be obtained.",
                                 tensor_name));
    return tensor_map_.at(tensor_name);
  }

  pir::Operation* IrOperation(const OpCall* op_call) const {
    PADDLE_ENFORCE_NE(
        operation_map_.count(op_call),
        0,
        common::errors::NotFound(
            "Not found operation. The drr operation [%s] must exist in the "
            "pattern graph to be obtained.",
            op_call->name()));
    return operation_map_.at(op_call);
  }

  template <typename T>
  T Attr(const std::string& attr_name) const {
    return IrAttrTypeCast<T>::To(GetIrAttr(attr_name));
  }

  pir::Value GetIrValue(const std::string& tensor_name) const {
    auto iter = tensor_map_.find(tensor_name);
    PADDLE_ENFORCE_NE(
        iter,
        tensor_map_.end(),
        common::errors::NotFound(
            "Not found tensor. The drr tensor [%s] is not found in the map, "
            "unable to obtain the corresponding IrValue.",
            tensor_name));
    return iter->second;
  }

  pir::Attribute GetIrAttr(const std::string& attr_name) const {
    auto iter = attr_map_.find(attr_name);
    PADDLE_ENFORCE_NE(
        iter,
        attr_map_.end(),
        common::errors::NotFound(
            "Not found attr. The drr attr [%s] is not found in the map, unable "
            "to obtain the corresponding Attribute.",
            attr_name));
    return iter->second;
  }

  const std::unordered_map<const OpCall*, pir::Operation*>& operation_map()
      const {
    return operation_map_;
  }

  const std::unordered_map<std::string, pir::Attribute>& attr_map() const {
    return attr_map_;
  }

  const std::unordered_map<std::string, pir::Value>& tensor_map() const {
    return tensor_map_;
  }

  void BindIrValue(const std::string& value_name, const pir::Value& value) {
    tensor_map_.emplace(value_name, value);
  }

  bool BindIrOperation(const OpCall* op_call, pir::Operation* op) {
    operation_map_.emplace(op_call, op);
    const auto& attrs = op_call->attributes();
    for (const auto& kv : attrs) {
      bool bind_success = std::visit(
          [&](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                         NormalAttribute>) {
              if (op->HasAttribute(kv.first)) {
                BindIrAttr(arg.name(), op->attribute(kv.first));
                return true;
              }
            }
            return false;
          },
          kv.second);
      if (!bind_success) {
        LOG(WARNING) << "Not found attribute [" << kv.first << "] in Op ["
                     << op->name()
                     << "], please check the "
                        "validity of the attribute name["
                     << kv.first << "].";
        return false;
      }
    }
    return true;
  }

 private:
  void BindIrAttr(const std::string& attr_name, pir::Attribute attr) {
    attr_map_.emplace(attr_name, attr);
  }

  std::unordered_map<std::string, pir::Value> tensor_map_;
  std::unordered_map<const OpCall*, pir::Operation*> operation_map_;
  std::unordered_map<std::string, pir::Attribute> attr_map_;
};

}  // namespace drr
}  // namespace paddle
