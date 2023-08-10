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

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/pattern_rewrite/drr/api/tensor_interface.h"
#include "paddle/ir/pattern_rewrite/drr/ir_tensor.h"
#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"

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

class MatchContextImpl final {
 public:
  MatchContextImpl() = default;

  const TensorInterface& Tensor(const std::string& tensor_name) const {
    return *tensor_map_.at(tensor_name);
  }

  template <typename T>
  T Attr(const std::string& attr_name) const {
    return attr_map_.at(attr_name)
        .dyn_cast<typename CppTypeToIrAttribute<T>::type>()
        .data();
  }

  const IrTensor& GetIrTensor(const std::string& tensor_name) const {
    return *tensor_map_.at(tensor_name);
  }

  const std::unordered_map<const OpCall*, ir::Operation*>& op_map() const {
    return op_map_;
  }

  void BindIrTensor(const std::string& tensor_name,
                    const std::shared_ptr<IrTensor>& tensor) {
    tensor_map_.emplace(tensor_name, tensor);
  }

  void BindIrOperation(const OpCall* op_call, const ir::Operation* ir_op) {
    op_map_.emplace(op_call, ir_op);
  }

  void BindIrAttr(const std::string& attr_name, ir::Attribute attr) {
    attr_map_.emplace(attr_name, attr);
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<IrTensor>> tensor_map_;
  std::unordered_map<const OpCall*, ir::Operation*> op_map_;
  std::unordered_map<std::string, ir::Attribute> attr_map_;
};

}  // namespace drr
}  // namespace ir
