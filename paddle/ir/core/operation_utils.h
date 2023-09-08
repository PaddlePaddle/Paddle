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
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/region.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"

namespace ir {
class Block;
using AttributeMap = std::unordered_map<std::string, Attribute>;

//===----------------------------------------------------------------------===//
// OperationArgument
//===----------------------------------------------------------------------===//

// This represents an operation arguments in an combined form, suitable for use
// with the builder APIs.
struct OperationArgument {
  std::vector<OpResult> inputs;
  AttributeMap attributes;
  std::vector<Type> output_types;
  OpInfo info;
  size_t num_regions{0};
  std::vector<Block*> successors;

 public:
  OperationArgument(IrContext* ir_context, const std::string& name);
  explicit OperationArgument(OpInfo info) : info(info) {}
  OperationArgument(const std::vector<OpResult>& operands,
                    const AttributeMap& attributes,
                    const std::vector<Type>& types,
                    OpInfo info,
                    size_t num_regions = 0,
                    const std::vector<Block*> successors = {})
      : inputs(operands),
        attributes(attributes),
        output_types(types),
        info(info),
        num_regions(num_regions),
        successors(successors) {}

  /// Add Operand.
  void AddOperand(OpResult operand) { inputs.emplace_back(operand); }

  template <class InputIt>
  void AddOperands(InputIt first, InputIt last);

  /// Add Output.
  void AddOutput(Type type) { output_types.emplace_back(type); }

  template <class InputIt>
  void AddOutputs(InputIt first, InputIt last);

  /// Add an attribute with the specified name.
  void AddAttribute(const std::string& name, Attribute attr) {
    attributes[name] = attr;
  }
  /// Add an array of named attributes.
  template <class InputIt>
  void AddAttributes(InputIt first, InputIt last);
  /// Get the context held by this operation state.
  IrContext* getContext() const { return info.ir_context(); }

  void AddSuccessor(Block* successor) { successors.emplace_back(successor); }
};

template <class InputIt>
void OperationArgument::AddOperands(InputIt first, InputIt last) {
  while (first != last) {
    inputs.emplace_back(*first++);
  }
}
template <class InputIt>
void OperationArgument::AddOutputs(InputIt first, InputIt last) {
  while (first != last) {
    output_types.emplace_back(*first++);
  }
}
template <class InputIt>
void OperationArgument::AddAttributes(InputIt first, InputIt last) {
  while (first != last) {
    attributes[first->first] = first->second;
    ++first;
  }
}

}  // namespace ir
