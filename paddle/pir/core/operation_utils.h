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

#include <initializer_list>
#include <memory>
#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/op_info.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"

namespace pir {
class Block;
using AttributeMap = std::unordered_map<std::string, Attribute>;

//===----------------------------------------------------------------------===//
// OperationArgument
//===----------------------------------------------------------------------===//

// This represents an operation arguments in an combined form, suitable for use
// with the builder APIs.
struct OperationArgument {
  std::vector<Value> inputs;
  AttributeMap attributes;
  std::vector<Type> output_types;
  OpInfo info;
  size_t num_regions{0};
  std::vector<Block*> successors;

 public:
  OperationArgument(IrContext* ir_context, const std::string& name);
  explicit OperationArgument(OpInfo info) : info(info) {}
  OperationArgument(const std::vector<Value>& inputs,
                    const AttributeMap& attributes,
                    const std::vector<Type>& types,
                    OpInfo info,
                    size_t num_regions = 0,
                    const std::vector<Block*> successors = {})
      : inputs(inputs),
        attributes(attributes),
        output_types(types),
        info(info),
        num_regions(num_regions),
        successors(successors) {}

  void AddInput(Value input) { inputs.emplace_back(input); }

  template <class InputIt>
  void AddInputs(InputIt first, InputIt last);

  void AddInputs(std::initializer_list<Value> value_list) {
    AddInputs(std::begin(value_list), std::end(value_list));
  }

  template <class ValueContainer>
  void AddInputs(const ValueContainer& value_container) {
    AddInputs(std::begin(value_container), std::end(value_container));
  }

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

  template <class AttrContainer>
  void AddAttributes(const AttrContainer& attr_container) {
    AddAttributes(std::begin(attr_container), std::end(attr_container));
  }

  /// Get the context held by this operation state.
  IrContext* getContext() const { return info.ir_context(); }

  void AddSuccessor(Block* successor) { successors.emplace_back(successor); }
};

template <class InputIt>
void OperationArgument::AddInputs(InputIt first, InputIt last) {
  while (first != last) {
    AddInput(*first++);
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

}  // namespace pir
