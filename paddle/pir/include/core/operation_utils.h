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

#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
class Block;
using AttributeMap = std::unordered_map<std::string, Attribute>;
using PropertyMap = std::unordered_map<std::string, Property>;

//===----------------------------------------------------------------------===//
// OperationArgument
//===----------------------------------------------------------------------===//

// This represents an operation arguments in an combined form, suitable for use
// with the builder APIs.
struct IR_API OperationArgument {
  std::vector<Value> inputs;
  AttributeMap attributes;
  std::vector<Type> output_types;
  OpInfo info;
  std::vector<Block*> successors;
  std::vector<std::unique_ptr<Region>> regions;

 public:
  OperationArgument(IrContext* ir_context, const std::string& name);
  explicit OperationArgument(OpInfo info) : info(info) {}
  OperationArgument(const std::vector<Value>& inputs,
                    const AttributeMap& attributes,
                    const std::vector<Type>& types,
                    OpInfo info,
                    const std::vector<Block*> successors = {})
      : inputs(inputs),
        attributes(attributes),
        output_types(types),
        info(info),
        successors(successors) {}

  OperationArgument& operator=(const OperationArgument&) = delete;
  OperationArgument(const OperationArgument&) = delete;

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

  void AddOutputs(std::initializer_list<Type> type_list) {
    AddOutputs(std::begin(type_list), std::end(type_list));
  }
  template <class TypeContainer>
  void AddOutputs(const TypeContainer& type_container) {
    AddOutputs(std::begin(type_container), std::end(type_container));
  }

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

  /// Create a region that should be attached to the operation.  These regions
  /// can be filled in immediately without waiting for Operation to be
  /// created.  When it is, the region bodies will be transferred.
  Region& AddRegion();

  /// Take a region that should be attached to the Operation.  The body of the
  /// region will be transferred when the Operation is created.  If the
  /// region is nullptr, a new empty region will be attached to the Operation.
  void AddRegion(std::unique_ptr<Region>&& region);

  // This interface is equivalent to calling AddRegion(nullptr) 'size' times.
  void AddRegions(size_t size);
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
