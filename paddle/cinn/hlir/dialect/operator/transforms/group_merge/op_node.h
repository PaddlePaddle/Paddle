// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/tensor_node.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace dialect {
namespace ir {

class OpNode {
 public:
  explicit OpNode(::pir::Operation* node)
      : node_(node), input_tensors_(node), output_tensors_(node) {}

  OpPatternKind kind() const {
    auto kind = hlir::framework::pir::CompatibleInfo::OpKind(*node_);
    if (kind == OpPatternKind::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be
      // element-wise.
      if (node_->name() != "broadcast_to") {
        return OpPatternKind::kElementWise;
      }
    }
    return kind;
  }

  class TensorListIterator {
   public:
    TensorListIterator(size_t index, ::pir::Operation* op)
        : iter_(index), op_(op) {}

    TensorListIterator& operator++() {
      ++iter_;
      return *this;
    }

    TensorListIterator operator++(int) {
      TensorListIterator tmp = *this;
      ++iter_;
      return tmp;
    }

    bool operator==(const TensorListIterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const TensorListIterator& other) const {
      return !(*this == other);
    }

    TensorNode operator*() const {
      return TensorNode(op_->operand_source(iter_));
    }

   private:
    size_t iter_;
    ::pir::Operation* op_;
  };

  using const_iterator = TensorListIterator;

  class InputTensorListView {
   public:
    explicit InputTensorListView(::pir::Operation* op) : op_(op) {}

    // InputTensorListView(const InputTensorListView& other) = delete;
    // InputTensorListView(InputTensorListView&& other) = delete;

    // InputTensorListView& operator=(const InputTensorListView& other) =
    // delete;

    size_t size() const { return op_->num_operands(); }

    TensorNode operator[](size_t index) const {
      return TensorNode(op_->operand_source(index));
    }

    const_iterator begin() const { return const_iterator(0, op_); }

    const_iterator end() const {
      return const_iterator(op_->num_operands(), op_);
    }

   private:
    ::pir::Operation* op_;
  };

  class OutputTensorListView {
   public:
    explicit OutputTensorListView(::pir::Operation* op) : op_(op) {}

    // OutputTensorListView(const OutputTensorListView& other) = delete;
    // OutputTensorListView(OutputTensorListView&& other) = delete;

    // OutputTensorListView& operator=(const OutputTensorListView& other) =
    // delete;

    size_t size() const { return op_->num_results(); }

    TensorNode operator[](size_t index) const {
      return TensorNode(op_->result(index));
    }

    const_iterator begin() const { return const_iterator(0, op_); }

    const_iterator end() const {
      return const_iterator(op_->num_results(), op_);
    }

   private:
    ::pir::Operation* op_;
  };

  bool operator==(const OpNode& other) const { return node_ == other.node_; }

  bool operator<(const OpNode& other) const { return node_ < other.node_; }

  const InputTensorListView& inputs() const { return input_tensors_; }

  const OutputTensorListView& outputs() const { return output_tensors_; }

  template <typename T>
  T GetAttr(const std::string& attr_name) const {
    auto attr =
        paddle::dialect::GetAttributeData(node_->attributes().at(attr_name));
    return paddle::get<T>(attr);
  }

 private:
  friend struct std::hash<OpNode>;

  ::pir::Operation* node_;

  const InputTensorListView input_tensors_;
  const OutputTensorListView output_tensors_;
};

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::dialect::ir::OpNode> {
  size_t operator()(const cinn::dialect::ir::OpNode& obj) const {
    return std::hash<size_t>()(reinterpret_cast<size_t>(obj.node_));
  }
};

}  // namespace std
