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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

namespace cinn {
namespace dialect {
namespace ir {

class OpNode;

class TensorNode final {
 public:
  TensorNode(::pir::Value value) : node_data_(value), consumers_(value) {}

  // Get the shape of tensor.
  const phi::DDim& shape() const {
    return node_data_.type()
        .dyn_cast<paddle::dialect::DenseTensorType>()
        .dims();
  }

  // Input data has no producer.
  bool HasProducer() const { return consumers_.size() != 0; }

  OpNode producer() const;

  class ConsumerOpListView {
   public:
    explicit ConsumerOpListView(pir::Value data) : node_data_(data) {}

    ConsumerOpListView(const ConsumerOpListView& other) = delete;
    ConsumerOpListView(ConsumerOpListView&& other) = delete;

    ConsumerOpListView& operator=(const ConsumerOpListView& other) = delete;

    using UseIterator = ::pir::ValueUseIterator<::pir::OpOperand>;
    class Iterator {
     public:
      explicit Iterator(UseIterator it) : iter_(it) {}

      Iterator& operator++() {
        ++iter_;
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        ++iter_;
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return iter_ == other.iter_;
      }

      bool operator!=(const Iterator& other) const { return !(*this == other); }

      OpNode operator*() const;

     private:
      UseIterator iter_;
    };

    size_t size() const { return node_data_.use_count(); }

    Iterator begin() const { return Iterator(node_data_.use_begin()); }

    Iterator end() const { return Iterator(node_data_.use_end()); }

   private:
    ::pir::Value node_data_;
  };

  const ConsumerOpListView& consumers() const { return consumers_; }

 private:
  ::pir::Value node_data_;

  const ConsumerOpListView consumers_;
};

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
