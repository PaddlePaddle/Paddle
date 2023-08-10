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

#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

namespace ir {
namespace cinn {
namespace api {

class OpNode;

class TensorNode final {
 public:
  TensorNode(::ir::Value node_data) : node_data_(node_data) {}

  // Get the shape of tensor.
  const phi::DDim& shape() const {
    return node_data_.dyn_cast<paddle::dialect::DenseTensorType>().dims();
  }

  // Input data has no producer.
  bool HasProducer() const { return node_data_.begin() != node_data_.end(); }

  OpNode producer() const { return OpNode(node_data_.GetDefiningOp()); }

  // class ConsumerOpListView {
  //  public:
  //   ConsumerOpListView(const std::set<common::Shared<common::GraphEdge>,
  //                                     common::GraphEdgeCompare>& edges,
  //                      const hlir::framework::Graph* graph)
  //       : edges_(edges), graph_(graph) {}

  //   ConsumerOpListView(const ConsumerOpListView& other) = delete;
  //   ConsumerOpListView(ConsumerOpListView&& other) = delete;

  //   ConsumerOpListView& operator=(const ConsumerOpListView& other) = delete;

  //   class Iterator {
  //    public:
  //     Iterator(std::set<common::Shared<common::GraphEdge>,
  //                       common::GraphEdgeCompare>::const_iterator it,
  //              const hlir::framework::Graph* graph)
  //         : iter_(it), graph_(graph) {}

  //     Iterator& operator++() {
  //       ++iter_;
  //       return *this;
  //     }

  //     Iterator operator++(int) {
  //       Iterator tmp = *this;
  //       ++iter_;
  //       return tmp;
  //     }

  //     bool operator==(const Iterator& other) const {
  //       return iter_ == other.iter_;
  //     }

  //     bool operator!=(const Iterator& other) const { return !(*this ==
  //     other); }

  //     OpNode operator*() const;

  //    private:
  //     std::set<common::Shared<common::GraphEdge>,
  //              common::GraphEdgeCompare>::const_iterator iter_;
  //     const hlir::framework::Graph* graph_;
  //   };

  //   size_t size() const { return edges_.size(); }

  //   Iterator begin() const { return Iterator(this->edges_.begin(), graph_); }

  //   Iterator end() const { return Iterator(this->edges_.end(), graph_); }

  //  private:
  //   const std::set<Shared<common::GraphEdge>, common::GraphEdgeCompare>&
  //   edges_;
  // };

  // const ConsumerOpListView& consumers() const { return consumers_; }

 private:
  ::ir::Value node_data_;

  // const ConsumerOpListView consumers_;
};

}  // namespace api
}  // namespace cinn
}  // namespace ir
