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

#include "paddle/cinn/api/op_node.h"

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

class OpGroup {
 public:
  explicit OpGroup(const std::shared_ptr<hlir::framework::Graph::Group>& group)
      : group_(group) {}

  OpGroup(const OpGroup& other) = default;

  using Comparator = hlir::framework::Graph::Group::SharedGroupComparator;
  using Hasher = hlir::framework::Graph::Group::SharedGroupHasher;

  class OpGroupListIterator {
   public:
    OpGroupListIterator(
        std::unordered_set<std::shared_ptr<hlir::framework::Graph::Group>,
                           Hasher,
                           Comparator>::const_iterator it)
        : iter_(it) {}

    OpGroupListIterator& operator++() {
      ++iter_;
      return *this;
    }

    OpGroupListIterator operator++(int) {
      OpGroupListIterator tmp = *this;
      ++iter_;
      return tmp;
    }

    bool operator==(const OpGroupListIterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const OpGroupListIterator& other) const {
      return !(*this == other);
    }

    OpGroup operator*() const { return OpGroup(*iter_); }

   private:
    std::unordered_set<std::shared_ptr<hlir::framework::Graph::Group>,
                       Hasher,
                       Comparator>::const_iterator iter_;
  };

  class ProducerOpGroupListView {
   public:
    ProducerOpGroupListView(
        const std::weak_ptr<hlir::framework::Graph::Group>& group)
        : group_(group) {}

    ProducerOpGroupListView(const ProducerOpGroupListView& other) = delete;
    ProducerOpGroupListView(ProducerOpGroupListView&& other) = delete;

    ProducerOpGroupListView& operator=(const ProducerOpGroupListView& other) =
        delete;

    using const_iterator = OpGroupListIterator;

    size_t size() const {
      CHECK(group_.lock());
      return group_.lock()->producer_groups().size();
    }

    const_iterator begin() const {
      CHECK(group_.lock());
      return const_iterator(group_.lock()->producer_groups().begin());
    }

    const_iterator end() const {
      CHECK(group_.lock());
      return const_iterator(group_.lock()->producer_groups().end());
    }

   private:
    const std::weak_ptr<hlir::framework::Graph::Group> group_;
  };

  class ConsumerOpGroupListView {
   public:
    ConsumerOpGroupListView(
        const std::weak_ptr<hlir::framework::Graph::Group>& group)
        : group_(group) {}

    ConsumerOpGroupListView(const ConsumerOpGroupListView& other) = delete;
    ConsumerOpGroupListView(ConsumerOpGroupListView&& other) = delete;

    ConsumerOpGroupListView& operator=(const ConsumerOpGroupListView& other) =
        delete;

    using const_iterator = OpGroupListIterator;

    size_t size() const {
      CHECK(group_.lock());
      return group_.lock()->consumer_groups().size();
    }

    const_iterator begin() const {
      CHECK(group_.lock());
      return const_iterator(group_.lock()->consumer_groups().begin());
    }

    const_iterator end() const {
      CHECK(group_.lock());
      return const_iterator(group_.lock()->consumer_groups().end());
    }

   private:
    const std::weak_ptr<hlir::framework::Graph::Group> group_;
  };

  const std::string& group_id() const { return group_.lock()->group_id; }

  hlir::framework::OpPatternKind kind() const { return group_.lock()->kind(); }

  // The WalkOpNodes function is used to traverse the op_nodes in the group and
  // execute the VisitOpNode function for each OpNode. This function is
  // equivalent to for loop for op_nodes in graph.
  //
  // In order to avoid unnecessary memory copies, we use WalkOpNodes function
  // instead of providing a function to get all op_nodes directly.
  //
  // Example: Get the all Reduction op_nodes in the group.
  //   OpGroup group = ...;
  //   std::set<api::OpNode> reduce_ op_set;
  //   // The lambda funtion of VisitOpNode to get reduction op_nodes.
  //   auto get_reduce_op = [&reduce_op_set](const api::OpNode& op){
  //     if (op.kind() == OpPatternKind::kReduction) {
  //       reduce_op_set.insert(op);
  //     }
  //   };
  //   group.WalkOpNodes(get_reduce_op);
  void WalkOpNodes(
      const std::function<void(const OpNode&)>& VisitOpNode) const {
    group_.lock()->WalkNodes([&](const hlir::framework::Node* node) {
      VisitOpNode(OpNode(node, group_.lock()->graph_));
    });
  }

  ProducerOpGroupListView producers() const {
    return ProducerOpGroupListView(group_);
  }

  ConsumerOpGroupListView consumers() const {
    return ConsumerOpGroupListView(group_);
  }

  std::shared_ptr<hlir::framework::Graph::Group> GetGroup() const {
    return group_.lock();
  }

  bool operator==(const OpGroup& other) const {
    return group_.lock().get() == other.group_.lock().get();
  }

  bool operator<(const OpGroup& other) const {
    return group_.lock().get() < other.group_.lock().get();
  }

 private:
  const std::weak_ptr<hlir::framework::Graph::Group> group_;
};

}  // namespace api
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::api::OpGroup> {
  size_t operator()(const cinn::api::OpGroup& obj) const {
    return std::hash<size_t>()(reinterpret_cast<size_t>(obj.GetGroup().get()));
  }
};

}  // namespace std
