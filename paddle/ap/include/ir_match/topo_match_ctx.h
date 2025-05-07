// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <unordered_map>
#include "paddle/ap/include/graph/adt.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/node_descriptor.h"

namespace ap::ir_match {

template <typename bg_node_t /*big graph node type*/,
          typename sg_node_t /*small graph node type*/>
struct TopoMatchCtxImpl {
  TopoMatchCtxImpl() {}
  TopoMatchCtxImpl(const TopoMatchCtxImpl&) = default;
  TopoMatchCtxImpl(TopoMatchCtxImpl&&) = default;

  bool operator==(const TopoMatchCtxImpl& other) const {
    return this == &other;
  }

  std::size_t num_matched_bg_nodes() const {
    return matched_bg_node2sg_node_.size();
  }

  bool HasBigGraphNode(const sg_node_t& node) const {
    return this->sg_node2bg_nodes_.count(node) > 0;
  }

  adt::Result<bg_node_t> GetSoleBigGraphNode(const sg_node_t& node) const {
    ADT_LET_CONST_REF(bg_nodes, GetBigGraphNodes(node));
    ADT_CHECK(bg_nodes->size() == 1);
    return *bg_nodes->begin();
  }

  std::optional<sg_node_t> GetMatchedSmallGraphNode(
      const bg_node_t& bg_node) const {
    const auto& iter = matched_bg_node2sg_node_.find(bg_node);
    if (iter == matched_bg_node2sg_node_.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  bool HasMatchedSmallGraphNode(const bg_node_t& bg_node) const {
    const auto& iter = matched_bg_node2sg_node_.find(bg_node);
    return iter != matched_bg_node2sg_node_.end();
  }

  adt::Result<const std::list<bg_node_t>*> GetBigGraphNodes(
      const sg_node_t& node) const {
    const auto& iter = this->sg_node2bg_nodes_.find(node);
    if (iter == this->sg_node2bg_nodes_.end()) {
      return adt::errors::KeyError{
          std::string() + "no node_id " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(node) + " found."};
    }
    return &iter->second;
  }

  adt::Result<std::list<bg_node_t>*> MutBigGraphNodes(
      const sg_node_t& node) const {
    const auto& iter = this->sg_node2bg_nodes_.find(node);
    if (iter == this->sg_node2bg_nodes_.end()) {
      return adt::errors::KeyError{
          std::string() + "no node_id " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(node) + " found."};
    }
    return const_cast<std::list<bg_node_t>*>(&iter->second);
  }

  adt::Result<adt::Ok> InitBigGraphNodes(const sg_node_t& sg_node,
                                         const std::list<bg_node_t>& val) {
    VLOG(4) << "InitBigGraphNodes. sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : val) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    auto* ptr = &this->sg_node2bg_nodes_[sg_node];
    ADT_CHECK(ptr->empty()) << adt::errors::KeyError{
        "InitBigGraphNodes failed. 'sg_node' has been matched to existed "
        "bg_nodes"};
    ADT_CHECK(!val.empty()) << adt::errors::MismatchError{
        "TopoMatchCtxImpl::InitBigGraphNodes: sg_node should not be matched to "
        "empty."};
    for (const auto& bg_node : val) {
      ADT_CHECK(!HasMatchedSmallGraphNode(bg_node)) << adt::errors::KeyError{
          "TopoMatchCtxImpl::InitBigGraphNodes failed. there is matched "
          "bg_node in 'val'"};
    }
    *ptr = val;
    if (ptr->size() == 1) {
      ADT_CHECK(matched_bg_node2sg_node_.emplace(*val.begin(), sg_node).second);
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> UpdateBigGraphNodes(
      const sg_node_t& sg_node, const std::unordered_set<bg_node_t>& val) {
    ADT_CHECK(!val.empty());
    for (const auto& bg_node : val) {
      const auto& opt_matched = GetMatchedSmallGraphNode(bg_node);
      ADT_CHECK(!opt_matched.has_value() || opt_matched.value() == sg_node)
          << adt::errors::KeyError{
                 "UpdateBigGraphNodes failed. there is matched bg_node in "
                 "'val'"};
    }
    auto* ptr = &this->sg_node2bg_nodes_[sg_node];
    VLOG(4) << "UpdateBigGraphNodes: sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", old_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : *ptr) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    VLOG(4) << "UpdateBigGraphNodes: sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", arg_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : val) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    for (auto lhs_iter = ptr->begin(); lhs_iter != ptr->end();) {
      if (val.count(*lhs_iter) > 0) {
        ++lhs_iter;
      } else {
        lhs_iter = ptr->erase(lhs_iter);
      }
    }
    VLOG(4) << "UpdateBigGraphNodes: sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", new_val: " <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : *ptr) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    if (ptr->size() == 1) {
      const auto& iter =
          matched_bg_node2sg_node_.emplace(*ptr->begin(), sg_node).first;
      ADT_CHECK(iter->second == sg_node);
    }
    return adt::Ok{};
  }

  adt::Result<std::shared_ptr<TopoMatchCtxImpl>> CloneAndSetUnsolved(
      const sg_node_t& sg_node, const bg_node_t& bg_node) const {
    auto ret = std::make_shared<TopoMatchCtxImpl>(*this);
    ret->matched_bg_node2sg_node_[bg_node] = sg_node;
    const auto& iter = ret->sg_node2bg_nodes_.find(sg_node);
    ADT_CHECK(iter != ret->sg_node2bg_nodes_.end());
    ADT_CHECK(iter->second.size() > 1);
    ret->sg_node2bg_nodes_[sg_node] = std::list<bg_node_t>{bg_node};
    return ret;
  }

  using SgNode2BgNodes = std::unordered_map<sg_node_t, std::list<bg_node_t>>;

  std::optional<typename SgNode2BgNodes::const_iterator> GetFirstUnsolved()
      const {
    for (auto iter = sg_node2bg_nodes_.begin(); iter != sg_node2bg_nodes_.end();
         ++iter) {
      if (iter->second.size() > 1) {
        return iter;
      }
    }
    return std::nullopt;
  }

  std::optional<typename SgNode2BgNodes::const_iterator> GetFirstMismatched()
      const {
    for (auto iter = sg_node2bg_nodes_.begin(); iter != sg_node2bg_nodes_.end();
         ++iter) {
      if (iter->second.empty()) {
        return iter;
      }
    }
    return std::nullopt;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitSmallGraphNode(const YieldT& Yield) const {
    for (const auto& [sg_node, _] : sg_node2bg_nodes_) {
      ADT_RETURN_IF_ERR(Yield(sg_node));
    }
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> LoopMutBigGraphNode(const YieldT& Yield) {
    for (auto& [_, bg_nodes] : sg_node2bg_nodes_) {
      ADT_LET_CONST_REF(ctrl, Yield(&bg_nodes));
      if (ctrl.template Has<adt::Break>()) {
        break;
      }
    }
    return adt::Ok{};
  }

 private:
  SgNode2BgNodes sg_node2bg_nodes_;
  std::unordered_map<bg_node_t, sg_node_t> matched_bg_node2sg_node_;
};

template <typename bg_node_t /*big graph node type*/,
          typename sg_node_t /*small graph node type*/>
ADT_DEFINE_RC(TopoMatchCtx, TopoMatchCtxImpl<bg_node_t, sg_node_t>);

}  // namespace ap::ir_match
