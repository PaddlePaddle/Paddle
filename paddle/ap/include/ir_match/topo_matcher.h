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

#include <map>
#include "glog/logging.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/graph/adt.h"
#include "paddle/ap/include/graph/graph_descriptor.h"
#include "paddle/ap/include/graph/graph_helper.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/node_arena.h"
#include "paddle/ap/include/ir_match/tags.h"
#include "paddle/ap/include/ir_match/topo_match_ctx.h"

namespace ap::ir_match {

using graph::GraphDescriptor;
using graph::GraphHelper;

template <typename bg_node_t,
          typename sg_node_t,
          typename BGTopoKind,
          typename SGTopoKind>
struct TopoMatcher {
  TopoMatcher(const GraphDescriptor<bg_node_t, BGTopoKind>& bg_descriptor,
              const GraphDescriptor<sg_node_t, SGTopoKind>& sg_descriptor)
      : bg_descriptor_(bg_descriptor), sg_descriptor_(sg_descriptor) {}

  TopoMatcher(const TopoMatcher&) = delete;
  TopoMatcher(TopoMatcher&&) = delete;

  adt::Result<TopoMatchCtx<bg_node_t, sg_node_t>> MatchByAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(topo_match_ctx,
                      MakeTopoMatchCtxFromAnchor(bg_node, anchor_node));
    ADT_RETURN_IF_ERR(UpdateByConnectionsUntilDone(
        &*topo_match_ctx.shared_ptr(), anchor_node));
    return topo_match_ctx;
  }

  adt::Result<adt::Ok> UpdateByConnectionsUntilDone(
      TopoMatchCtxImpl<bg_node_t, sg_node_t>* ctx,
      const sg_node_t& anchor_node) {
    size_t kDeadloopDetectionSize = 999999;
    while (true) {
      ADT_LET_CONST_REF(updated, UpdateAllByConnections(ctx, anchor_node));
      if (!updated) {
        break;
      }
      if (--kDeadloopDetectionSize <= 0) {
        return adt::errors::RuntimeError{"Dead loop detected."};
      }
    }
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMisMatchedNodes(
      const TopoMatchCtx<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& anchor_node,
      const DoEachT& DoEach) const {
    auto DoEachSGNode = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      if (!ctx->HasBigGraphNode(sg_node)) {
        return DoEach(sg_node);
      }
      return adt::Ok{};
    };
    adt::BfsWalker<sg_node_t> bfs_walker =
        GraphHelper<sg_node_t, SGTopoKind>(sg_descriptor_).GetBfsWalker();
    return bfs_walker(anchor_node, DoEachSGNode);
  }

  adt::Result<bool> IsGraphMatched(
      const TopoMatchCtx<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& anchor_node) const {
    adt::BfsWalker<sg_node_t> bfs_walker =
        GraphHelper<sg_node_t, SGTopoKind>(sg_descriptor_).GetBfsWalker();
    std::size_t num_sg_nodes = 0;
    auto AccNumSgNodes = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      ADT_CHECK(ctx->HasBigGraphNode(sg_node))
          << adt::errors::MismatchError{"IsGraphMatched: sg_node not matched."};
      ADT_LET_CONST_REF(bg_nodes, ctx->GetBigGraphNodes(sg_node));
      ADT_CHECK(bg_nodes->size() == 1) << adt::errors::MismatchError{
          "IsGraphMatched: more than 1 bg_nodes matched to one sg_node."};
      ++num_sg_nodes;
      return adt::Ok{};
    };
    const auto& ret = bfs_walker(anchor_node, AccNumSgNodes);
    if (ret.HasError()) {
      ADT_CHECK(ret.GetError().template Has<adt::errors::MismatchError>())
          << ret.GetError();
      return false;
    }
    return num_sg_nodes == ctx->num_matched_bg_nodes();
  }

  adt::Result<std::size_t> NumUndeterminedNodes(
      const GraphMatchCtx<bg_node_t>& ctx) const {
    std::size_t num_undetermined_nodes = 0;
    using LoopCtrl = adt::Result<adt::LoopCtrl>;
    auto Acc = [&](auto* lst) -> LoopCtrl {
      num_undetermined_nodes += (lst->size() > 1);
      return adt::Continue{};
    };
    ADT_RETURN_IF_ERR(
        ctx->topo_match_ctx.shared_ptr()->LoopMutBigGraphNode(Acc));
    return num_undetermined_nodes;
  }

  adt::Result<std::optional<std::list<bg_node_t>*>>
  MutFirstUndeterminedBigGraphNodes(GraphMatchCtx<bg_node_t>* ctx) const {
    std::optional<std::list<bg_node_t>*> ret;
    using LoopCtrl = adt::Result<adt::LoopCtrl>;
    auto Find = [&](auto* lst) -> LoopCtrl {
      if (lst->size() > 1) {
        ret = lst;
        return adt::Break{};
      }
      return adt::Continue{};
    };
    ADT_RETURN_IF_ERR(
        (*ctx)->topo_match_ctx.shared_ptr()->LoopMutBigGraphNode(Find));
    return ret;
  }

  template <typename RematchT>
  adt::Result<adt::Ok> InplaceForcePickOneLastUndetermined(
      GraphMatchCtx<bg_node_t>* ctx,
      const RematchT& Rematch,
      int loop_limit) const {
    while (true) {
      if (--loop_limit < 0) {
        return adt::errors::TypeError{
            "dead loop detected in InplaceForcePickOneLastUndetermined()"};
      }
      ADT_LET_CONST_REF(num_undetermined_nodes, NumUndeterminedNodes(*ctx));
      if (num_undetermined_nodes == 0) {
        return adt::Ok{};
      }
      if (num_undetermined_nodes == 1) {
        break;
      }
      ADT_LET_CONST_REF(opt_lst, MutFirstUndeterminedBigGraphNodes(ctx));
      ADT_CHECK(opt_lst.has_value());
      ADT_CHECK(opt_lst.value()->size() > 1);
      opt_lst.value()->resize(1);
      ADT_LET_CONST_REF(ctrl, Rematch(ctx));
      if (ctrl.template Has<adt::Break>()) {
        return adt::Ok{};
      }
    }
    ADT_LET_CONST_REF(opt_lst, MutFirstUndeterminedBigGraphNodes(ctx));
    ADT_CHECK(opt_lst.has_value());
    ADT_CHECK(opt_lst.value()->size() > 1);
    std::list<bg_node_t> candidate_lst(*opt_lst.value());
    opt_lst.value()->resize(1);
    for (const auto& node : candidate_lst) {
      *opt_lst.value()->begin() = node;
      ADT_LET_CONST_REF(ctrl, Rematch(ctx));
      if (ctrl.template Has<adt::Break>()) {
        return adt::Ok{};
      }
    }
    return adt::Ok{};
  }

 private:
  adt::Result<TopoMatchCtx<bg_node_t, sg_node_t>> MakeTopoMatchCtxFromAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    TopoMatchCtx<bg_node_t, sg_node_t> match_ctx{};
    const auto& ptn_bfs_walker =
        GraphHelper<sg_node_t, SGTopoKind>(sg_descriptor_).GetBfsWalker();
    auto InitMatchCtx = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      if (sg_node == anchor_node) {
        std::list<bg_node_t> bg_nodes{bg_node};
        ADT_RETURN_IF_ERR(match_ctx->InitBigGraphNodes(anchor_node, bg_nodes));
      } else {
        ADT_RETURN_IF_ERR(TopoMatchCtxInitNode(&*match_ctx, sg_node));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ptn_bfs_walker(anchor_node, InitMatchCtx));
    return match_ctx;
  }

  adt::Result<bool> UpdateAllByConnections(
      TopoMatchCtxImpl<bg_node_t, sg_node_t>* match_ctx,
      const sg_node_t& anchor_node) {
    const auto& ptn_bfs_walker =
        GraphHelper<sg_node_t, SGTopoKind>(sg_descriptor_).GetBfsWalker();
    bool updated = false;
    auto Update = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      // no need to update anchor_node.
      if (anchor_node == sg_node) {
        return adt::Ok{};
      }
      if (match_ctx->HasBigGraphNode(sg_node)) {
        ADT_LET_CONST_REF(current_updated,
                          UpdateByConnections(match_ctx, sg_node));
        updated = updated || current_updated;
      } else {
        ADT_RETURN_IF_ERR(TopoMatchCtxInitNode(match_ctx, sg_node));
        updated = true;
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ptn_bfs_walker(anchor_node, Update));
    return updated;
  }

  adt::Result<bool> UpdateByConnections(
      TopoMatchCtxImpl<bg_node_t, sg_node_t>* ctx, const sg_node_t& sg_node) {
    ADT_LET_CONST_REF(bg_nodes_ptr, ctx->GetBigGraphNodes(sg_node));
    const size_t old_num_bg_nodes = bg_nodes_ptr->size();
    auto Update = [&](const sg_node_t& nearby_node,
                      tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(bg_nodes,
                        GetMatchedBigGraphNodesFromConnected(
                            *ctx, sg_node, nearby_node, is_upstream));
      ADT_CHECK(!bg_nodes.empty()) << adt::errors::RuntimeError{
          std::string() + "small_graph_node: " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node) +
          ", old_big_graph_nodes: " + GetNodesDebugIds(bg_nodes_ptr) +
          ", nearby_node: " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(nearby_node) +
          ", is_nearby_node_from_upstream: " +
          std::to_string(is_upstream.value())};
      ADT_RETURN_IF_ERR(ctx->UpdateBigGraphNodes(sg_node, bg_nodes));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, sg_node, Update));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, sg_node, Update));
    return old_num_bg_nodes != bg_nodes_ptr->size();
  }

  std::string GetNodesDebugIds(const std::list<bg_node_t>* nodes) const {
    std::ostringstream ss;
    int i = 0;
    for (const auto& node : *nodes) {
      if (i++ > 0) {
        ss << " ";
      }
      ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(node);
    }
    return ss.str();
  }

  adt::Result<adt::Ok> TopoMatchCtxInitNode(
      TopoMatchCtxImpl<bg_node_t, sg_node_t>* ctx, const sg_node_t& sg_node) {
    ADT_CHECK(!ctx->HasBigGraphNode(sg_node));
    bool inited = false;
    auto InitOrUpdate =
        [&](const sg_node_t& node,
            tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      if (!inited) {
        ADT_LET_CONST_REF(bg_nodes,
                          GetInitialMatchedBigGraphNodesFromConnected(
                              *ctx, sg_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->InitBigGraphNodes(sg_node, bg_nodes));
        inited = (bg_nodes.size() > 0);
      } else {
        ADT_LET_CONST_REF(bg_nodes,
                          GetMatchedBigGraphNodesFromConnected(
                              *ctx, sg_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->UpdateBigGraphNodes(sg_node, bg_nodes));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, sg_node, InitOrUpdate));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, sg_node, InitOrUpdate));
    ADT_CHECK(inited) << adt::errors::MismatchError{
        "sg_node not successfully inited."};
    return adt::Ok{};
  }

  adt::Result<std::list<bg_node_t>> GetInitialMatchedBigGraphNodesFromConnected(
      const TopoMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream) {
    std::list<bg_node_t> bg_nodes;
    const auto& DoEachMatched =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      bg_nodes.emplace_back(bg_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedBigGraphNodesFromConnected(
        ctx, sg_node, from_node, is_from_node_upstream, DoEachMatched));
    return bg_nodes;
  }

  adt::Result<std::unordered_set<bg_node_t>>
  GetMatchedBigGraphNodesFromConnected(
      const TopoMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream) {
    std::unordered_set<bg_node_t> bg_nodes;
    const auto& DoEachMatched =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      bg_nodes.insert(bg_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedBigGraphNodesFromConnected(
        ctx, sg_node, from_node, is_from_node_upstream, DoEachMatched));
    return bg_nodes;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedBigGraphNodesFromConnected(
      const TopoMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(sg_node_topo_cstr,
                      sg_descriptor_.GetSmallGraphNodeTopoCstr(sg_node));
    const auto& VisitBigGraphNode =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(topo_matched,
                        bg_descriptor_.TopoSatisfy(bg_node, sg_node_topo_cstr));
      bool matched = topo_matched;
      if (matched) {
        ap::graph::NodeDescriptor<bg_node_t> node_descriptor{};
        ADT_LET_CONST_REF(
            attrs_matched,
            node_descriptor.AttrsSatisfyIfBothAreOpsOrValues(bg_node, sg_node));
        matched = attrs_matched;
      }
      if (!matched) {
        return adt::Ok{};
      }
      const auto& opt_matched_sg_node = ctx.GetMatchedSmallGraphNode(bg_node);
      if (!opt_matched_sg_node.has_value() ||
          opt_matched_sg_node.value() == sg_node) {
        return DoEach(bg_node);
      }
      return adt::Ok{};
    };
    ADT_LET_CONST_REF(from_bg_nodes_ptr, ctx.GetBigGraphNodes(from_node));
    for (const bg_node_t& from_bg_node : *from_bg_nodes_ptr) {
      if (is_from_node_upstream.value()) {
        ADT_RETURN_IF_ERR(bg_descriptor_.VisitDownstreamNodes(
            from_bg_node, VisitBigGraphNode));
      } else {
        ADT_RETURN_IF_ERR(
            bg_descriptor_.VisitUpstreamNodes(from_bg_node, VisitBigGraphNode));
      }
    }
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedUpstream(
      const TopoMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const DoEachT& DoEach) {
    auto Visit = [&](const sg_node_t& src) -> adt::Result<adt::Ok> {
      if (ctx.HasBigGraphNode(src)) {
        return DoEach(src, tIsUpstream<bool>{true});
      }
      return adt::Ok{};
    };
    return sg_descriptor_.VisitUpstreamNodes(sg_node, Visit);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedDownstream(
      const TopoMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const DoEachT& DoEach) {
    auto Visit = [&](const sg_node_t& dst) -> adt::Result<adt::Ok> {
      if (ctx.HasBigGraphNode(dst)) {
        return DoEach(dst, tIsUpstream<bool>{false});
      }
      return adt::Ok{};
    };
    return sg_descriptor_.VisitDownstreamNodes(sg_node, Visit);
  }

  GraphDescriptor<bg_node_t, BGTopoKind> bg_descriptor_;
  GraphDescriptor<sg_node_t, SGTopoKind> sg_descriptor_;
};

}  // namespace ap::ir_match
