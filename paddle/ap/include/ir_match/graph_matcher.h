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
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/adt.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/graph_match_ctx.h"
#include "paddle/ap/include/ir_match/topo_matcher.h"

namespace ap::ir_match {

template <typename bg_node_t, typename BGTopoKind, typename SGTopoKind>
struct GraphMatcher {
  using DrrNode = drr::Node;
  using DrrNativeIrOp = drr::NativeIrOp<DrrNode>;
  using sg_node_t = graph::Node<DrrNode>;

  TopoMatcher<bg_node_t, sg_node_t, BGTopoKind, SGTopoKind> topo_matcher_;

  GraphMatcher(const GraphDescriptor<bg_node_t, BGTopoKind>& bg_descriptor,
               const GraphDescriptor<sg_node_t, SGTopoKind>& sg_descriptor)
      : topo_matcher_(bg_descriptor, sg_descriptor) {}

  GraphMatcher(const GraphMatcher&) = delete;
  GraphMatcher(GraphMatcher&&) = delete;

  adt::Result<GraphMatchCtx<bg_node_t>> MatchByAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(topo_match_ctx,
                      topo_matcher_.MatchByAnchor(bg_node, anchor_node));
    return GraphMatchCtx<bg_node_t>{topo_match_ctx};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMisMatchedNodes(
      const GraphMatchCtx<bg_node_t>& graph_match_ctx,
      const sg_node_t& anchor_node,
      const DoEachT& DoEach) const {
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    return topo_matcher_.VisitMisMatchedNodes(
        topo_match_ctx, anchor_node, DoEach);
  }

  adt::Result<adt::Ok> UpdateByConnectionsUntilDone(
      GraphMatchCtx<bg_node_t>* ctx, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(new_topo_match_ctx,
                      Solve((*ctx)->topo_match_ctx, anchor_node));
    (*ctx)->topo_match_ctx = new_topo_match_ctx;
    return adt::Ok{};
  }

  adt::Result<bool> IsGraphMatched(const GraphMatchCtx<bg_node_t>& ctx,
                                   const sg_node_t& anchor_node) const {
    return topo_matcher_.IsGraphMatched(ctx->topo_match_ctx, anchor_node);
  }

  adt::Result<bool> HasUndetermined(const GraphMatchCtx<bg_node_t>& ctx) const {
    return topo_matcher_.HasUndetermined(ctx);
  }

  template <typename ReMatchT>
  adt::Result<adt::Ok> InplaceForcePickOneLastUndetermined(
      GraphMatchCtx<bg_node_t>* ctx, const ReMatchT& ReMatch) const {
    return InplaceForcePickOneLastUndetermined(
        ctx, ReMatch, /*loop_limit=*/9999);
  }

  template <typename ReMatchT>
  adt::Result<adt::Ok> InplaceForcePickOneLastUndetermined(
      GraphMatchCtx<bg_node_t>* ctx,
      const ReMatchT& ReMatch,
      int loop_limit) const {
    return topo_matcher_.InplaceForcePickOneLastUndetermined(
        ctx, ReMatch, loop_limit);
  }

 private:
  using TopoMatchCtxT = TopoMatchCtx<bg_node_t, sg_node_t>;

  adt::Result<TopoMatchCtxT> Solve(TopoMatchCtxT topo_match_ctx,
                                   const sg_node_t& anchor_node) {
    ADT_RETURN_IF_ERR(topo_matcher_.UpdateByConnectionsUntilDone(
        &*topo_match_ctx, anchor_node));
    const auto& opt_iter = topo_match_ctx->GetFirstUnsolved();
    if (!opt_iter.has_value()) {
      return topo_match_ctx;
    }
    const auto& unsolved_sg_node = opt_iter.value()->first;
    for (const auto& proprosal_bg_node : opt_iter.value()->second) {
      ADT_LET_CONST_REF(impl,
                        topo_match_ctx->CloneAndSetUnsolved(unsolved_sg_node,
                                                            proprosal_bg_node));
      TopoMatchCtxT proprosal_topo_match_ctx{impl};
      ADT_LET_CONST_REF(solved, Solve(proprosal_topo_match_ctx, anchor_node));
      if (!solved->GetFirstMismatched().has_value()) {
        return solved;
      }
    }
    // all proposals failed.
    return topo_match_ctx;
  }
};

}  // namespace ap::ir_match
