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

#include "paddle/cinn/common/is_reachable_predicator.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/fuse_helper.h"

namespace cinn {
namespace hlir {
namespace pass {

template <typename FusePassCtxT>
class GraphGroupFuseHelper final : public FuseHelper {
 public:
  explicit GraphGroupFuseHelper(const FusePassCtxT* ctx) : ctx_(ctx) {}

  bool AllOutputsSameSize(const OpGroupPtr& first,
                          const OpGroupPtr& second) const override;

  bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src,
                                       const OpGroupPtr& dst) const override;

  bool ElementwiseFuseBroadcast(const OpGroupPtr& src,
                                const OpGroupPtr& dst) const override;

  bool HorizontalWithInjective(const OpGroupPtr& src,
                               const OpGroupPtr& dst) const override;

  bool ElementwiseFuseReduce(const OpGroupPtr& src,
                             const OpGroupPtr& dst) const override;

  bool BroadcastFuseReduce(const OpGroupPtr& src,
                           const OpGroupPtr& dst) const override;

  bool InjectiveHorizontalWithReduce(const OpGroupPtr& src,
                                     const OpGroupPtr& dst) const override;

  bool ReduceFuseElementwise(const OpGroupPtr& src,
                             const OpGroupPtr& dst) const override;

  bool ReduceFuseBroadcast(const OpGroupPtr& src,
                           const OpGroupPtr& dst) const override;

  bool ReduceFuseReduce(const OpGroupPtr& src,
                        const OpGroupPtr& dst) const override;

  bool IsReachable(const OpGroupPtr& lhs,
                   const OpGroupPtr& rhs) const override {
    return IsReachableInDag(lhs, rhs) || IsReachableInDag(rhs, lhs);
  }

  bool DetectCycleIfFuse(const OpGroupPtr& lhs,
                         const OpGroupPtr& rhs) const override {
    return ReachableIfDirectEdgeIgnored(lhs, rhs) ||
           ReachableIfDirectEdgeIgnored(rhs, lhs);
  }

  bool IsConsumerSetsReachable(
      const OpGroupPtr& group,
      const std::unordered_set<OpGroupPtr>& consumers) const override {
    for (const auto& consumer : consumers) {
      if (group == consumer) {
        continue;
      }
      if (IsReachableInDag(consumer, group)) {
        return true;
      }
    }
    return false;
  }

 private:
  bool IsReachableInDag(const OpGroupPtr& producer,
                        const OpGroupPtr& consumer) const {
    const auto& MinDepth4Node = [&](const OpGroupPtr& node) {
      return node.GetGroup()->min_depth;
    };
    const auto& MaxDepth4Node = [&](const OpGroupPtr& node) {
      return node.GetGroup()->max_depth;
    };
    const auto& VisitNextNodes =
        [&](const OpGroupPtr& node,
            const std::function<void(OpGroupPtr)>& Visit) {
          for (const auto& node_producer : node.producers()) {
            Visit(node_producer);
          }
        };
    common::IsReachablePredicator<OpGroupPtr> is_reachable(
        MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    return is_reachable(consumer, producer, [](OpGroupPtr) {});
  }

  bool ReachableIfDirectEdgeIgnored(const OpGroupPtr& producer,
                                    const OpGroupPtr& consumer) const {
    const auto& MinDepth4Node = [&](const OpGroupPtr& node) {
      return node.GetGroup()->min_depth;
    };
    const auto& MaxDepth4Node = [&](const OpGroupPtr& node) {
      return node.GetGroup()->max_depth;
    };
    const auto& VisitNextNodes =
        [&](const OpGroupPtr& node,
            const std::function<void(OpGroupPtr)>& Visit) {
          for (const auto& node_producer : node.producers()) {
            if (node == consumer && node_producer == producer) {
              continue;
            }
            Visit(node_producer);
          }
        };
    common::IsReachablePredicator<OpGroupPtr> is_reachable(
        MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    return is_reachable(consumer, producer, [](OpGroupPtr) {});
  }

  const FusePassCtxT* ctx_;
};

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::AllOutputsSameSize(
    const OpGroupPtr& first, const OpGroupPtr& second) const {
  return is_same_size(
      &ctx_->graph_group_fusion_helper(), first.GetGroup(), second.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return honrizontal_elementwise_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_broadcast(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalWithInjective(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return horizontal_with_injective(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::BroadcastFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return broadcast_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::InjectiveHorizontalWithReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return injective_horizontal_with_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseElementwise(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_elementwise(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_broadcast(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_reduce(
      &ctx_->graph_group_fusion_helper(), src.GetGroup(), dst.GetGroup());
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
