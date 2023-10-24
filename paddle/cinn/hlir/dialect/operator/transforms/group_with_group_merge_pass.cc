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

#include <map>
#include <set>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/transforms/op_group.h"
#include "paddle/pir/core/value.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_with_group_merge_pass_utils.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/op_with_group_merge_pass.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/op_with_group_merge_util.h"
#include "paddle/phi/core/flags.h"

PD_DECLARE_bool(enhance_vertical_fusion_with_recompute);

namespace cinn {
namespace dialect {
namespace ir {

using GroupPtr = std::shared_ptr<ir::Group>;
using GroupList = std::vector<GroupPtr>;

using Comparator = ir::Group::SharedGroupComparator;
using Hasher = ir::Group::SharedGroupHasher;

using OpGroupPtr = ir::OpGroup;
using OpGroupList = std::vector<OpGroupPtr>;

using ConditionFunction = std::function<bool(const GroupPtr&, const GroupPtr&)>;

class FuseHelper {
 public:
  virtual ~FuseHelper() = default;

  virtual bool AllOutputsSameSize(const OpGroupPtr& first,
                                  const OpGroupPtr& second) const = 0;

  virtual bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src,
                                               const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseBroadcast(const OpGroupPtr& src,
                                        const OpGroupPtr& dst) const = 0;

  virtual bool HorizontalWithInjective(const OpGroupPtr& src,
                                       const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseReduce(const OpGroupPtr& src,
                                     const OpGroupPtr& dst) const = 0;

  virtual bool BroadcastFuseReduce(const OpGroupPtr& src,
                                   const OpGroupPtr& dst) const = 0;

  virtual bool InjectiveHorizontalWithReduce(const OpGroupPtr& src,
                                             const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseElementwise(const OpGroupPtr& src,
                                     const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseBroadcast(const OpGroupPtr& src,
                                   const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseReduce(const OpGroupPtr& src,
                                const OpGroupPtr& dst) const = 0;

  virtual bool IsReachable(const OpGroupPtr& lhs,
                           const OpGroupPtr& rhs) const = 0;

  virtual bool DetectCycleIfFuse(const OpGroupPtr& src,
                                 const OpGroupPtr& dst) const = 0;

  virtual bool IsConsumerSetsReachable(
      const OpGroupPtr& group,
      const std::unordered_set<OpGroupPtr>& consumers) const = 0;

 protected:
  FuseHelper() = default;
};

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
    // const auto& MinDepth4Node = [&](const OpGroupPtr& node) {
    //   return node.GetGroup()->min_depth;
    // };
    // const auto& MaxDepth4Node = [&](const OpGroupPtr& node) {
    //   return node.GetGroup()->max_depth;
    // };
    // const auto& VisitNextNodes =
    //     [&](const OpGroupPtr& node,
    //         const std::function<void(OpGroupPtr)>& Visit) {
    //       for (const auto& node_producer : node.producers()) {
    //         Visit(node_producer);
    //       }
    //     };
    // common::IsReachablePredicator<OpGroupPtr> is_reachable(
    //     MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    // return is_reachable(consumer, producer, [](OpGroupPtr) {});
    // TODO(phlrain) : support IsReachable
    return false;
  }

  bool ReachableIfDirectEdgeIgnored(const OpGroupPtr& producer,
                                    const OpGroupPtr& consumer) const {
    // const auto& MinDepth4Node = [&](const OpGroupPtr& node) {
    //   return node.GetGroup()->min_depth;
    // };
    // const auto& MaxDepth4Node = [&](const OpGroupPtr& node) {
    //   return node.GetGroup()->max_depth;
    // };
    // const auto& VisitNextNodes =
    //     [&](const OpGroupPtr& node,
    //         const std::function<void(OpGroupPtr)>& Visit) {
    //       for (const auto& node_producer : node.producers()) {
    //         if (node == consumer && node_producer == producer) {
    //           continue;
    //         }
    //         Visit(node_producer);
    //       }
    //     };
    // common::IsReachablePredicator<OpGroupPtr> is_reachable(
    //     MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    // return is_reachable(consumer, producer, [](OpGroupPtr) {});
    // TODO(phlrain) : support IsReachable
    return false;
  }

  const FusePassCtxT* ctx_;
};

class FusePassCtx {
 public:
  virtual ~FusePassCtx() {}

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void MarkFusible(const OpGroupPtr& first,
                           const OpGroupPtr& second) = 0;

 protected:
  FusePassCtx() = default;
};

class LightwareFusePassCtx : public FusePassCtx {
 public:
  virtual ~LightwareFusePassCtx() {}

  virtual const OpGroupPtr& PickOpGroup() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void MarkFusible(const OpGroupPtr& first,
                           const OpGroupPtr& second) = 0;

  virtual void MarkFusible(const OpGroupList& candidates) = 0;

 protected:
  LightwareFusePassCtx() = default;
};

class GraphGroupLightwareFusePassCtx final : public LightwareFusePassCtx {
 public:
  GraphGroupLightwareFusePassCtx(
      const OpGroupPtr& group,
      const std::function<void(const OpGroupPtr& first,
                               const OpGroupPtr& second)>& MarkFusible)
      : group_(group),
        MarkFusible_(MarkFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  GraphGroupLightwareFusePassCtx(
      const OpGroupPtr& group,
      const std::function<void(const OpGroupList& candidates)>&
          MarkGroupListFusible)
      : group_(group),
        MarkGroupListFusible_(MarkGroupListFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  const OpGroupPtr& PickOpGroup() const override { return group_; }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void MarkFusible(const OpGroupPtr& first, const OpGroupPtr& second) override {
    MarkFusible_(first, second);
  }

  void MarkFusible(const OpGroupList& candidates) override {
    MarkGroupListFusible_(candidates);
  }

 private:
  const OpGroupPtr& group_;
  const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)>
      MarkFusible_;
  const std::function<void(const OpGroupList& candidates)>
      MarkGroupListFusible_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

class InputFusePassCtx : public FusePassCtx {
 public:
  virtual ~InputFusePassCtx() {}

  virtual const OpGroupList& PickConsumersWithSameInputs() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void MarkFusible(const OpGroupPtr& first,
                           const OpGroupPtr& second) = 0;

  virtual void MarkFusible(const OpGroupList& candidates) = 0;

 protected:
  InputFusePassCtx() = default;
};

class GraphGroupInputFusePassCtx final : public InputFusePassCtx {
 public:
  GraphGroupInputFusePassCtx(
      const OpGroupList& groups,
      const std::function<void(const OpGroupPtr& first,
                               const OpGroupPtr& second)>& MarkFusible)
      : groups_(groups),
        MarkFusible_(MarkFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupInputFusePassCtx>(this)) {}

  GraphGroupInputFusePassCtx(
      const OpGroupList& groups,
      const std::function<void(const OpGroupList& candidates)>&
          MarkGroupListFusible)
      : groups_(groups),
        MarkGroupListFusible_(MarkGroupListFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupInputFusePassCtx>(this)) {}

  const OpGroupList& PickConsumersWithSameInputs() const override {
    return groups_;
  }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void MarkFusible(const OpGroupPtr& first, const OpGroupPtr& second) override {
    MarkFusible_(first, second);
  }

  void MarkFusible(const OpGroupList& candidates) override {
    MarkGroupListFusible_(candidates);
  }

 private:
  const OpGroupList& groups_;
  const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)>
      MarkFusible_;
  const std::function<void(const OpGroupList& candidates)>
      MarkGroupListFusible_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::AllOutputsSameSize(
    const OpGroupPtr& first, const OpGroupPtr& second) const {
  return is_same_size(first.GetGroup(), second.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return honrizontal_elementwise_fuse_reduce(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_broadcast(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalWithInjective(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return horizontal_with_injective(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_reduce(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::BroadcastFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return broadcast_fuse_reduce(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::InjectiveHorizontalWithReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return injective_horizontal_with_reduce(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseElementwise(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_elementwise(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseBroadcast(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_broadcast(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseReduce(
    const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_reduce(src.GetGroup(), dst.GetGroup());
}

template <typename FusePassCtxT>
struct HorizontalFuseUtil {
  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;

  static bool DetectFusabilityByKind(FusePassCtxT* ctx,
                                     const OpGroupPtr& src,
                                     const OpGroupPtr& dst) {
    const KindKeyT kind_pair(src.kind(), dst.kind());
    const auto& map = GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    auto out = iter->second(src, dst);
    return out;
  }

  typedef bool (*ConditionT)(const OpGroupPtr& src, const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{kElementWise, kElementWise}, &IsSameSize},
        {{kElementWise, kBroadcast}, &IsSameSize},
        {{kElementWise, kInjective}, &IsSameSize},
        {{kElementWise, kReduction}, &HorizontalElementwiseFuseReduce},

        {{kBroadcast, kElementWise}, &IsSameSize},
        {{kBroadcast, kBroadcast}, &IsSameSize},
        {{kBroadcast, kInjective}, &IsSameSize},
        {{kBroadcast, kReduction}, &IsSameSize},

        {{kInjective, kElementWise}, &IsSameSize},
        {{kInjective, kBroadcast}, &IsSameSize},
        {{kInjective, kInjective}, &IsSameSize},
        {{kInjective, kReduction}, &IsSameSize},

        {{kReduction, kElementWise}, &HorizontalElementwiseFuseReduce},
        {{kReduction, kBroadcast}, &IsSameSize},
        {{kReduction, kInjective}, &IsSameSize},
        {{kReduction, kReduction}, &ReduceFuseReduce},
    };
  }

  static bool IsSameSize(const OpGroupPtr& src, const OpGroupPtr& dst) {
    return cinn::dialect::ir::IsSameSize(src, dst);
  }

  static bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src,
                                              const OpGroupPtr& dst) {
    // if same shape with horizontal relation
    if (IsSameSize(src, dst)) {
      return true;
    }

    const OpGroupPtr* ele_group = nullptr;
    const OpGroupPtr* reduce_group = nullptr;

    if (src.kind() == kReduction) {
      ele_group = &dst;
      reduce_group = &src;
    } else {
      ele_group = &src;
      reduce_group = &dst;
    }

    size_t size_ele =
        phi::product(GetMasterNode(*ele_group).outputs()[0].shape());

    bool can_fuse = false;
    reduce_group->WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
      if (op.kind() == OpPatternKind::kReduction) {
        size_t size_master = phi::product(op.outputs()[0].shape());
        if (size_ele == size_master) {
          can_fuse = true;
        }
      }
    });

    return can_fuse;
  }

  static bool ReduceFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) {
    // return ctx->fuse_helper().ReduceFuseReduce(src, dst);
    return reduce_fuse_reduce(src.GetGroup(), dst.GetGroup());
  }
};

class FusePass {
 public:
  virtual ~FusePass() = default;

  virtual const std::string FuseMode() const = 0;

  virtual int Benefit() const = 0;

 protected:
  FusePass() = default;
};

class InputFusePass : public FusePass {
 public:
  virtual ~InputFusePass() = default;

  virtual void operator()(InputFusePassCtx* ctx) const = 0;

  const std::string FuseMode() const final { return "InputFuse"; }

  virtual int Benefit() const = 0;

 protected:
  InputFusePass() = default;
};

class DefaultInputFusePass final : public InputFusePass {
 public:
  DefaultInputFusePass() : InputFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(InputFusePassCtx* ctx) const override {
    const auto& consumer_set = ctx->PickConsumersWithSameInputs();

    const std::unordered_set<OpGroupPtr> consumer_candidates =
        [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& consumer : consumer_set) {
        if (consumer.kind() == kElementWise || consumer.kind() == kBroadcast ||
            consumer.kind() == kInjective || consumer.kind() == kReduction) {
          consumers.insert(consumer);
        }
      }
      return consumers;
    }();
    if (consumer_candidates.size() <= 1) {
      return;
    }

    std::vector<OpGroupList> fusionable_consumers;
    for (auto& candidate : consumer_candidates) {
      if (ctx->fuse_helper().IsConsumerSetsReachable(candidate,
                                                     consumer_candidates)) {
        continue;
      }
      if (fusionable_consumers.empty()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }
      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!HorizontalFuseUtil<InputFusePassCtx>::DetectFusabilityByKind(
                ctx, candidate, last)) {
          continue;
        }
        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    for (const auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        ctx->MarkFusible(groups);
      }
    }
    VLOG(1) << "DefaultInputFusePass Finish";
  }
};

class LightwareFusePass : public FusePass {
 public:
  virtual ~LightwareFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const = 0;

  virtual int Benefit() const = 0;

 protected:
  LightwareFusePass() = default;
};

class HorizontalFusePass : public LightwareFusePass {
 public:
  virtual ~HorizontalFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  const std::string FuseMode() const final { return "HorizontalFuse"; }

  virtual int Benefit() const = 0;

 protected:
  HorizontalFusePass() = default;
};

class DefaultHorizontalFusePass final : public HorizontalFusePass {
 public:
  DefaultHorizontalFusePass() : HorizontalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer = ctx->PickOpGroup();
    const std::unordered_set<OpGroupPtr> consumer_candidates =
        [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& consumer : producer.consumers()) {
        if (consumer.kind() == kElementWise || consumer.kind() == kBroadcast ||
            consumer.kind() == kInjective || consumer.kind() == kReduction) {
          consumers.insert(consumer);
        }
      }
      return consumers;
    }();
    if (consumer_candidates.size() <= 1) {
      return;
    }

    std::vector<OpGroupList> fusionable_consumers;
    for (auto& candidate : consumer_candidates) {
      if (ctx->fuse_helper().IsConsumerSetsReachable(candidate,
                                                     consumer_candidates)) {
        continue;
      }
      if (fusionable_consumers.empty()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }
      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!HorizontalFuseUtil<LightwareFusePassCtx>::DetectFusabilityByKind(
                ctx, candidate, last)) {
          continue;
        }
        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    for (const auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        // Trick for BERT, maybe not required, wait for substitution from
        // unordered_set to set
        if (groups.size() == 2) {
          OpGroupList fuse_group;
          if (groups[1].group_id().substr(0, 4) == "cast" &&
              groups[0].group_id() == "reshape_split") {
            fuse_group.push_back(groups[1]);
            fuse_group.push_back(groups[0]);
            ctx->MarkFusible(fuse_group);
            continue;
          }
        }
        ctx->MarkFusible(groups);
      }
    }
  }
};

class VerticalFusePass : public LightwareFusePass {
 public:
  virtual ~VerticalFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  const std::string FuseMode() const final { return "VerticalFuse"; }

  virtual int Benefit() const = 0;

 protected:
  VerticalFusePass() = default;
};

class DefaultVerticalFusePass final : public VerticalFusePass {
 public:
  DefaultVerticalFusePass() : VerticalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer = ctx->PickOpGroup();
    const OpGroupList consumers = [&]() {
      OpGroupList consumers;
      for (const auto& consumer : producer.consumers()) {
        consumers.push_back(consumer);
      }
      return consumers;
    }();
    if (consumers.size() == 0) {
      return;
    }

    std::vector<OpGroupPtr> candidates;
    for (size_t i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        break;
      }
      candidates.push_back(consumer);
    }
    if (candidates.size() == consumers.size() &&
        producer.kind() == kElementWise) {
      return;
    }

    for (size_t i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        VLOG(4) << "Can't fuse because detect cycle";
        continue;
      }
      ctx->MarkFusible(producer, consumer);
    }
  }

  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;
  bool DetectFusabilityByKind(LightwareFusePassCtx* ctx,
                              const OpGroupPtr& src,
                              const OpGroupPtr& dst) const {
    const KindKeyT kind_pair(src.kind(), dst.kind());
    const auto& map = GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }

  typedef bool (*ConditionT)(LightwareFusePassCtx* ctx,
                             const OpGroupPtr& src,
                             const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{OpPatternKind::kElementWise, kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kElementWise, kBroadcast},
         &DefaultVerticalFusePass::ElementwiseFuseBroadcast},
        {{OpPatternKind::kElementWise, kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kElementWise, kReduction},
         &DefaultVerticalFusePass::ElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, kBroadcast},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kBroadcast, kReduction},
         &DefaultVerticalFusePass::BroadcastFuseReduce},

        {{OpPatternKind::kInjective, kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, kBroadcast},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kInjective, kReduction},
         &DefaultVerticalFusePass::InjectiveHorizontalWithReduce},

        {{OpPatternKind::kReduction, kElementWise},
         &DefaultVerticalFusePass::ReduceFuseElementwise},
        {{OpPatternKind::kReduction, kBroadcast},
         &DefaultVerticalFusePass::ReduceFuseBroadcast},
        {{OpPatternKind::kReduction, kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kReduction, kReduction},
         &DefaultVerticalFusePass::ReduceFuseReduce},
    };
  }

  static bool IsSameSize(LightwareFusePassCtx* ctx,
                         const OpGroupPtr& src,
                         const OpGroupPtr& dst) {
    return cinn::dialect::ir::IsSameSize(src, dst);
  }

  static bool ElementwiseFuseBroadcast(LightwareFusePassCtx* ctx,
                                       const OpGroupPtr& src,
                                       const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseBroadcast(src, dst);
  }

  static bool HorizontalWithInjective(LightwareFusePassCtx* ctx,
                                      const OpGroupPtr& src,
                                      const OpGroupPtr& dst) {
    return ctx->fuse_helper().HorizontalWithInjective(src, dst);
  }

  static bool ElementwiseFuseReduce(LightwareFusePassCtx* ctx,
                                    const OpGroupPtr& src,
                                    const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseReduce(src, dst);
  }

  static bool BroadcastFuseReduce(LightwareFusePassCtx* ctx,
                                  const OpGroupPtr& src,
                                  const OpGroupPtr& dst) {
    return ctx->fuse_helper().BroadcastFuseReduce(src, dst);
  }

  static bool InjectiveHorizontalWithReduce(LightwareFusePassCtx* ctx,
                                            const OpGroupPtr& src,
                                            const OpGroupPtr& dst) {
    return ctx->fuse_helper().InjectiveHorizontalWithReduce(src, dst);
  }

  static bool ReduceFuseElementwise(LightwareFusePassCtx* ctx,
                                    const OpGroupPtr& src,
                                    const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseElementwise(src, dst);
  }

  static bool ReduceFuseBroadcast(LightwareFusePassCtx* ctx,
                                  const OpGroupPtr& src,
                                  const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseBroadcast(src, dst);
  }

  static bool ReduceFuseReduce(LightwareFusePassCtx* ctx,
                               const OpGroupPtr& src,
                               const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseReduce(src, dst);
  }
};

class RecomputeFusePass : public LightwareFusePass {
 public:
  virtual ~RecomputeFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  const std::string FuseMode() const final { return "RecomputeFuse"; }

  virtual int Benefit() const = 0;

 protected:
  RecomputeFusePass() = default;
};

class DefaultRecomputeFusePass final : public RecomputeFusePass {
 public:
  DefaultRecomputeFusePass() : RecomputeFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer = ctx->PickOpGroup();
    const OpGroupList consumers = [&]() {
      OpGroupList consumers;
      for (const auto& consumer : producer.consumers()) {
        consumers.push_back(consumer);
      }
      return consumers;
    }();
    // Borrows unsafe_candidates and candidates concept from origin
    // fusion_merge_pass
    std::vector<OpGroupPtr> unsafe_candidates;
    std::vector<OpGroupPtr> candidates;
    for (size_t i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      unsafe_candidates.push_back(consumer);
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        continue;
      }
      candidates.push_back(consumer);
    }

    if (!candidates.empty() && unsafe_candidates.size() == consumers.size() &&
        producer.kind() == kElementWise) {
      for (const auto& consumer : consumers) {
        ctx->MarkFusible(producer, consumer);
      }
    }
  }

  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;
  bool DetectFusabilityByKind(LightwareFusePassCtx* ctx,
                              const OpGroupPtr& src,
                              const OpGroupPtr& dst) const {
    const KindKeyT kind_pair(src.kind(), dst.kind());
    const auto& map = DefaultVerticalFusePass::GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }
};

struct LightwareFusePassComparator {
  bool operator()(const std::shared_ptr<LightwareFusePass>& lhs,
                  const std::shared_ptr<LightwareFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

struct InputFusePassComparator {
  bool operator()(const std::shared_ptr<InputFusePass>& lhs,
                  const std::shared_ptr<InputFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

class FusionPassMap {
 public:
  static FusionPassMap& Instance() {
    static FusionPassMap global_fusion_pass_map;
    return global_fusion_pass_map;
  }

  bool Has(const std::string& pass_name) const {
    return map_.find(pass_name) != map_.end();
  }

  void Insert(const std::string& pass_name,
              const std::shared_ptr<FusePass>& pass) {
    CHECK(!Has(pass_name)) << "FusePass " << pass_name
                           << " has already been registered.";
    map_.insert({pass_name, pass});
  }

  std::shared_ptr<FusePass> Get(const std::string& pass_name) const {
    auto it = map_.find(pass_name);
    CHECK(it != map_.end())
        << "FusePass " << pass_name << " has not been registered.";
    return it->second;
  }

  // fuse_mode: HorizontalFuse, VerticalFuse, RecomputeFuse
  std::vector<std::shared_ptr<LightwareFusePass>> GetLightwareFusePassesByMode(
      const std::string& fuse_mode) const {
    CHECK(fuse_mode == "HorizontalFuse" || fuse_mode == "VerticalFuse" ||
          fuse_mode == "RecomputeFuse")
        << "fuse_mode only supports HorizontalFuse, VerticalFuse and "
           "RecomputeFuse. Please check your input modes = "
        << fuse_mode;
    std::set<std::shared_ptr<LightwareFusePass>, LightwareFusePassComparator>
        candidate_passes;
    for (const auto& iter : map_) {
      if (fuse_mode == iter.second->FuseMode()) {
        candidate_passes.insert(
            std::dynamic_pointer_cast<LightwareFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<LightwareFusePass>>(
        candidate_passes.begin(), candidate_passes.end());
  }

  std::vector<std::shared_ptr<InputFusePass>> GetInputFusePasses() const {
    std::set<std::shared_ptr<InputFusePass>, InputFusePassComparator>
        candidate_passes;
    for (const auto& iter : map_) {
      if (iter.second->FuseMode() == "InputFuse") {
        candidate_passes.insert(
            std::dynamic_pointer_cast<InputFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<InputFusePass>>(candidate_passes.begin(),
                                                       candidate_passes.end());
  }

 private:
  FusionPassMap() = default;
  std::unordered_map<std::string, std::shared_ptr<FusePass>> map_;

  DISABLE_COPY_AND_ASSIGN(FusionPassMap);
};

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename PassClassT>
class FusionPassRegistrar final : public Registrar {
 public:
  explicit FusionPassRegistrar(const std::string& pass_name) {
    FusionPassMap::Instance().Insert(
        pass_name, std::shared_ptr<PassClassT>(new PassClassT()));
  }
};

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class GeneralFusionMergePassHelper {
 public:
  explicit GeneralFusionMergePassHelper(const GroupList& group_list) {
    fusion_groups_ = group_list;
    // init input to consumers.
    InitInputToConsumers();
    // init fusion group index.
    InitFusionGroupsAndIndex();

    if (!FusionPassMap::Instance().Has("DefaultHorizontalFusePass")) {
      FusionPassMap::Instance().Insert(
          "DefaultHorizontalFusePass",
          std::make_shared<ir::DefaultHorizontalFusePass>());
    }
    if (!FusionPassMap::Instance().Has("DefaultVerticalFusePass")) {
      FusionPassMap::Instance().Insert(
          "DefaultVerticalFusePass",
          std::make_shared<ir::DefaultVerticalFusePass>());
    }

    if (!FusionPassMap::Instance().Has("DefaultRecomputeFusePass")) {
      FusionPassMap::Instance().Insert(
          "DefaultRecomputeFusePass",
          std::make_shared<ir::DefaultRecomputeFusePass>());
    }

    if (!FusionPassMap::Instance().Has("DefaultInputFusePass")) {
      FusionPassMap::Instance().Insert(
          "DefaultInputFusePass", std::make_shared<ir::DefaultInputFusePass>());
    }
  }

  GroupList operator()() {
    // run fusion merge untill no update.
    DoFusionMerge();
    for (auto& group : fusion_groups_) {
      VLOG(3) << "Fusion Group -> " << group->group_id;
      for (auto& sub_group : group->fused_sub_groups) {
        VLOG(3) << "  Fused Sub-Group -> " << sub_group->group_id;
      }
      for (const auto& producer : group->producer_groups()) {
        VLOG(3) << "  Producer -> " << producer->group_id;
      }
      for (const auto& consumer : group->consumer_groups()) {
        VLOG(3) << "  Consumer -> " << consumer->group_id;
      }
    }
    return fusion_groups_;
  }

 private:
  void DoFusionMerge() {
    VLOG(3) << "DoFusionMerge...!";
    while (DoGeneralHorizontalFusion()) {
    }
    while (DoGeneralVerticalFusion()) {
    }
    while (DoGeneralRecomputeAndVerticalFusion()) {
    }
  }

  bool DoGeneralHorizontalFusion() {
    VLOG(3) << "DoGeneralHorizontalFusion...!";
    bool updated = false;
    for (size_t idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
    }

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralVerticalFusion() {
    VLOG(3) << "DoGeneralVerticalFusion...!";
    bool updated = false;
    for (size_t idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
      updated |= GeneralVerticalFuse(producer);
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralRecomputeAndVerticalFusion() {
    VLOG(3) << "DoGeneralRecomputeAndVerticalFusion...!";
    bool updated = false;
    for (size_t idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> "
              << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      bool recompute_success = GeneralRecomputeFuse(producer);
      updated |= recompute_success;
      if (!recompute_success) {
        updated |= GeneralVerticalFuse(producer);
      }
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  void UpdateFusionGroup() {
    VLOG(3) << "UpdateFusionGroup...";
    GroupList fusion_groups;
    std::unordered_set<GroupPtr, Hasher, Comparator> fusion_groups_set;
    // update fusion_groups_
    for (auto& group : fusion_groups_) {
      if (!group->belong_groups.size()) {
        fusion_groups.push_back(group);
        fusion_groups_set.insert(group);
      }
    }
    // keep group in order
    fusion_groups_.clear();
    fusion_groups_index_.clear();
    while (!fusion_groups_set.empty()) {
      bool is_ring = true;
      for (size_t idx = 0; idx < fusion_groups.size(); ++idx) {
        auto& group = fusion_groups[idx];
        if (!group.get()) {
          continue;
        }

        bool exist = false;
        for (const auto& producer : group->producer_groups()) {
          if (fusion_groups_set.count(producer)) {
            VLOG(4) << group->group_id << " " << producer->group_id;
            exist = true;
            break;
          }
        }

        if (!exist) {
          fusion_groups_index_[group] = fusion_groups_.size();
          fusion_groups_.push_back(group);
          fusion_groups_set.erase(group);
          group.reset();
          is_ring = false;
          continue;
        }
      }
      if (is_ring) {
        LOG(FATAL) << "Exists Ring, Please Check!";
      }
    }
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawHorizontalFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "HorizontalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>&
  GetHorizontalFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawHorizontalFusePasses();
    return fuse_passes;
  }

  void EnableFusedHorizontalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer.consumers().size() <= 1) {
      return;
    }
    const auto& fuse_passes = GetHorizontalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralHorizontalFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralHorizontalFuse handling producer : "
            << producer->group_id;
    const auto& GetFusableConsumerGroupLists =
        [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& MarkFusible = [&](const OpGroupList& candidates) {
        tagged_lists.push_back(candidates);
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(ir::OpGroup(producer),
                                              MarkFusible);
      EnableFusedHorizontalGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(group.GetGroup());
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  std::vector<std::shared_ptr<InputFusePass>> RawInputFusePasses() const {
    return FusionPassMap::Instance().GetInputFusePasses();
  }

  const std::vector<std::shared_ptr<InputFusePass>>& GetInputFusePasses()
      const {
    thread_local static std::vector<std::shared_ptr<InputFusePass>>
        fuse_passes = RawInputFusePasses();
    return fuse_passes;
  }

  void EnableFusedInputGroups(InputFusePassCtx* ctx) const {
    const auto& fuse_passes = GetInputFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool CallGeneralInputFusePass(
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    VLOG(3) << "CallGeneralInputFusePass...!";
    const auto& GetFusableConsumerGroupLists =
        [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& MarkFusible = [&](const OpGroupList& candidates) {
        tagged_lists.push_back(candidates);
      };
      OpGroupList consumer_groups;
      consumer_groups.reserve(consumers.size());
      for (auto& consumer : consumers) {
        consumer_groups.push_back(ir::OpGroup(consumer));
      }
      GraphGroupInputFusePassCtx fuse_ctx(consumer_groups, MarkFusible);
      EnableFusedInputGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(group.GetGroup());
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  void HorizontalFuse(const GroupList& consumers) {
    VLOG(3) << "HorizontalFuse Groups...";
    // create fusion group
    auto fused_group = std::make_shared<ir::Group>();
    // As recompute exist which may case sub-group used by more than one time.
    std::vector<GroupPtr> repeat_sub_groups;
    std::unordered_set<GroupPtr, Hasher, Comparator> sub_group_set;
    // find the first consumer.
    GroupPtr first_consumer(nullptr);
    // fuse all group into fusion group.
    for (const auto& consumer : consumers) {
      VLOG(3) << "fuse consumer " << consumer->group_id << " into fused_group!";
      // update depth
      fused_group->max_depth =
          std::max(fused_group->max_depth, consumer->max_depth);
      fused_group->min_depth =
          std::min(fused_group->min_depth, consumer->min_depth);
      // update group id
      if (fused_group->group_id.size()) {
        fused_group->group_id += "_" + consumer->group_id;
      } else {
        fused_group->group_id = consumer->group_id;
      }
      // set op pattern kind
      fused_group->op_pattern_kind =
          static_cast<int>(fused_group->op_pattern_kind) >=
                  static_cast<int>(consumer->op_pattern_kind)
              ? fused_group->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      for (auto& node : consumer->input_nodes) {
        if (fused_group->input_nodes.count(node.first)) {
          fused_group->input_nodes[node.first] += node.second;
        } else {
          fused_group->input_nodes.insert(node);
        }
      }
      // output node
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }
      // internal node
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // master node
      for (auto& node : consumer->master_nodes) {
        if (GetOpKind(node->name()) == kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }
      // insert sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          // check sub group is repeat.
          if (sub_group_set.count(sub_group)) {
            VLOG(3) << sub_group->group_id << " is repeated!";
            repeat_sub_groups.push_back(sub_group);
            continue;
          }
          // record sub group
          sub_group_set.insert(sub_group);

          // insert to fused sub group.
          fused_group->fused_sub_groups.push_back(sub_group);
          // update belongs group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      // producer group
      for (auto& producer : *consumer->mut_producer_groups()) {
        fused_group->mut_producer_groups()->insert(producer);
        // update producer's consumer
        producer->mut_consumer_groups()->erase(consumer);
        producer->mut_consumer_groups()->insert(fused_group);
      }
      // consumer group
      for (auto& gconsumer : *consumer->mut_consumer_groups()) {
        fused_group->mut_consumer_groups()->insert(gconsumer);
        // update consumer's producer
        gconsumer->mut_producer_groups()->erase(consumer);
        gconsumer->mut_producer_groups()->insert(fused_group);
      }
      // belongs group
      consumer->belong_groups.insert(fused_group);

      // find the first consumer.
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id
          << " index in fusion_groups_index_!";
      if (first_consumer.get()) {
        if (fusion_groups_index_[consumer] <
            fusion_groups_index_[first_consumer]) {
          first_consumer = consumer;
        }
      } else {
        first_consumer = consumer;
      }
    }

    // if node is output nodes of sub_group, check it can't be internal node.
    for (auto& sub_group : repeat_sub_groups) {
      // check each output node in sub_group.
      for (auto& node : sub_group->output_nodes) {
        // if node is not output node of fused_group.
        if (!fused_group->output_nodes.count(node)) {
          fused_group->internal_nodes.insert(node);
        }
      }
    }

    if (static_cast<int>(kReduction) >
        static_cast<int>((consumers.back())->op_pattern_kind)) {
      auto consumer = consumers.back();

      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }
    } else {
      for (auto consumer = consumers.rbegin(); consumer != consumers.rend();
           ++consumer) {
        ::pir::Operation* master_node = nullptr;
        for (auto& node : (*consumer)->master_nodes) {
          if (GetOpKind(node->name()) != kReduction) {
            master_node = node;
            break;
          }
        }
        if (master_node) {
          // VLOG(3) << "Insert Master node : " << master_node->id()
          //         << " into group : " << fused_group->group_id;
          fused_group->master_nodes.insert(master_node);
          break;
        }
      }
    }

    auto postion = fusion_groups_index_[first_consumer];
    fusion_groups_[postion] = fused_group;
    fusion_groups_index_[fused_group] = postion;

    CHECK(fused_group->output_nodes.size())
        << "No output node is found, " << fused_group->group_id;
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawVerticalFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "VerticalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>& GetVerticalFusePasses()
      const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawVerticalFusePasses();
    return fuse_passes;
  }

  void TagVerticalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer.consumers().size() == 0) {
      return;
    }
    const auto& fuse_passes = GetVerticalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralVerticalFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralVerticalFuse...!";
    using GroupSets = std::vector<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& MarkFusible = [&](const OpGroupPtr& first,
                                    const OpGroupPtr& second) {
        tagged_sets.push_back(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(ir::OpGroup(producer),
                                              MarkFusible);
      TagVerticalGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet =
        [&]() -> std::unordered_set<GroupPtr, Hasher, Comparator> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr, Hasher, Comparator> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(group_pair.second.GetGroup());
      }
      return ret;
    };

    bool update = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size()) {
      SelectConsumerToFuse(producer, &consumer_groups);
    }
    if (consumer_groups.size() > 0) {
      VerticalFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void VerticalFuse(const GroupPtr& producer,
                    const std::unordered_set<GroupPtr, Hasher, Comparator>&
                        fusionable_consumers) {
    VLOG(3) << "VerticalFuse...!";
    GroupList fused_groups;
    GroupPtr master_fuesd_group(nullptr);
    for (auto& consumer : fusionable_consumers) {
      auto fused_group = std::make_shared<ir::Group>();
      // update depth using consumer depth.
      fused_group->max_depth =
          std::max(producer->max_depth, consumer->max_depth);
      fused_group->min_depth =
          std::min(producer->min_depth, consumer->min_depth);
      // update group id
      fused_group->group_id = producer->group_id + "_" + consumer->group_id;
      VLOG(3) << "fuse producer " << producer->group_id << " into consumer "
              << consumer->group_id;
      // fuse producer into fusion group
      fused_group->op_pattern_kind =
          static_cast<int>(producer->op_pattern_kind) >=
                  static_cast<int>(consumer->op_pattern_kind)
              ? producer->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      fused_group->input_nodes = producer->input_nodes;

      // internal nodes
      if (producer->fused_sub_groups.size()) {
        for (auto& node : producer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // convert producer's output node to internal.
      for (auto node : producer->output_nodes) {
        // if node is used more than 1 time.
        if (consumer->input_nodes.count(node)) {
          if (consumer->input_nodes[node] > 1 && node->num_operands() > 0) {
            fused_group->internal_nodes.insert(node);
          }
        }
      }
      // master nodes
      for (auto& node : producer->master_nodes) {
        if (GetOpKind(node->name()) == kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }

      // producer groups
      for (auto& group : *producer->mut_producer_groups()) {
        fused_group->mut_producer_groups()->insert(group);
        // update producer's producer's consumer
        group->mut_consumer_groups()->erase(producer);
        group->mut_consumer_groups()->insert(fused_group);
      }

      // sub groups
      if (producer->fused_sub_groups.size()) {
        for (auto& group : producer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(group);
          // update belong group
          group->belong_groups.erase(producer);
          group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(producer);
      }
      producer->belong_groups.insert(fused_group);

      // input nodes
      for (auto& input_node : consumer->input_nodes) {
        // if input node not in producer output.
        if (!producer->output_nodes.count(input_node.first)) {
          if (fused_group->input_nodes.count(input_node.first)) {
            fused_group->input_nodes[input_node.first] += input_node.second;
          } else {
            fused_group->input_nodes.insert(input_node);
          }
        }
      }

      // output nodes
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }

      // internal nodes
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }

      // master nodes
      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }

      // producer nodes
      for (auto& group : *consumer->mut_producer_groups()) {
        if (group.get() != producer.get()) {
          fused_group->mut_producer_groups()->insert(group);
          // update consumer's producer's consumer
          group->mut_consumer_groups()->erase(consumer);
          group->mut_consumer_groups()->insert(fused_group);
        }
      }

      // consumer nodes
      for (auto& group : *consumer->mut_consumer_groups()) {
        fused_group->mut_consumer_groups()->insert(group);
        // update consumer's consumer's producer
        group->mut_producer_groups()->erase(consumer);
        group->mut_producer_groups()->insert(fused_group);
      }

      // sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          if (std::find(fused_group->fused_sub_groups.begin(),
                        fused_group->fused_sub_groups.end(),
                        sub_group) == fused_group->fused_sub_groups.end()) {
            fused_group->fused_sub_groups.push_back(sub_group);
          }
          // update belong group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      consumer->belong_groups.insert(fused_group);

      fused_groups.push_back(fused_group);
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id
          << " index in fusion_groups_index_!";
      auto postion = fusion_groups_index_[consumer];
      fusion_groups_[postion] = fused_group;
      fusion_groups_index_[fused_group] = postion;

      if (!master_fuesd_group.get()) {
        master_fuesd_group = fused_group;
      }
      CHECK(fused_group->output_nodes.size())
          << "No output node is found, " << fused_group->group_id;
    }

    for (auto& node : producer->output_nodes) {
      bool be_output = true;
      for (const auto& consumer : producer->consumer_groups()) {
        // if consumer is in fusionable.
        if (fusionable_consumers.count(consumer)) {
          if (consumer->input_nodes.count(node)) {
            be_output = false;
          }
          continue;
        }
        // if consumer is not in fusionable.
        if (consumer->input_nodes.count(node)) {
          be_output = true;
          break;
        }
        // others node is as graph output.
      }

      if (output_nodes_set_.count(node)) {
        be_output = true;
      }

      if (be_output) {
        // VLOG(4) << "Insert Id " << node->id() << " Into Group "
        //         << master_fuesd_group->group_id;
        master_fuesd_group->output_nodes.insert(node);
      }
    }
    // insert unfusionable consumer groups
    for (auto& consumer : *producer->mut_consumer_groups()) {
      if (fusionable_consumers.count(consumer)) {
        continue;
      }
      master_fuesd_group->mut_consumer_groups()->insert(consumer);
      // update consumer's producer
      consumer->mut_producer_groups()->erase(producer);
      consumer->mut_producer_groups()->insert(master_fuesd_group);
    }
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawRecomputeFusePasses()
      const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode(
        "RecomputeFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>&
  GetRecomputeFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>>
        fuse_passes = RawRecomputeFusePasses();
    return fuse_passes;
  }

  void TagRecomputeGroups(LightwareFusePassCtx* ctx) const {
    const auto& fuse_passes = GetRecomputeFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralRecomputeFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralRecomputeFuse handling producer : "
            << producer->group_id;
    using GroupSets = std::set<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& MarkFusible = [&](const OpGroupPtr& first,
                                    const OpGroupPtr& second) {
        tagged_sets.insert(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(ir::OpGroup(producer),
                                              MarkFusible);
      TagRecomputeGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet =
        [&]() -> std::unordered_set<GroupPtr, Hasher, Comparator> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr, Hasher, Comparator> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(group_pair.second.GetGroup());
      }
      return ret;
    };

    bool update = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size() > 0) {
      CHECK(consumer_groups.size() == producer->mut_consumer_groups()->size())
          << "Recompute requires fuse all consumers!";
      RecomputeFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void RecomputeFuse(const GroupPtr& producer,
                     const std::unordered_set<GroupPtr, Hasher, Comparator>&
                         fusionable_consumers) {
    VerticalFuse(producer, fusionable_consumers);
  }

  void SelectConsumerToFuse(
      const GroupPtr& producer,
      std::unordered_set<GroupPtr, Hasher, Comparator>* fusionable_consumers) {
    // if is const op

    // TODO(phlrain) : support constant
    // if (is_const_group(this, producer)) {
    if (false) {
      std::unordered_set<GroupPtr, Hasher, Comparator> candidates;
      for (auto& consumer : *fusionable_consumers) {
        // if can be output node.
        if (is_same_shape(producer, consumer)) {
          candidates.insert(consumer);
        } else {
          VLOG(4) << "Fuse Producer : " << producer->group_id
                  << " into Consumer : " << consumer->group_id;
          consumer->group_id = producer->group_id + "_" + consumer->group_id;
          // just merge the node into group.
          auto& sub_group = consumer->fused_sub_groups.front();
          sub_group->group_id = producer->group_id + "_" + sub_group->group_id;
          sub_group->nodes.insert(sub_group->nodes.begin(),
                                  producer->CollectNodes()[0]);
          sub_group->nodes_set.insert(producer->CollectNodes()[0]);
          // remove depency.
          consumer->input_nodes.erase(producer->CollectNodes()[0]);
          consumer->mut_producer_groups()->erase(producer);
          producer->mut_consumer_groups()->erase(consumer);
        }
      }

      CHECK_GE(producer->consumer_groups().size(), candidates.size());
      if (producer->consumer_groups().size() == 0 && candidates.size() == 0 &&
          output_nodes_set_.count(producer->CollectNodes()[0]) == 0) {
        producer->belong_groups.insert(*fusionable_consumers->begin());
      }

      *fusionable_consumers = candidates;
      return;
    }
    // 1 to 1 fusion.
    if (producer->consumer_groups().size() == 1) {
      return;
    }

    // TODO(phlrain): support flags
    // if (FLAGS_enhance_vertical_fusion_with_recompute) {
    if (false) {
      std::vector<GroupPtr> candidates;
      for (auto& consumer : *fusionable_consumers) {
        if (consumer->op_pattern_kind == kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto producer_output_shape = phi::vectorize(
            GetValueShape((*producer->output_nodes.begin())->result(0)));

        auto consumer_output_shape = phi::vectorize(
            GetValueShape((*consumer->output_nodes.begin())->result(0)));

        auto consumer_master_input_shape = phi::vectorize(GetValueShape(
            (*(consumer->master_nodes.begin()))->operand_source(0)));

        int producer_output_numel =
            std::accumulate(producer_output_shape.begin(),
                            producer_output_shape.end(),
                            1,
                            std::multiplies<int>());
        int consumer_output_numel =
            std::accumulate(consumer_output_shape.begin(),
                            consumer_output_shape.end(),
                            1,
                            std::multiplies<int>());
        int consumer_master_input_numel =
            std::accumulate(consumer_master_input_shape.begin(),
                            consumer_master_input_shape.end(),
                            1,
                            std::multiplies<int>());
        if (producer_output_numel == consumer_output_numel) {
          candidates.push_back(consumer);
          continue;
        }

        if (producer->op_pattern_kind != kInjective &&
            consumer->op_pattern_kind == kReduction &&
            producer_output_numel == consumer_master_input_numel) {
          candidates.push_back(consumer);
        }
      }
      sort(candidates.begin(),
           candidates.end(),
           [](const auto& lhs, const auto& rhs) {
             return lhs->op_pattern_kind < rhs->op_pattern_kind;
           });

      fusionable_consumers->clear();
      if (candidates.size()) {
        fusionable_consumers->insert(*candidates.begin());
      }
    } else {
      std::vector<GroupPtr> candidates;
      for (auto& consumer : *fusionable_consumers) {
        if (consumer->op_pattern_kind == kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto shape0 = phi::vectorize(
            GetValueShape((*producer->output_nodes.begin())->result(0)));
        auto shape1 = phi::vectorize(
            GetValueShape((*consumer->output_nodes.begin())->result(0)));

        if (std::accumulate(
                shape0.begin(), shape0.end(), 1, std::multiplies<int>()) ==
            std::accumulate(
                shape1.begin(), shape1.end(), 1, std::multiplies<int>())) {
          candidates.push_back(consumer);
        }
      }

      fusionable_consumers->clear();
      if (candidates.size()) {
        fusionable_consumers->insert(candidates.front());
      }
    }
  }

  bool IsDependency(
      const GroupPtr& producer_g,
      const GroupPtr& consumer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);

    std::unordered_set<GroupPtr, Hasher, Comparator> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (const auto& producer_and_list : candidate->producer_groups()) {
        if (producer_and_list.get() == producer_g.get()) {
          continue;
        }
        const auto& producer =
            std::dynamic_pointer_cast<ir::Group>(producer_and_list);
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool IsDependencySimplify(
      const GroupPtr& producer_g,
      const GroupPtr& consumer,
      const std::unordered_set<GroupPtr, Hasher, Comparator>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);
    // check upper.
    int check_upper_depth = producer_g.get() ? producer_g->max_depth : INT_MAX;
    std::unordered_set<GroupPtr, Hasher, Comparator> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (auto& producer_and_list : candidate->producer_groups()) {
        if (producer_and_list.get() == producer_g.get()) {
          continue;
        }
        const auto& producer =
            std::dynamic_pointer_cast<ir::Group>(producer_and_list);
        if (producer->min_depth > check_upper_depth) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool GeneralInputFuse() {
    VLOG(3) << "GeneralInputFuse...!";
    auto updated = false;
    UpdateInputToConsumers();
    for (auto& input_consumers : input_to_consumers_) {
      // if group set size == 1.
      if (input_consumers.second.size() == 1) {
        continue;
      }
      // do input fusion.
      auto st = CallGeneralInputFusePass(input_consumers.second);
      if (st) {
        // fused consumers, update
        UpdateInputToConsumers();
      }
      updated |= st;
    }

    return updated;
  }

  void UpdateInputToConsumers() {
    for (auto& input_consumers : input_to_consumers_) {
      auto& consumers = input_consumers.second;
      std::unordered_set<GroupPtr, Hasher, Comparator> updated_consumers;
      for (auto& consumer : consumers) {
        std::queue<GroupPtr> fused_groups;
        fused_groups.push(consumer);
        while (!fused_groups.empty()) {
          auto& cur = fused_groups.front();
          fused_groups.pop();
          // if group is sub group
          if (cur->belong_groups.empty()) {
            updated_consumers.insert(cur);
          } else {
            for (auto& belong_group : cur->belong_groups) {
              if (belong_group->group_id == cur->group_id) {
                updated_consumers.insert(belong_group);
              } else {
                fused_groups.push(belong_group);
              }
            }
          }
        }
      }
      consumers = updated_consumers;
    }
  }

  void InitInputToConsumers() {
    VLOG(3) << "InitInputToConsumers...!";
    // init input data node -> fusion group map.
    for (auto& group : fusion_groups_) {
      for (auto& node : group->nodes_set) {
        // collect producer node data.
        for (size_t i = 0; i < node->num_operands(); ++i) {
          auto in = node->operand_source(i);
          if (in) {
            input_to_consumers_[in].insert(group);
          }
        }
      }
    }
  }

  void InitFusionGroupsAndIndex() {
    VLOG(3) << "InitFusionGroupsAndIndex...!";
    // init the postion of groups in fusion groups.
    for (size_t idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto group = fusion_groups_[idx];
      auto belong_group = std::make_shared<ir::Group>();
      // copy from group.
      belong_group->max_depth = group->depth;
      belong_group->min_depth = group->depth;
      belong_group->group_id = group->group_id;
      belong_group->input_nodes = group->input_nodes;
      belong_group->output_nodes = group->output_nodes;
      belong_group->op_pattern_kind = group->op_pattern_kind;
      belong_group->master_nodes = group->master_nodes;
      (*belong_group->mut_producer_groups()) = group->producer_groups();
      (*belong_group->mut_consumer_groups()) = group->consumer_groups();
      belong_group->fused_sub_groups.push_back(group);
      group->belong_groups.insert(belong_group);
      // replace group to fused_group
      fusion_groups_[idx] = belong_group;
      // record idx
      fusion_groups_index_[belong_group] = idx;
    }

    // update producer and consumer.
    for (auto& group : fusion_groups_) {
      std::unordered_set<GroupPtr, Hasher, Comparator> producers;
      std::unordered_set<GroupPtr, Hasher, Comparator> consumers;

      for (const auto& producer : group->producer_groups()) {
        CHECK(producer->belong_groups.size());
        producers.insert(*producer->belong_groups.begin());
      }

      for (auto& consumer : *group->mut_consumer_groups()) {
        CHECK(consumer->belong_groups.size());
        consumers.insert(*consumer->belong_groups.begin());
      }
      CHECK_EQ(group->producer_groups().size(), producers.size());
      CHECK_EQ(group->consumer_groups().size(), consumers.size());
      (*group->mut_producer_groups()) = producers;
      (*group->mut_consumer_groups()) = consumers;
    }
  }

  GroupList fusion_groups_;
  std::unordered_map<GroupPtr, int> fusion_groups_index_;
  std::unordered_set<const ::pir::Operation*> output_nodes_set_;
  std::unordered_map<::pir::Value,
                     std::unordered_set<GroupPtr, Hasher, Comparator>>
      input_to_consumers_;
};

GroupList GeneralFusionMergePassInternal(const GroupList& group_list) {
  if (group_list.size() <= 1) {
    VLOG(3) << "Don't do Fusoin Merge Pass...!";
    return group_list;
  }

  GeneralFusionMergePassHelper fusion_merge_pass_helper(group_list);
  auto res = fusion_merge_pass_helper();

  return res;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
