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

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/utils.h"

// This file is the protocol of the pattern fuser. Please implement
// ConvertToStmtPattern and MergePatternImpl in the specializations.

namespace cinn::fusion {

static StmtPattern ConvertToStmtPattern(const PatternContent& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    auto result =
        ReducePattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    auto result = TrivialPattern(
        {content.op}, content.op, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Unsupported op for fusion: %s", OpsDebugStr({content.op})));
    auto result =
        UnsupportedPattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  }
}

// Trivial x other

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const TrivialPattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  auto result = TrivialPattern(
      contents,
      second.sink_op(),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.set_loop_axis_mapping(TrivialSinkLoopAxisMappingMerge(
      first.loop_axis_mapping(), second.loop_axis_mapping()));
  return result;
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const ReducePattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  auto result = ReducePattern(
      contents,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.set_loop_axis_mapping(TrivialSinkLoopAxisMappingMerge(
      first.loop_axis_mapping(), second.loop_axis_mapping()));
  return result;
}

template <typename A, typename B>
B FusePatternIfConnected(A up_pattern,
                         B down_pattern,
                         std::vector<pir::Operation*> connect_ops) {
  if (AnyFirstInSecond(connect_ops, down_pattern.ops())) {
    return std::get<B>(MergePatternImpl(up_pattern, down_pattern));
  } else {
    return down_pattern;
  }
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const ReduceTreePattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());

  auto old_children = second.children();
  std::vector<ReduceTreePattern> new_children;
  for (const auto& old_child : old_children) {
    new_children.emplace_back(
        FusePatternIfConnected(first, old_child, connect_ops));
  }
  auto result = ReduceTreePattern(
      new_children,
      FusePatternIfConnected(first, second.GetRootPattern(), connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.set_loop_axis_mapping(TrivialSinkLoopAxisMappingMerge(
      first.loop_axis_mapping(), second.loop_axis_mapping()));
  return result;
}

static StmtPattern MergePatternImpl(
    const TrivialPattern& first, const ReduceTreePlusTrivialPattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());
  auto result = ReduceTreePlusTrivialPattern(
      FusePatternIfConnected(first, second.tree, connect_ops),
      FusePatternIfConnected(first, second.sink_trivial, connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.fake_reduce_iter_idx = second.fake_reduce_iter_idx;
  result.set_loop_axis_mapping(TrivialSinkLoopAxisMappingMerge(
      first.loop_axis_mapping(), second.loop_axis_mapping()));
  return result;
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const AnchorPattern& second) {
  return AnchorPattern(
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second)),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_),
      TrivialSinkLoopAxisMappingMerge(first.loop_axis_mapping(),
                                      second.loop_axis_mapping()));
}

// RR & RT

static int InsertUpstreamIntoTree(const ReduceTreePattern& upstream,
                                  ReduceTreePattern& downstream) {  // NOLINT
  auto is_direct_upstream = [&](const ReducePattern& upstream,
                                const ReducePattern& downstream) -> bool {
    auto upstream_result = upstream.GetReduceOp()->result(0);
    auto user_ops = FindUserOp(downstream.ops(), upstream_result);
    return !user_ops.empty();
  };

  if (is_direct_upstream(upstream.GetRootPattern(),
                         downstream.GetRootPattern())) {
    downstream.InsertChild(upstream);
    return 1;
  }
  int insert_num = 0;
  for (auto& child : downstream.children()) {
    insert_num += InsertUpstreamIntoTree(upstream, child);
  }
  return insert_num;
}

static StmtPattern MergePatternImpl(const ReduceTreePattern& upstream,
                                    const ReduceTreePattern& downstream) {
  ReduceTreePattern result = ReduceTreePattern(
      downstream.children(),
      downstream.GetRootPattern(),
      std::make_shared<FusionTracker>(upstream.tracker_,
                                      downstream.tracker_));  // copy first.
  int insert_num = InsertUpstreamIntoTree(upstream, result);
  result.set_loop_axis_mapping(LoopAxisMappingMerge(
      upstream.loop_axis_mapping(), downstream.loop_axis_mapping(), false));
  PADDLE_ENFORCE_EQ(insert_num,
                    1,
                    ::common::errors::PreconditionNotMet(
                        "Must insert only once, but insert %d", insert_num));
  return result;
}

static StmtPattern MergePatternImpl(const ReduceTreePattern& first,
                                    const TrivialPattern& second) {
  auto result = ReduceTreePlusTrivialPattern(
      first,
      second,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.set_loop_axis_mapping(ReducePlusTrivialLoopAxisMappingMerge(
      first.loop_axis_mapping(), second.loop_axis_mapping()));
  return result;
}

static StmtPattern MergePattern(const StmtPattern& first,
                                const StmtPattern& second) {
  VLOG(4) << "MergePattern: " << GetPatternId(first) << " x "
          << GetPatternId(second);
  const auto PatternMatch = adt::match{
      [&](const ReduceTreePattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const ReduceTreePattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReducePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePlusTrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const AnchorPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const auto& lhs, const auto& rhs) -> StmtPattern {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Not support for MergePatternImpl"));
      },
  };
  return std::visit(PatternMatch, first, second);
}

static void SetReturnInstr(const StmtPattern& s) {
  std::visit(
      [](const auto& impl) {
        impl.tracker_->append(std::make_shared<ReturnInstr>(impl.id()));
      },
      s);
}

}  // namespace cinn::fusion
