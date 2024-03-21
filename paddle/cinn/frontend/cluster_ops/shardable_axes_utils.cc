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

#include "paddle/cinn/frontend/cluster_ops/shardable_axes_utils.h"

namespace cinn::frontend::cluster_ops {

OldName2NewName GetOldName2NewName(const ShardableAxes& old_sa,
                                   const ShardableAxes& new_sa) {
  OldName2NewName old_name2new_name;
  for (const auto& [old_axis, old_name] : old_sa) {
    for (const auto& [new_axis, new_name] : new_sa) {
      if (old_axis == new_axis) {
        CHECK(old_name2new_name.emplace(old_name, new_name).second);
      }
    }
  }
  return old_name2new_name;
}

void UpdateShardableAxes(const OldName2NewName& old2new, ShardableAxes* sa) {
  for (auto iter = sa->begin(); iter != sa->end();) {
    const auto& pair_it = old2new.find(iter->axis_name);
    if (pair_it != old2new.end()) {
      iter->axis_name = pair_it->second;
      ++iter;
    } else {
      iter = sa->erase(iter);
    }
  }
}

ShardableAxes GetCommonShardableAxes(const ShardableAxes& lhs,
                                     const ShardableAxes& rhs) {
  ShardableAxes ret;
  for (const auto& lhs_axis : lhs) {
    for (const auto& rhs_axis : rhs) {
      if (lhs_axis == rhs_axis) {
        ret.emplace_back(lhs_axis);
      }
    }
  }
  return ret;
}

ShardableAxes MakeFullyShardableAxes(const size_t rank) {
  ShardableAxes ret;
  for (int i = 0; i < rank; ++i) {
    ret.emplace_back(ShardableAxis{
        .axis = i,
        .axis_name =
            std::string("D") + std::to_string(ShardableAxis::UnqiueSeqNo()),
    });
  }
  return ret;
}

ShardableAxes MakeReduceOpInputShardableAxes(
    const size_t input_rank, const std::vector<int64_t>& reduce_axes) {
  if (reduce_axes.empty()) return ShardableAxes{};
  for (int64_t reduce_axis : reduce_axes) {
    CHECK_GE(reduce_axis, 0);
    CHECK_LT(reduce_axis, input_rank);
  }
  const auto IsReduceAxis = [&](int64_t i) {
    return std::find(reduce_axes.begin(), reduce_axes.end(), i) !=
           reduce_axes.end();
  };
  ShardableAxes ret;
  for (int64_t i = 0; i < input_rank; ++i) {
    if (IsReduceAxis(i)) continue;
    ret.emplace_back(ShardableAxis{
        .axis = static_cast<int>(i),
        .axis_name =
            std::string("D") + std::to_string(ShardableAxis::UnqiueSeqNo()),
    });
  }
  return ret;
}

ShardableAxes MakeBroadcastOpInputShardableAxes(
    const size_t input_rank, const std::vector<int64_t>& broadcast_axes) {
  for (int64_t axis : broadcast_axes) {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
  }
  const auto IsBroadcastAxis = [&](int64_t i) {
    return std::find(broadcast_axes.begin(), broadcast_axes.end(), i) !=
           broadcast_axes.end();
  };
  ShardableAxes ret;
  for (int64_t i = 0; i < input_rank; ++i) {
    if (IsBroadcastAxis(i)) continue;
    ret.emplace_back(ShardableAxis{
        .axis = static_cast<int>(i),
        .axis_name =
            std::string("D") + std::to_string(ShardableAxis::UnqiueSeqNo()),
    });
  }
  return ret;
}

}  // namespace cinn::frontend::cluster_ops
