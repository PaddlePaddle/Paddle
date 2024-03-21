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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/frontend/cluster_ops/common_utils.h"

namespace cinn::frontend::cluster_ops {

struct OpAndOperandIndex {
  const pir::Operation* op;
  const int operand_index;

  bool operator==(const OpAndOperandIndex& other) const {
    return this->op == other.op && this->operand_index == other.operand_index;
  }
};

}  // namespace cinn::frontend::cluster_ops
namespace std {

template <>
struct hash<cinn::frontend::cluster_ops::OpAndOperandIndex> {
  size_t operator()(
      const cinn::frontend::cluster_ops::OpAndOperandIndex& op_operand) const {
    return cinn::adt::hash_combine(
        std::hash<const pir::Operation*>()(op_operand.op),
        op_operand.operand_index);
  }
};

}  // namespace std

namespace cinn::frontend::cluster_ops {

struct ShardableAxis {
  int axis;
  std::string axis_name;

  bool operator==(const ShardableAxis& other) const {
    return this->axis == other.axis && this->axis_name == other.axis_name;
  }

  static int64_t UnqiueSeqNo() {
    static std::atomic<int64_t> cnt(0);
    return ++cnt;
  }
};

using ShardableAxes = std::vector<ShardableAxis>;
using ShardableAxes4ValueT =
    std::function<std::optional<const ShardableAxes*>(pir::Value)>;
using OldName2NewName = std::unordered_map<std::string, std::string>;

struct SoleOutputShardableAxes {
  ShardableAxes shardable_axes;
};

struct ShardableAxesSignature {
  SoleOutputShardableAxes sole_output_sa;
  std::unordered_map<OpAndOperandIndex, ShardableAxes> input_shardable_axes;
};

OldName2NewName GetOldName2NewName(const ShardableAxes& old_sa,
                                   const ShardableAxes& new_sa);

void UpdateShardableAxes(const OldName2NewName& old2new, ShardableAxes* sa);

ShardableAxes GetCommonShardableAxes(const ShardableAxes& lhs,
                                     const ShardableAxes& rhs);

ShardableAxes MakeFullyShardableAxes(const size_t rank);

ShardableAxes MakeReduceOpInputShardableAxes(
    const size_t input_rank, const std::vector<int64_t>& reduce_axes);

ShardableAxes MakeBroadcastOpInputShardableAxes(
    const size_t input_rank, const std::vector<int64_t>& broadcast_axes);

}  // namespace cinn::frontend::cluster_ops
