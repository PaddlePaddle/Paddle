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

#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"

namespace cinn::frontend::cluster_ops {

class DefaultShardableAxesProvider final : public ShardableAxesProvider {
 private:
  const pir::ShapeConstraintIRAnalysis* shape_analysis_;

 public:
  explicit DefaultShardableAxesProvider(
      const pir::ShapeConstraintIRAnalysis* shape_analysis)
      : shape_analysis_(shape_analysis) {}

  ShardableAxesSignature MakeShardableAxesSignature4Op(
      const pir::Operation* op) override {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (kind == hlir::framework::kReduction) {
      return MakeShardableAxesSignature4ReduceOp(op);
    } else if (kind == hlir::framework::kElementWise) {
      return MakeShardableAxesSignature4ElementWiseOp(op);
    } else if (kind == hlir::framework::kBroadcast) {
      return MakeShardableAxesSignature4BroadcastOp(op);
    } else {
      LOG(ERROR) << "[ShardableAxesSignature] no shardable axes signature "
                    "found. op_name:"
                 << op->name();
    }
    return MakeEmptyShardableAxesSignature(op);
  }

 private:
  ShardableAxes SequeezeShardableAxes(const ShardableAxes& sa) {
    ShardableAxes ret_sa(sa);
    for (int i = 0; i < ret_sa.size(); ++i) {
      for (int j = i + 1; j < ret_sa.size(); ++j) {
        CHECK_LT(ret_sa.at(i).axis, ret_sa.at(j).axis);
      }
      ret_sa.at(i).axis = i;
    }
    return ret_sa;
  }

  using InputSignature = std::unordered_map<OpAndOperandIndex, ShardableAxes>;

  ShardableAxesSignature MakeEmptyShardableAxesSignature(
      const pir::Operation* op) {
    const int result_idx = GetOutputShardableAxesResultIdx(op);
    pir::Value output = op->result(result_idx);
    ShardableAxes output_sa = MakeFullyShardableAxes(GetRank(output));
    InputSignature empty_input_sig;
    for (int i = 0; i < op->num_operands(); ++i) {
      empty_input_sig[OpAndOperandIndex{op, i}] = ShardableAxes{};
    }
    return ShardableAxesSignature{
        .sole_output_sa =
            SoleOutputShardableAxes{
                .shardable_axes = output_sa,
            },
        .input_shardable_axes = empty_input_sig,
    };
  }

  ShardableAxesSignature MakeShardableAxesSignature4ReduceOp(
      const pir::Operation* reduce_op) {
    const size_t input_rank = GetRank(reduce_op->operand_source(0));
    const auto& reduce_axes = GetReduceAxes(reduce_op);
    const ShardableAxes input_sa =
        MakeReduceOpInputShardableAxes(input_rank, reduce_axes);
    using InputSignature = std::unordered_map<OpAndOperandIndex, ShardableAxes>;
    const ShardableAxes output_sa =
        (GetReduceOpKeepDims(reduce_op) ? input_sa
                                        : SequeezeShardableAxes(input_sa));
    return ShardableAxesSignature{
        .sole_output_sa =
            SoleOutputShardableAxes{
                .shardable_axes = output_sa,
            },
        .input_shardable_axes =
            InputSignature{
                {OpAndOperandIndex{reduce_op, 0}, input_sa},
            },
    };
  }

  bool IsDisabledElementwiseOp(const pir::Operation* op) {
    if (op->isa<cinn::dialect::ReshapeOp>()) return true;
    return false;
  }

  ShardableAxesSignature MakeShardableAxesSignature4ElementWiseOp(
      const pir::Operation* op) {
    if (IsDisabledElementwiseOp(op)) {
      LOG(ERROR) << "[ShardableAxesSignature] no shardable axes signature "
                    "found. op_name : "
                 << op->name();
      return MakeEmptyShardableAxesSignature(op);
    }
    const size_t rank = [&] {
      std::optional<size_t> rank;
      for (int i = 0; i < op->num_operands(); ++i) {
        if (rank.has_value()) {
          CHECK_EQ(rank.value(), GetRank(op->operand_source(i)));
        } else {
          rank = GetRank(op->operand_source(i));
        }
      }
      const int result_idx = GetOutputShardableAxesResultIdx(op);
      if (rank.has_value()) {
        CHECK_EQ(rank.value(), GetRank(op->result(result_idx)));
      } else {
        rank = GetRank(op->result(result_idx));
      }
      CHECK(rank.has_value());
      return rank.value();
    }();
    const ShardableAxes output_shardable_axes = MakeFullyShardableAxes(rank);
    std::unordered_map<OpAndOperandIndex, ShardableAxes> input_shardable_axes;
    for (int i = 0; i < op->num_operands(); ++i) {
      input_shardable_axes[OpAndOperandIndex{op, i}] = output_shardable_axes;
    }
    return ShardableAxesSignature{
        .sole_output_sa =
            SoleOutputShardableAxes{
                .shardable_axes = output_shardable_axes,
            },
        .input_shardable_axes = input_shardable_axes,
    };
  }

  std::optional<std::tuple<pir::Value, /*input_dix*/ int, pir::Value>>
  GetGetBroadcastOpInputOuputValue(const pir::Operation* op) {
    auto* mut_op = const_cast<pir::Operation*>(op);
    if (op->isa<paddle::dialect::ExpandOp>()) {
      auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
      return std::tuple{expand_op.x(), 0, expand_op.out()};
    }
    if (op->isa<cinn::dialect::BroadcastOp>()) {
      auto broadcast_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
      return std::tuple{broadcast_op.x(), 0, broadcast_op.out()};
    }
    return std::nullopt;
  }

  ShardableAxesSignature MakeShardableAxesSignature4BroadcastOp(
      const pir::Operation* op) {
    const auto& input_output_pair = GetGetBroadcastOpInputOuputValue(op);
    if (!input_output_pair.has_value()) {
      LOG(ERROR) << "[ShardableAxesSignature] no shardable axes signature "
                    "found. op_name : "
                 << op->name();
      return MakeEmptyShardableAxesSignature(op);
    }
    const auto& [input, input_idx, output] = input_output_pair.value();
    const int input_rank = GetRank(input);
    const int rank_diff = GetRank(output) - input_rank;
    CHECK_GE(rank_diff, 0);
    const auto& broadcast_axes = [&] {
      std::vector<int64_t> broadcast_axes;
      for (int i = 0; i < input_rank; ++i) {
        int o = i + rank_diff;
        if (!shape_analysis_->IsProductEqual(input, {i}, output, {o})) {
          broadcast_axes.push_back(i);
        }
      }
      return broadcast_axes;
    }();
    const ShardableAxes input_sa =
        MakeBroadcastOpInputShardableAxes(input_rank, broadcast_axes);
    const ShardableAxes output_sa = [&] {
      ShardableAxes output_sa(input_sa);
      for (auto& shardable_axis : output_sa) {
        shardable_axis.axis += rank_diff;
      }
      return output_sa;
    }();
    return ShardableAxesSignature{
        .sole_output_sa =
            SoleOutputShardableAxes{
                .shardable_axes = output_sa,
            },
        .input_shardable_axes =
            InputSignature{
                {OpAndOperandIndex{op, input_idx}, input_sa},
            },
    };
  }
};

std::shared_ptr<ShardableAxesProvider> MakeDefaultShardableAxesProvider(
    const pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return std::make_shared<DefaultShardableAxesProvider>(shape_analysis);
}

int GetOutputShardableAxesResultIdx(const pir::Operation* op) { return 0; }

}  // namespace cinn::frontend::cluster_ops
