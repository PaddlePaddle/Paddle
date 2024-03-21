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
#include <optional>

namespace cinn::frontend::cluster_ops {

struct ShardableAxesUtil {
  using OldName2NewName = std::unordered_map<std::string, std::string>;

  static OldName2NewName GetOldName2NewName(const ShardableAxes& old_sa,
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

  static void UpdateShardableAxes(const OldName2NewName& old2new,
                                  ShardableAxes* sa) {
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

  static ShardableAxes GetCommonShardableAxes(const ShardableAxes& lhs,
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

  static ShardableAxes MakeFullyShardableAxes(const size_t rank) {
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

  static ShardableAxes MakeReduceOpInputShardableAxes(
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

  static ShardableAxes MakeBroadcastOpInputShardableAxes(
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
};

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
    ShardableAxes output_sa =
        ShardableAxesUtil::MakeFullyShardableAxes(GetRank(output));
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
        ShardableAxesUtil::MakeReduceOpInputShardableAxes(input_rank,
                                                          reduce_axes);
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
    const ShardableAxes output_shardable_axes =
        ShardableAxesUtil::MakeFullyShardableAxes(rank);
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
        ShardableAxesUtil::MakeBroadcastOpInputShardableAxes(input_rank,
                                                             broadcast_axes);
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

/*====================== ShardableAxesInferer Methods ======================*/

ShardableAxesSignature ShardableAxesInferer::MakeShardableAxesSignature4Op(
    const pir::Operation* op) {
  return shardable_axes_provider_->MakeShardableAxesSignature4Op(op);
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::InferShardableAxesFromSink(const pir::Operation* sink,
                                                 const OpTopo& op_topo) {
  auto reversed_walker = GetOpsReversedTopoWalker(op_topo);
  CHECK_GT(op_topo.ops->count(sink), 0);
  const int result_idx = GetOutputShardableAxesResultIdx(sink);
  size_t rank = GetRank(sink->result(result_idx));
  const auto& init_sa = ShardableAxesUtil::MakeFullyShardableAxes(rank);
  return ReversedInferShardableAxes(reversed_walker, sink, init_sa);
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::InferShardableAxes(const OpSetPtr& ops) {
  VLOG(4) << "InferShardableAxes";
  auto reversed_walker = GetOpsReversedTopoWalker(OpTopo{
      .ops = ops,
  });
  const auto& sinks = GetSinks(*ops);
  const auto& sink_and_init_value =
      GetSinkAndInitValues(reversed_walker, ops, sinks);
  return ReversedInferShardableAxes(
      reversed_walker, sink_and_init_value.begin(), sink_and_init_value.end());
}

template <typename InputIt>
std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::ReversedInferShardableAxes(
    const common::TopoWalker<const pir::Operation*>& reversed_walker,
    InputIt sink_and_init_begin,
    InputIt sink_and_init_end) {
  std::unordered_map<pir::Value, ShardableAxes> value2shardable_axes;
  std::list<const pir::Operation*> sinks;
  for (auto iter = sink_and_init_begin; iter != sink_and_init_end; ++iter) {
    sinks.push_back(iter->first.defining_op());
    value2shardable_axes[iter->first] = iter->second;
  }
  const auto& UpdateValue2ShardableAxes = [&](pir::Value value,
                                              const ShardableAxes& sa) {
    auto iter = value2shardable_axes.find(value);
    if (iter != value2shardable_axes.end()) {
      iter->second =
          ShardableAxesUtil::GetCommonShardableAxes(iter->second, sa);
    } else {
      value2shardable_axes[value] = sa;
    }
  };
  reversed_walker(sinks.begin(), sinks.end(), [&](const auto* op) {
    auto shardable_axes_sig = MakeShardableAxesSignature4Op(op);
    const auto& sole_output_sa = shardable_axes_sig.sole_output_sa;
    const int result_idx = GetOutputShardableAxesResultIdx(op);
    const auto& old2new = ShardableAxesUtil::GetOldName2NewName(
        sole_output_sa.shardable_axes,
        value2shardable_axes.at(op->result(result_idx)));
    for (auto& pair : shardable_axes_sig.input_shardable_axes) {
      const auto& [my_op, input_idx] = pair.first;
      CHECK_EQ(my_op, op);
      auto* input_shardable_axes = &pair.second;
      ShardableAxesUtil::UpdateShardableAxes(old2new, input_shardable_axes);
      pir::Value input_value = op->operand_source(input_idx);
      UpdateValue2ShardableAxes(input_value, *input_shardable_axes);
    }
  });
  return value2shardable_axes;
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::ReversedInferShardableAxes(
    const common::TopoWalker<const pir::Operation*>& reversed_walker,
    const pir::Operation* sink,
    const ShardableAxes& init_sa) {
  using OpAndInitValue = std::pair<pir::Value, ShardableAxes>;
  const int result_idx = GetOutputShardableAxesResultIdx(sink);
  std::array<OpAndInitValue, 1> sinks{
      OpAndInitValue{sink->result(result_idx), init_sa}};
  return ReversedInferShardableAxes(
      reversed_walker, sinks.begin(), sinks.end());
}

std::unordered_map<const pir::Operation*, ShardableAxesSignature>
ShardableAxesInferer::GetOp2ShardableAxesSignature(const OpSetPtr& ops) {
  VLOG(4) << "GetOp2ShardableAxesSignature";
  std::unordered_map<const pir::Operation*, ShardableAxesSignature> ret;
  for (const auto* op : *ops) {
    ret[op] = MakeShardableAxesSignature4Op(op);
  }
  return ret;
}

std::map<std::string, std::vector<std::string>>
ShardableAxesInferer::GetAxisName2BoundAxisName(
    const OpSetPtr& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature) {
  const auto GetInputShardableAxes = [&](const OpAndOperandIndex& op_and_idx)
      -> std::optional<const ShardableAxes*> {
    const auto& [op, idx] = op_and_idx;
    const auto* input_op = op->operand_source(idx).defining_op();
    if (ops->count(input_op) == 0) return std::nullopt;
    const auto& iter = op2shardable_axes_signature.find(input_op);
    if (iter == op2shardable_axes_signature.end()) return std::nullopt;
    const auto& output_sa = iter->second.sole_output_sa.shardable_axes;
    return &output_sa;
  };
  std::map<std::string, std::vector<std::string>> axis_name2bound_axis_name;
  const auto UpdateAxisName2BoundAxisName = [&](const ShardableAxes& input_sa,
                                                const ShardableAxes& sa) {
    for (const auto& [input_axis, input_axis_name] : input_sa) {
      for (const auto& [axis, axis_name] : sa) {
        if (input_axis != axis) continue;
        axis_name2bound_axis_name[axis_name].push_back(input_axis_name);
        axis_name2bound_axis_name[input_axis_name].push_back(axis_name);
      }
    }
  };
  for (const auto& [op, signature] : op2shardable_axes_signature) {
    for (const auto& [op_and_idx, sa] : signature.input_shardable_axes) {
      const auto& input_sa = GetInputShardableAxes(op_and_idx);
      if (!input_sa.has_value()) continue;
      UpdateAxisName2BoundAxisName(*input_sa.value(), sa);
    }
  }
  return axis_name2bound_axis_name;
}

std::unordered_map<std::string, std::string>
ShardableAxesInferer::GetAxisName2UnionFindSetRoot(
    const OpSetPtr& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature) {
  VLOG(4) << "GetAxisName2UnionFindSetRoot";
  const auto axis_name2bound_axis_name =
      GetAxisName2BoundAxisName(ops, op2shardable_axes_signature);
  using NodeVisitor = std::function<void(const std::string&)>;
  const auto VisitNext = [&](const std::string& axis_name,
                             const NodeVisitor& DoEach) {
    const auto& iter = axis_name2bound_axis_name.find(axis_name);
    if (iter == axis_name2bound_axis_name.end()) return;
    for (const auto& input_axis_name : iter->second) {
      DoEach(input_axis_name);
    }
  };
  common::BfsWalker<std::string> walk(VisitNext);
  std::unordered_map<std::string, std::string> axis_name2root;
  for (const auto& [union_find_root, _] : axis_name2bound_axis_name) {
    if (axis_name2root.count(union_find_root) > 0) continue;
    walk(union_find_root, [&](const std::string& axis_name) {
      CHECK(axis_name2root.emplace(axis_name, union_find_root).second);
    });
  }
  return axis_name2root;
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::GetSinkAndInitShardableAxes(
    const std::list<const pir::Operation*>& sinks,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature,
    const std::unordered_map<std::string, std::string>&
        axis_name2union_find_set_root) {
  VLOG(4) << "GetSinkAndInitShardableAxes";
  const auto& ConvertByBoundAxisName = [&](const ShardableAxes& sa) {
    ShardableAxes ret_sa;
    for (const auto& [axis, axis_name] : sa) {
      const auto& iter = axis_name2union_find_set_root.find(axis_name);
      CHECK(iter != axis_name2union_find_set_root.end());
      ret_sa.emplace_back(ShardableAxis{
          .axis = axis,
          .axis_name = iter->second,
      });
    }
    return ret_sa;
  };
  std::unordered_map<pir::Value, ShardableAxes> sink2sa;
  for (const auto* sink : sinks) {
    const auto& sig_iter = op2shardable_axes_signature.find(sink);
    CHECK(sig_iter != op2shardable_axes_signature.end());
    const auto& sole_output_sa = sig_iter->second.sole_output_sa;
    const auto& output_shardable_axes = sole_output_sa.shardable_axes;
    const int result_idx = GetOutputShardableAxesResultIdx(sink);
    sink2sa[sink->result(result_idx)] =
        ConvertByBoundAxisName(output_shardable_axes);
  }
  return sink2sa;
}

void ShardableAxesInferer::RenameDuplicatedAxisName(
    std::unordered_map<pir::Value, ShardableAxes>* sink2sa) {
  VLOG(4) << "RenameDuplicatedAxisName";
  const auto& RenameDuplicated = [&](ShardableAxes* sa) {
    std::set<std::string> existed_axis_name;
    for (auto& [_, axis_name] : *sa) {
      if (!existed_axis_name.emplace(axis_name).second) {
        axis_name =
            axis_name + "_" + std::to_string(ShardableAxis::UnqiueSeqNo());
      } else {
        // do nothing.
      }
    }
  };
  for (auto& [_, sa] : *sink2sa) {
    RenameDuplicated(&sa);
  }
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::GetSinkAndInitValues(
    const common::TopoWalker<const pir::Operation*>& reverse_walker,
    const OpSetPtr& ops,
    const std::list<const pir::Operation*>& sinks) {
  VLOG(4) << "GetSinkAndInitValues";
  const auto& op2shardable_axes_signature = GetOp2ShardableAxesSignature(ops);
  const auto& axis_name2union_find_set_root =
      GetAxisName2UnionFindSetRoot(ops, op2shardable_axes_signature);
  std::unordered_map<pir::Value, ShardableAxes> sink_and_inits =
      GetSinkAndInitShardableAxes(
          sinks, op2shardable_axes_signature, axis_name2union_find_set_root);
  RenameDuplicatedAxisName(&sink_and_inits);
  return sink_and_inits;
}

}  // namespace cinn::frontend::cluster_ops
