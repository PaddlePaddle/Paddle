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

#include "paddle/cinn/frontend/group_pattern_util.h"

#include <algorithm>
#include <optional>
#include <typeinfo>
#include <variant>

#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn::frontend {

namespace {
using OpPatternKind = cinn::hlir::framework::OpPatternKind;

using IS = api::InjectiveSourcePattern<frontend::FrontendPattern>;
using R = api::ReductionPattern<frontend::FrontendPattern>;
using PS = api::PartialShardablePattern<frontend::FrontendPattern>;
using StmtPattern = api::StmtPattern<frontend::FrontendPattern>;
using StmtsPattern = api::StmtsPattern<frontend::FrontendPattern>;
using OpSet = std::unordered_set<const pir::Operation*>;
using OpSetPtr = std::shared_ptr<OpSet>;

using OpVisitor = std::function<void(const pir::Operation*)>;
using StmtVisitor = std::function<void(const StmtPattern*)>;

struct OpTopo {
  OpSetPtr ops;

  static OpTopo Make(const std::vector<const pir::Operation*>& ops) {
    auto ops_set = std::make_shared<OpSet>(ops.begin(), ops.end());
    return OpTopo{
        .ops = ops_set,
    };
  }

  template <typename OpVisitorT>
  void VisitInputOp(const pir::Operation* op, const OpVisitorT& DoEach) const {
    if (this->ops->count(op) == 0) return;
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (this->ops->count(input_op) == 0) continue;
      DoEach(input_op);
    }
  }

  template <typename OpVisitorT>
  void VisitOutputOp(const pir::Operation* op, const OpVisitorT& DoEach) const {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin();
           consumer_it != output.use_end();
           ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (consumer_op->isa<pir::YieldOp>()) continue;
        if (this->ops->count(consumer_op) == 0) continue;
        DoEach(consumer_op);
      }
    }
  }
};

int GetOutputShardableAxesResultIdx(const pir::Operation* op) { return 0; }

OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

bool IsGeneralInjective(const pir::Operation* op) {
  hlir::framework::OpPatternKind op_pattern_kind = GetOpPatternKind(op);
  return op_pattern_kind == hlir::framework::kElementWise ||
         op_pattern_kind == hlir::framework::kBroadcast ||
         op_pattern_kind == hlir::framework::kInjective;
}

bool IsISPattern(const StmtPattern& pattern) {
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern) {
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern) {
  return std::holds_alternative<R>(pattern);
}

std::list<const pir::Operation*> GetSinks(const OpSet& ops) {
  const auto IsSink = [&](const pir::Operation* op) {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin();
           consumer_it != output.use_end();
           ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (consumer_op->isa<pir::YieldOp>()) continue;
        if (ops.count(consumer_op) > 0) return false;
      }
    }
    return true;
  };
  std::list<const pir::Operation*> sinks;
  for (const auto* op : ops) {
    if (IsSink(op)) {
      sinks.push_back(op);
    }
  }
  return sinks;
}

const pir::Operation* GetSoleSink(const OpSet& ops) {
  const auto& sinks = GetSinks(ops);
  CHECK_EQ(sinks.size(), 1);
  return *sinks.begin();
}

template <typename DoEachT>
void VisitStmtOpImpl(const IS& injective_source, const DoEachT& DoEach) {
  for (const auto* op : injective_source.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const PS& partial_shardable, const DoEachT& DoEach) {
  for (const auto* op : partial_shardable.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const R& reduce, const DoEachT& DoEach) {
  std::visit(adt::match{
                 [](const std::monostate&) {
                   // do nothing.
                 },
                 [&](const IS& injective_source) {
                   VisitStmtOpImpl(injective_source, DoEach);
                 },
                 [&](const PS& partial_shardable) {
                   VisitStmtOpImpl(partial_shardable, DoEach);
                 },
             },
             reduce.input);
  DoEach(reduce.reduce_op_pattern.reduce_op);
}

template <typename DoEachT>
void VisitStmtOp(const StmtPattern& stmt, const DoEachT& DoEach) {
  std::visit([&](const auto& impl) { VisitStmtOpImpl(impl, DoEach); }, stmt);
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInThisFusionOp(
    const std::vector<const pir::Operation*>& ops) {
  std::set<const pir::Operation*> set;
  for (const pir::Operation* op : ops) {
    if (!op->isa<::pir::YieldOp>()) {
      set.insert(op);
    }
  }
  return [set = std::move(set)](const pir::Operation* op) {
    return set.count(op) > 0;
  };
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const OpTopo& op_topo) {
  const auto& IsSource = [&](const pir::Operation* op) {
    std::size_t num_inputs = 0;
    op_topo.VisitInputOp(op,
                         [&](const pir::Operation* input) { ++num_inputs; });
    return num_inputs == 0;
  };

  const auto starts = [&] {
    std::list<const pir::Operation*> starts;
    for (const auto* op : *op_topo.ops) {
      if (IsSource(op)) {
        starts.push_back(op);
      } else {
        // do nothing.
      }
    }
    return starts;
  }();

  std::unordered_map<const pir::Operation*, bool> op_2_is_injective_source;

  auto IsInputsAllInjectiveSource = [&](const pir::Operation* op) {
    bool is_inputs_all_injective_source = true;
    op_topo.VisitInputOp(op, [&](const pir::Operation* input) {
      is_inputs_all_injective_source = (is_inputs_all_injective_source &&
                                        op_2_is_injective_source.at(input));
    });
    return is_inputs_all_injective_source;
  };
  const auto VisitInput = [&](const pir::Operation* op,
                              const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitOutput = [&](const pir::Operation* op,
                               const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> walker{VisitInput, VisitOutput};
  walker(starts.begin(), starts.end(), [&](const pir::Operation* op) {
    op_2_is_injective_source[op] =
        (IsGeneralInjective(op) && IsInputsAllInjectiveSource(op));
  });
  return [map = std::move(op_2_is_injective_source)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

common::TopoWalker<const pir::Operation*> GetOpsReversedTopoWalker(
    const OpTopo& op_topo) {
  const auto VisitUpStreamInOps = [op_topo](const pir::Operation* op,
                                            const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitDownStreamInOps = [op_topo](const pir::Operation* op,
                                              const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> reversed_walker(
      VisitDownStreamInOps, VisitUpStreamInOps);
  return reversed_walker;
}

size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

std::vector<int64_t> GetReduceAxes(const pir::Operation* reduce_op) {
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("dim");
  CHECK(attr_val.isa<::pir::ArrayAttribute>());
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int64_t> reduce_axes;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
    reduce_axes.push_back(axis);
  }
  return reduce_axes;
}

bool GetReduceOpKeepDims(const pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keep_dim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>();
}

class DefaultShardableAxesProvider final : public ShardableAxesProvider {
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

  const pir::ShapeConstraintIRAnalysis* shape_analysis_;
};

class ShardableAxesInferer {
 public:
  explicit ShardableAxesInferer(
      const std::shared_ptr<ShardableAxesProvider>& shardable_axes_provider)
      : shardable_axes_provider_(shardable_axes_provider) {}

  ShardableAxesInferer(const ShardableAxesInferer&) = default;
  ShardableAxesInferer(ShardableAxesInferer&&) = default;

  ShardableAxesSignature MakeShardableAxesSignature4Op(
      const pir::Operation* op) {
    return shardable_axes_provider_->MakeShardableAxesSignature4Op(op);
  }

  std::unordered_map<pir::Value, ShardableAxes> InferShardableAxesFromSink(
      const pir::Operation* sink, const OpTopo& op_topo) {
    auto reversed_walker = GetOpsReversedTopoWalker(op_topo);
    CHECK_GT(op_topo.ops->count(sink), 0);
    const int result_idx = GetOutputShardableAxesResultIdx(sink);
    size_t rank = GetRank(sink->result(result_idx));
    const auto& init_sa = ShardableAxesUtil::MakeFullyShardableAxes(rank);
    return ReversedInferShardableAxes(reversed_walker, sink, init_sa);
  }

  std::unordered_map<pir::Value, ShardableAxes> InferShardableAxes(
      const OpSetPtr& ops) {
    auto reversed_walker = GetOpsReversedTopoWalker(OpTopo{
        .ops = ops,
    });
    const auto& sinks = GetSinks(*ops);
    const auto& sink_and_init_value =
        GetSinkAndInitValues(reversed_walker, ops, sinks);
    return ReversedInferShardableAxes(reversed_walker,
                                      sink_and_init_value.begin(),
                                      sink_and_init_value.end());
  }

 private:
  template <typename InputIt>
  std::unordered_map<pir::Value, ShardableAxes> ReversedInferShardableAxes(
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

  std::unordered_map<pir::Value, ShardableAxes> ReversedInferShardableAxes(
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
  GetOp2ShardableAxesSignature(const OpSetPtr& ops) {
    std::unordered_map<const pir::Operation*, ShardableAxesSignature> ret;
    for (const auto* op : *ops) {
      ret[op] = MakeShardableAxesSignature4Op(op);
    }
    return ret;
  }

  std::map<std::string, std::vector<std::string>> GetAxisName2BoundAxisName(
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

  std::unordered_map<std::string, std::string> GetAxisName2UnionFindSetRoot(
      const OpSetPtr& ops,
      const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
          op2shardable_axes_signature) {
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

  std::unordered_map<pir::Value, ShardableAxes> GetSinkAndInitShardableAxes(
      const std::list<const pir::Operation*>& sinks,
      const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
          op2shardable_axes_signature,
      const std::unordered_map<std::string, std::string>&
          axis_name2union_find_set_root) {
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

  void RenameDuplicatedAxisName(
      std::unordered_map<pir::Value, ShardableAxes>* sink2sa) {
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

  std::unordered_map<pir::Value, ShardableAxes> GetSinkAndInitValues(
      const common::TopoWalker<const pir::Operation*>& reverse_walker,
      const OpSetPtr& ops,
      const std::list<const pir::Operation*>& sinks) {
    const auto& op2shardable_axes_signature = GetOp2ShardableAxesSignature(ops);
    const auto& axis_name2union_find_set_root =
        GetAxisName2UnionFindSetRoot(ops, op2shardable_axes_signature);
    std::unordered_map<pir::Value, ShardableAxes> sink_and_inits =
        GetSinkAndInitShardableAxes(
            sinks, op2shardable_axes_signature, axis_name2union_find_set_root);
    RenameDuplicatedAxisName(&sink_and_inits);
    return sink_and_inits;
  }

  std::shared_ptr<ShardableAxesProvider> shardable_axes_provider_;
};

std::function<size_t(const pir::Operation*)> MakeTopoOrderFinderOfOp(
    const std::vector<const pir::Operation*>& ops) {
  std::unordered_map<const pir::Operation*, size_t> op2order_in_block;
  size_t order = 0;
  for (const pir::Operation* op : ops) {
    op2order_in_block[op] = ++order;
  }
  return [map = std::move(op2order_in_block)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

pir::Value GetStmtBigestShapeValueImpl(const IS& injective_source) {
  const auto* sink_op = injective_source.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValueImpl(const R& reduce_pattern) {
  const auto* sink_op = reduce_pattern.reduce_op_pattern.reduce_op;
  CHECK_EQ(sink_op->num_operands(), 1);
  return sink_op->operand_source(0);
}

pir::Value GetStmtBigestShapeValueImpl(const PS& partial_shardable) {
  const auto* sink_op = partial_shardable.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValue(const StmtPattern& stmt) {
  return std::visit(
      [&](const auto& impl) { return GetStmtBigestShapeValueImpl(impl); },
      stmt);
}

const pir::Operation* GetStmtSoleSinkImpl(const IS& injective_source) {
  return injective_source.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const PS& partial_shardable) {
  return partial_shardable.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const R& reduce) {
  return reduce.reduce_op_pattern.reduce_op;
}

const pir::Operation* GetStmtSoleSinkOp(const StmtPattern& stmt) {
  return std::visit([](const auto& impl) { return GetStmtSoleSinkImpl(impl); },
                    stmt);
}

void SortStmtPtrs(
    std::vector<const StmtPattern*>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
  auto GetOrderValue4Stmt = [&](const StmtPattern* stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(*stmt);
    return OrderValue4Op(sink_op);
  };
  const auto Cmp = [&](const auto* lhs, const auto* rhs) {
    const auto& lhs_order = GetOrderValue4Stmt(lhs);
    const auto& rhs_order = GetOrderValue4Stmt(rhs);
    return lhs_order < rhs_order;
  };
  std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
}

class StmtFusionHelper {
 public:
  StmtFusionHelper(const std::vector<const pir::Operation*>& ops,
                   const ShardableAxesInferer& shardable_axes_inferer)
      : ops_(ops), shardable_axes_inferer_(shardable_axes_inferer) {
    this->op_topo_ = OpTopo::Make(ops);
    this->IsInThisOpList = MakePredicatorIsInThisFusionOp(ops);
    this->IsInjectiveSource = MakePredicatorIsInjectiveSource(this->op_topo_);
    this->GetOrderValue4Op = MakeTopoOrderFinderOfOp(ops);
  }

  GroupPattern FuseToGroupPattern() {
    std::vector<StmtPattern> stmt_patterns = ConvertToStmtsPattern();
    if (const auto& error = Fuse_IS_x_IS_2_IS(&stmt_patterns))
      return error.value();
    if (const auto& error = Fuse_PS_x_PS_2_PS(&stmt_patterns))
      return error.value();
    if (const auto& error = Fuse_IS_x_PS_2_PS(&stmt_patterns))
      return error.value();
    if (const auto& error = Fuse_IS_x_R_2_R(&stmt_patterns))
      return error.value();
    if (const auto& error = Fuse_PS_x_R_2_R(&stmt_patterns))
      return error.value();
    SortStmtPatterns(&stmt_patterns);
    return stmt_patterns;
  }

 private:
  std::vector<StmtPattern> ConvertToStmtsPattern() {
    std::vector<StmtPattern> ret;
    for (const auto* op : ops_) {
      if (!IsInThisOpList(op)) continue;
      ret.emplace_back(ConvertToStmtPattern(op));
    }
    return ret;
  }

  void SortStmtPatterns(std::vector<StmtPattern>* stmt_patterns) {
    std::vector<const StmtPattern*> stmt_ptr_patterns = [&] {
      std::vector<const StmtPattern*> stmt_ptr_patterns;
      stmt_ptr_patterns.reserve(stmt_patterns->size());
      for (const auto& stmt_pattern : *stmt_patterns) {
        stmt_ptr_patterns.push_back(&stmt_pattern);
      }
      return stmt_ptr_patterns;
    }();
    SortStmtPtrs(&stmt_ptr_patterns, this->GetOrderValue4Op);
    *stmt_patterns = [&] {
      std::vector<StmtPattern> sorted_stmts;
      sorted_stmts.reserve(stmt_ptr_patterns.size());
      for (const auto* stmt_ptr : stmt_ptr_patterns) {
        sorted_stmts.push_back(*stmt_ptr);
      }
      return sorted_stmts;
    }();
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(
      std::vector<StmtPattern>* stmt_patterns) {
    const auto ConstructISPattern = [&](const auto& ops) {
      return IS{
          .ops = ops,
          .sole_sink = GetSoleSink(OpSet(ops.begin(), ops.end())),
      };
    };
    return MultiFuse(IsISPattern, ConstructISPattern, stmt_patterns);
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns) {
    const auto ConstructPSPattern = [&](const auto& ops) {
      auto op_topo = OpTopo::Make(ops);
      const auto shardable_axes_signature = GetShardableAxesSignature(op_topo);
      return PS{
          .ops = ops,
          .sole_sink = GetSoleSink(OpSet(ops.begin(), ops.end())),
          .shardable_axes_signature = shardable_axes_signature,
      };
    };
    return MultiFuse(IsPSPattern, ConstructPSPattern, stmt_patterns);
  }

  struct FusePolicy_IS_x_PS_2_PS {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsPSPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<IS>(upstream), std::get<PS>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const PS& downstream) {
      const auto& ops = [&] {
        std::vector<const pir::Operation*> ops(upstream.ops.begin(),
                                               upstream.ops.end());
        for (const auto* downstream_op : downstream.ops) {
          if (std::find(ops.begin(), ops.end(), downstream_op) == ops.end()) {
            ops.push_back(downstream_op);
          }
        }
        return ops;
      }();
      const auto& shardable_axes_signature =
          MergeShardableAxesSignature(upstream, downstream);
      return StmtPattern(PS{
          .ops = ops,
          .sole_sink = downstream.sole_sink,
          .shardable_axes_signature = shardable_axes_signature,
      });
    }

    static ShardableAxesSignature MergeShardableAxesSignature(
        const IS& upstream, const PS& downstream) {
      LOG(FATAL) << "TODO(tianchao)";
    }
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns) {
    return FuseFilteredStmtPatterns<FusePolicy_IS_x_PS_2_PS>(stmt_patterns);
  }
  struct FusePolicy_IS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsRPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<IS>(upstream), std::get<R>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const R& downstream) {
      if (downstream.HasFusedInput()) {
        return ErrorGroupPattern{
            .ops = {downstream.reduce_op_pattern.reduce_op},
            .error_string = "The input of reduce has been fused.",
        };
      }
      R new_pattern = R(downstream);
      new_pattern.input = upstream;
      return StmtPattern(std::move(new_pattern));
    }
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(
      std::vector<StmtPattern>* stmt_patterns) {
    return FuseFilteredStmtPatterns<FusePolicy_IS_x_R_2_R>(stmt_patterns);
  }

  struct FusePolicy_PS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsRPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<PS>(upstream), std::get<R>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const PS& upstream, const R& downstream) {
      if (downstream.HasFusedInput()) {
        return ErrorGroupPattern{
            .ops = {downstream.reduce_op_pattern.reduce_op},
            .error_string = "The input of reduce has been fused.",
        };
      }
      R new_pattern = R(downstream);
      new_pattern.input = upstream;
      return StmtPattern(new_pattern);
    }
  };

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(
      std::vector<StmtPattern>* stmt_patterns) {
    return FuseFilteredStmtPatterns<FusePolicy_PS_x_R_2_R>(stmt_patterns);
  }

  StmtPattern ConvertToStmtPattern(const pir::Operation* op) {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (IsInjectiveSource(op)) {
      return ConvertToIS(op);
    } else if (kind == hlir::framework::kReduction) {
      return ConvertReductionOpToReductionPattern(op);
    } else if (kind == hlir::framework::kElementWise) {
      return ConvertOpToPS(op);
    } else if (kind == hlir::framework::kBroadcast) {
      return ConvertOpToPS(op);
    } else {
      LOG(FATAL)
          << "only kReduction, kElementWise, kBroadcast supported. op_name:"
          << op->name();
    }
    LOG(FATAL) << "Dead code";
  }

  IS ConvertToIS(const pir::Operation* op) {
    VLOG(4) << "Converting Op to IS";
    return IS{
        .ops = {op},
        .sole_sink = op,
    };
  }

  R ConvertReductionOpToReductionPattern(const pir::Operation* op) {
    VLOG(4) << "Converting Op to R";
    return R{{}, {op}};
  }

  PS ConvertOpToPS(const pir::Operation* op) {
    VLOG(4) << "Converting Op to PS";
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    const auto shardable_axes_signature =
        shardable_axes_inferer_.MakeShardableAxesSignature4Op(op);
    return PS{
        .ops = {op},
        .sole_sink = op,
        .shardable_axes_signature = shardable_axes_signature,
    };
  }

  using StmtPtr4OpT =
      std::function<std::optional<StmtPattern*>(const pir::Operation*)>;
  static StmtPtr4OpT MakeStmtFinderFromOp(std::vector<StmtPattern>* stmts) {
    std::unordered_map<const pir::Operation*, StmtPattern*> op2stmt_ptr;
    for (auto& stmt : *stmts) {
      VisitStmtOp(stmt, [&](const auto* op) { op2stmt_ptr[op] = &stmt; });
    }
    return [map = std::move(op2stmt_ptr)](
               const pir::Operation* op) -> std::optional<StmtPattern*> {
      const auto iter = map.find(op);
      if (iter == map.end()) return std::nullopt;
      return iter->second;
    };
  }

  template <typename IsChozenPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsChozenPatternT& IsChozenPattern,
      const ConstructPatternT& ConstructPattern,
      std::vector<StmtPattern>* stmts) {
    const auto StmtFinder = MakeStmtFinderFromOp(stmts);
    const auto VisitInputStmt = [&](const StmtPattern* stmt,
                                    const StmtVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo_.VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            if (IsChozenPattern(*input_stmt.value())) {
              DoEach(input_stmt.value());
            }
          }
        });
      });
    };
    const auto VisitOutputStmt = [&](const StmtPattern* stmt,
                                     const StmtVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo_.VisitOutputOp(op, [&](const pir::Operation* output) {
          if (const auto& output_stmt = StmtFinder(output)) {
            if (IsChozenPattern(*output_stmt.value())) {
              DoEach(output_stmt.value());
            }
          }
        });
      });
    };
    const auto IsSinkPattern = [&](const StmtPattern* stmt) {
      if (!IsChozenPattern(*stmt)) return false;
      std::size_t num_injective_src_outputs = 0;
      VisitOutputStmt(stmt, [&](const auto& consumer) {
        num_injective_src_outputs += IsChozenPattern(*consumer);
      });
      return num_injective_src_outputs == 0;
    };
    const auto Cmp = [&](const auto* lhs, const auto* rhs) {
      return this->GetOrderValue4Op(lhs) < this->GetOrderValue4Op(rhs);
    };
    common::BfsWalker<const StmtPattern*> reverse_walker(VisitInputStmt);
    const auto& GetAllUpstreamOps = [&](const StmtPattern* stmt_ptr) {
      std::vector<const pir::Operation*> visited_ops;
      reverse_walker(stmt_ptr, [&](const StmtPattern* node) {
        VisitStmtOp(*node, [&](const auto* op) { visited_ops.push_back(op); });
      });
      std::sort(visited_ops.begin(), visited_ops.end(), Cmp);
      return visited_ops;
    };

    std::vector<StmtPattern> ret_stmts = [&] {
      std::vector<StmtPattern> ret_stmts;
      ret_stmts.reserve(stmts->size());
      for (const auto& stmt : *stmts) {
        if (!IsChozenPattern(stmt)) {
          ret_stmts.push_back(stmt);
        } else {
          // do nothing.
        }
      }
      return ret_stmts;
    }();
    for (auto& stmt : *stmts) {
      if (!IsSinkPattern(&stmt)) continue;
      ret_stmts.emplace_back(ConstructPattern(GetAllUpstreamOps(&stmt)));
    }
    *stmts = ret_stmts;
    return std::nullopt;
  }

  struct StmtIterPair {
    std::list<StmtPattern*>::iterator upstream_iter;
    std::list<StmtPattern*>::iterator downstream_iter;
  };

  bool IsConnected(const StmtPtr4OpT& StmtFinder,
                   const StmtPattern* upstream,
                   const StmtPattern* downstream) {
    const auto VisitInputStmt = [&](const StmtPattern* stmt,
                                    const StmtVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo_.VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            DoEach(input_stmt.value());
          }
        });
      });
    };

    bool found = false;
    VisitInputStmt(downstream, [&](const StmtPattern* input_pattern) {
      if (input_pattern == upstream) {
        found = true;
      }
    });
    return found;
  }

  template <typename FuseTargetConditionT>
  std::optional<StmtIterPair> FindConnetedPattenPairWithCondition(
      const StmtPtr4OpT& StmtFinder,
      std::list<StmtPattern*>* stmt_ptrs,
      const FuseTargetConditionT& FuseTargetCondition) {
    for (auto dst_iter = stmt_ptrs->begin(); dst_iter != stmt_ptrs->end();
         ++dst_iter) {
      for (auto src_iter = stmt_ptrs->begin(); src_iter != stmt_ptrs->end();
           ++src_iter) {
        if (src_iter == dst_iter) continue;
        if (!IsConnected(StmtFinder, *src_iter, *dst_iter)) continue;
        if (FuseTargetCondition(**src_iter, **dst_iter)) {
          return StmtIterPair{
              .upstream_iter = src_iter,
              .downstream_iter = dst_iter,
          };
        }
      }
    }
    return std::nullopt;
  }

  template <typename FusionPolicy>
  std::optional<ErrorGroupPattern> FuseFilteredStmtPatterns(
      std::vector<StmtPattern>* stmt_patterns) {
    std::list<StmtPattern*> stmts_iters = [&] {
      std::list<StmtPattern*> stmts_iters;
      for (auto& stmt : *stmt_patterns) {
        stmts_iters.push_back(&stmt);
      }
      return stmts_iters;
    }();
    const auto StmtFinder = MakeStmtFinderFromOp(stmt_patterns);
    const auto EraseOld = [&](const StmtIterPair& pattern_pair) {
      stmts_iters.erase(pattern_pair.upstream_iter);
      stmts_iters.erase(pattern_pair.downstream_iter);
    };
    const auto& InsertNew = [&](const StmtPattern& stmt_pattern) {
      stmt_patterns->push_back(stmt_pattern);
      stmts_iters.push_back(&stmt_patterns->back());
    };
    while (true) {
      const auto& pattern_pair = FindConnetedPattenPairWithCondition(
          StmtFinder, &stmts_iters, &FusionPolicy::FuseCondition);
      if (!pattern_pair.has_value()) break;
      const std::variant<StmtPattern, ErrorGroupPattern>& new_pattern =
          FusionPolicy::MergePattern(**pattern_pair.value().upstream_iter,
                                     **pattern_pair.value().downstream_iter);

      if (std::holds_alternative<ErrorGroupPattern>(new_pattern)) {
        return std::get<ErrorGroupPattern>(new_pattern);
      }
      EraseOld(pattern_pair.value());
      InsertNew(std::get<StmtPattern>(new_pattern));
    }
    *stmt_patterns = [&] {
      std::vector<StmtPattern> ret_patterns;
      ret_patterns.reserve(stmts_iters.size());
      for (const auto& stmt_iter : stmts_iters) {
        ret_patterns.push_back(*stmt_iter);
      }
      return ret_patterns;
    }();
    return std::nullopt;
  }

  ShardableAxesSignature GetShardableAxesSignature(const OpTopo& op_topo) {
    const pir::Operation* sink = [&] {
      const auto& sinks = GetSinks(*op_topo.ops);
      CHECK_EQ(sinks.size(), 1) << "ops must have only one sink node.";
      return *sinks.begin();
    }();
    const auto& value2shardable_axes =
        shardable_axes_inferer_.InferShardableAxesFromSink(sink, op_topo);
    const auto& IsInputOpOperand = [&](const auto* op, int input_idx) {
      const auto& defining_op = op->operand_source(input_idx).defining_op();
      return IsInThisOpList(defining_op) &&
             op_topo.ops->count(defining_op) == 0;
    };
    const auto& input_op_operands = [&] {
      std::vector<OpAndOperandIndex> op_operands;
      for (const auto* op : *op_topo.ops) {
        for (int i = 0; i < op->num_operands(); ++i) {
          if (!IsInputOpOperand(op, i)) continue;
          op_operands.emplace_back(OpAndOperandIndex{op, i});
        }
      }
      return op_operands;
    }();
    const auto& shardable_axes_sig = [&] {
      ShardableAxesSignature signature;
      int result_idx = GetOutputShardableAxesResultIdx(sink);
      signature.sole_output_sa = SoleOutputShardableAxes{
          .shardable_axes = value2shardable_axes.at(sink->result(result_idx)),
      };
      for (const auto& pair : input_op_operands) {
        const auto& [op, idx] = pair;
        pir::Value input = op->operand_source(idx);
        signature.input_shardable_axes[pair] = value2shardable_axes.at(input);
      }
      return signature;
    }();
    return shardable_axes_sig;
  }

 private:
  std::vector<const pir::Operation*> ops_;
  ShardableAxesInferer shardable_axes_inferer_;
  OpTopo op_topo_;
  std::function<bool(const pir::Operation*)> IsInThisOpList;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
  std::function<size_t(const pir::Operation*)> GetOrderValue4Op;
};

class ClusteringEngine {
 public:
  ClusteringEngine(const std::vector<const pir::Operation*>& ops,
                   const ShardableAxesInferer& shardable_axes_inferer,
                   const std::shared_ptr<ClusteringPolicy>& clustering_policy)
      : ops_(ops),
        op_topo_(OpTopo::Make(ops)),
        shardable_axes_inferer_(shardable_axes_inferer),
        clustering_policy_(clustering_policy) {}

  ClusteringResult ClusterOps() {
    VLOG(4) << "- Raw Parsing";
    const std::vector<StmtPattern> stmt_patterns = [&] {
      GroupPattern raw_parsed =
          StmtFusionHelper(ops_, shardable_axes_inferer_).FuseToGroupPattern();
      CHECK(!std::holds_alternative<ErrorGroupPattern>(raw_parsed))
          << std::get<ErrorGroupPattern>(raw_parsed).error_string;
      CHECK(std::holds_alternative<std::vector<StmtPattern>>(raw_parsed));
      return std::get<std::vector<StmtPattern>>(raw_parsed);
    }();
    auto OrderValue4Op = MakeTopoOrderFinderOfOp(ops_);
    VLOG(4) << "- Making Acyclic Same Cluster Bfs Walker";
    common::BfsWalker<const StmtPattern*> walker =
        MakeAcyclicSameClusterBfsWalker(stmt_patterns);
    std::vector<std::vector<const StmtPattern*>> stmts_list;
    VLOG(4) << "- Visit Connect Component";
    VisitConnectedComponent(walker, stmt_patterns, [&](auto stmt_ptrs) {
      SortStmtPtrs(&stmt_ptrs, OrderValue4Op);
      stmts_list.push_back(stmt_ptrs);
    });
    VLOG(4) << "- Sort Stmts List";
    SortStmtsList(&stmts_list, OrderValue4Op);
    VLOG(4) << "- Make Clustering Result";
    return clustering_policy_->MakeClusteringResult(stmts_list);
  }

 private:
  void SortStmtsList(
      std::vector<std::vector<const StmtPattern*>>* stmt_ptrs,
      const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
    auto GetOrderValue = [&](const std::vector<const StmtPattern*>& stmts) {
      CHECK(!stmts.empty());
      return OrderValue4Op(GetStmtSoleSinkOp(*stmts.back()));
    };
    auto Cmp = [&](const auto& lhs, const auto& rhs) {
      return GetOrderValue(lhs) < GetOrderValue(rhs);
    };
    std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
  }

  template <typename DoEachComponentT>
  void VisitConnectedComponent(
      const common::BfsWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const DoEachComponentT& DoEachComponent) {
    std::unordered_set<const StmtPattern*> visited;
    for (const auto& start : stmt_patterns) {
      if (visited.count(&start)) continue;
      std::vector<const StmtPattern*> component;
      walker(&start, [&](const auto* stmt) {
        component.push_back(stmt);
        CHECK(visited.emplace(stmt).second);
      });
      DoEachComponent(component);
    }
  }

  common::BfsWalker<const StmtPattern*> MakeAcyclicSameClusterBfsWalker(
      const std::vector<StmtPattern>& stmt_patterns) {
    const auto entire_topo_walk = MakeTopoWalker(op_topo_, stmt_patterns);
    const auto ClusterRoot4Stmt =
        MakeClusterRoot4Stmt(entire_topo_walk, stmt_patterns);
    const auto IsInSameCluster = [=](const auto* lhs, const auto* rhs) {
      return ClusterRoot4Stmt(lhs) == ClusterRoot4Stmt(rhs);
    };
    const auto IsAcyclicConnected = MakePredicatorIsAcyclicConnected(
        entire_topo_walk, stmt_patterns, ClusterRoot4Stmt);
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    const auto VisitAcyclicClusterNext = [=](const StmtPattern* stmt,
                                             const NodeVisitor& DoEach) {
      entire_topo_walk.VisitPrevNodes(stmt, [&](const StmtPattern* input) {
        if (!IsInSameCluster(input, stmt)) return;
        if (!IsAcyclicConnected(input, stmt)) return;
        DoEach(input);
      });
      entire_topo_walk.VisitNextNodes(stmt, [&](const StmtPattern* output) {
        if (!IsInSameCluster(stmt, output)) return;
        if (!IsAcyclicConnected(stmt, output)) return;
        DoEach(output);
      });
    };
    return common::BfsWalker<const StmtPattern*>(VisitAcyclicClusterNext);
  }

  using IsAcyclicConnectedT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;
  using ClusterRoot4StmtT =
      std::function<const StmtPattern*(const StmtPattern*)>;

  IsAcyclicConnectedT MakePredicatorIsAcyclicConnected(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
    const auto AllTopClosureUpstreams4Stmt = MakeAllTopClosureUpstreams4Stmt(
        walker, stmt_patterns, ClusterRoot4Stmt);
    const auto IsSrcAcyclicConnectedToDst = [&](const auto* src,
                                                const auto* dst) {
      // return true if there exist no other clusters's node in
      // all_topo_closure_upstreams(dst) - all_topo_closure_upstreams(src)
      const auto* src_upstreams = AllTopClosureUpstreams4Stmt(src);
      const auto* dst_upstreams = AllTopClosureUpstreams4Stmt(dst);
      std::vector<const StmtPattern*> diff_stmts;
      std::set_difference(dst_upstreams->begin(),
                          dst_upstreams->end(),
                          src_upstreams->begin(),
                          src_upstreams->end(),
                          std::back_inserter(diff_stmts));
      const auto* cluster_root = ClusterRoot4Stmt(src);
      CHECK_EQ(cluster_root, ClusterRoot4Stmt(dst));
      for (const auto* diff_stmt : diff_stmts) {
        if (ClusterRoot4Stmt(diff_stmt) != cluster_root) return false;
      }
      return true;
    };
    using Src2AcyclicConnectedDst =
        std::map<const StmtPattern*, std::set<const StmtPattern*>>;
    Src2AcyclicConnectedDst src2acyclic_connected_dst;
    for (const auto& stmt : stmt_patterns) {
      const auto* src = &stmt;
      auto* acyclic_connected_dst = &src2acyclic_connected_dst[src];
      walker.VisitNextNodes(src, [&](const auto* dst) {
        if (!(acyclic_connected_dst->count(dst) == 0)) return;
        if (!(ClusterRoot4Stmt(src) == ClusterRoot4Stmt(dst))) return;
        if (IsSrcAcyclicConnectedToDst(src, dst)) {
          acyclic_connected_dst->insert(dst);
        }
      });
    }
    return [map = std::move(src2acyclic_connected_dst)](
               const StmtPattern* src, const StmtPattern* dst) {
      const auto& iter = map.find(src);
      if (iter == map.end()) return false;
      return iter->second.count(dst) > 0;
    };
  }

  struct TopoClosure {
    std::list<const StmtPattern*> sources;
    std::list<const StmtPattern*> sinks;
    std::unordered_set<const StmtPattern*> stmts;
  };

  using TopoClosure4RootStmtT =
      std::function<std::optional<const TopoClosure*>(const StmtPattern*)>;

  using AllTopClosureUpstreams4StmtT =
      std::function<const std::set<const StmtPattern*>*(const StmtPattern*)>;

  AllTopClosureUpstreams4StmtT MakeAllTopClosureUpstreams4Stmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
    const auto TopoClosure4RootStmt = MakeTopoClosure4RootStmt(
        entire_topo_walker, stmt_patterns, ClusterRoot4Stmt);
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
        stmt2all_topo_closure_upstreams;
    for (const auto& stmt_pattern : stmt_patterns) {
      if (stmt2all_topo_closure_upstreams.count(&stmt_pattern)) continue;
      const auto* cluster_root = ClusterRoot4Stmt(&stmt_pattern);
      const auto& topo_closure = TopoClosure4RootStmt(cluster_root);
      CHECK(topo_closure.has_value());
      VisitStmtTopoClosureUpstreams(
          entire_topo_walker,
          *topo_closure.value(),
          [&](const auto* stmt, const auto& all_topo_closure_upstreams) {
            if (ClusterRoot4Stmt(stmt) != cluster_root) return;
            CHECK(stmt2all_topo_closure_upstreams
                      .emplace(stmt, all_topo_closure_upstreams)
                      .second);
          });
    }
    return [map = std::move(stmt2all_topo_closure_upstreams)](
               const StmtPattern* stmt) {
      const auto iter = map.find(stmt);
      if (iter == map.end()) {
        static const std::set<const StmtPattern*> empty;
        return &empty;
      }
      return &iter->second;
    };
  }

  TopoClosure4RootStmtT MakeTopoClosure4RootStmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitClusterInput = [&](const StmtPattern* stmt,
                                 const NodeVisitor& DoEach) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
        if (ClusterRoot4Stmt(stmt) == ClusterRoot4Stmt(input)) {
          DoEach(input);
        }
      });
    };
    auto IsClusterSource = [&](const auto* stmt) {
      size_t num_inputs = 0;
      VisitClusterInput(stmt, [&](const auto*) { ++num_inputs; });
      return num_inputs == 0;
    };
    auto VisitClusterOutput = [&](const StmtPattern* stmt,
                                  const NodeVisitor& DoEach) {
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
        if (ClusterRoot4Stmt(stmt) == ClusterRoot4Stmt(output)) {
          DoEach(output);
        }
      });
    };
    auto IsClusterSink = [&](const auto* stmt) {
      size_t num_outputs = 0;
      VisitClusterOutput(stmt, [&](const auto*) { ++num_outputs; });
      return num_outputs == 0;
    };
    auto VisitClusterNext = [&](const StmtPattern* stmt,
                                const NodeVisitor& DoEach) {
      VisitClusterInput(stmt, DoEach);
      VisitClusterOutput(stmt, DoEach);
    };
    common::BfsWalker<const StmtPattern*> cluster_bfs_walker(VisitClusterNext);
    const auto IsReachable = MakeIsReachable(entire_topo_walker, stmt_patterns);
    std::unordered_map<const StmtPattern*, TopoClosure> root_stmt2topo_closure;
    for (const auto& stmt_pattern : stmt_patterns) {
      const auto* cluster_root = ClusterRoot4Stmt(&stmt_pattern);
      if (cluster_root != &stmt_pattern) continue;
      CHECK(!(root_stmt2topo_closure.count(cluster_root)));
      auto* topo_closure = &root_stmt2topo_closure[cluster_root];
      cluster_bfs_walker(cluster_root, [&](const auto* stmt) {
        if (IsClusterSource(stmt)) {
          topo_closure->sources.push_back(stmt);
        }
        if (IsClusterSink(stmt)) {
          topo_closure->sinks.push_back(stmt);
        }
      });
      topo_closure->stmts = CollectSubGraphAllStmts(entire_topo_walker,
                                                    IsReachable,
                                                    topo_closure->sources,
                                                    topo_closure->sinks);
    }
    return [map = std::move(root_stmt2topo_closure)](
               const StmtPattern* stmt) -> std::optional<const TopoClosure*> {
      const auto iter = map.find(stmt);
      if (iter == map.end()) return std::nullopt;
      return &iter->second;
    };
  }

  using IsReachableT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;

  std::unordered_set<const StmtPattern*> CollectSubGraphAllStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const IsReachableT& IsReachable,
      const std::list<const StmtPattern*> sources,
      const std::list<const StmtPattern*> sinks) {
    auto IsConnectedToOneSource = [&](const auto* stmt) {
      for (const auto* source : sources) {
        if (IsReachable(source, stmt)) return true;
      }
      return false;
    };
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitInput = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
        if (IsConnectedToOneSource(input)) {
          DoEach(input);
        }
      });
    };
    auto IsConnectedToOneSink = [&](const auto* stmt) {
      for (const auto* sink : sinks) {
        if (IsReachable(stmt, sink)) return true;
      }
      return false;
    };
    auto VisitOutput = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
        if (IsConnectedToOneSink(output)) {
          DoEach(output);
        }
      });
    };
    auto VisitNext = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      VisitInput(stmt, DoEach);
      VisitOutput(stmt, DoEach);
    };
    std::unordered_set<const StmtPattern*> ret;
    common::BfsWalker<const StmtPattern*> bfs_walker(VisitNext);
    bfs_walker(sources.begin(), sources.end(), [&](const auto* stmt) {
      ret.insert(stmt);
    });
    return ret;
  }

  template <typename DoEachStmtAndTopoClosureUpstreamsT>
  void VisitStmtTopoClosureUpstreams(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const TopoClosure& topo_closure,
      const DoEachStmtAndTopoClosureUpstreamsT&
          DoEachStmtAndTopoClosureUpstreams) {
    const auto IsInTopoClosure = [&](const auto* stmt) {
      return topo_closure.stmts.count(stmt) > 0;
    };
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitInput = [&](const auto* stmt, const NodeVisitor& Visit) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
        if (IsInTopoClosure(input)) {
          Visit(input);
        }
      });
    };
    auto VisitOutput = [&](const auto* stmt, const NodeVisitor& Visit) {
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
        if (IsInTopoClosure(output)) {
          Visit(output);
        }
      });
    };
    common::TopoWalker<const StmtPattern*> closure_walker(VisitInput,
                                                          VisitOutput);
    const auto& sources = topo_closure.sources;
    std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
        stmt2all_topo_closure_upstreams;
    closure_walker(sources.begin(), sources.end(), [&](const auto* stmt) {
      auto* stmt_upstreams = &stmt2all_topo_closure_upstreams[stmt];
      VisitInput(stmt, [&](const auto* input) {
        stmt_upstreams->insert(input);
        const auto& input_upstreams = stmt2all_topo_closure_upstreams[input];
        stmt_upstreams->insert(input_upstreams.begin(), input_upstreams.end());
      });
      const auto* const_stmt_upstreams = stmt_upstreams;
      DoEachStmtAndTopoClosureUpstreams(stmt, *const_stmt_upstreams);
    });
  }

  IsReachableT MakeIsReachable(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns) {
    const auto& sources = [&] {
      std::list<const StmtPattern*> sources;
      const auto IsSource = [&](const auto* stmt) {
        size_t num_upstreams = 0;
        walker.VisitPrevNodes(stmt, [&](const auto*) { ++num_upstreams; });
        return num_upstreams == 0;
      };
      for (const auto& stmt : stmt_patterns) {
        if (IsSource(&stmt)) {
          sources.push_back(&stmt);
        }
      }
      return sources;
    }();

    std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
        stmt2upstreams;
    walker(sources.begin(), sources.end(), [&](const auto* stmt) {
      (void)stmt2upstreams[stmt];
      walker.VisitPrevNodes(stmt, [&](const auto* upstream) {
        stmt2upstreams[stmt].insert(upstream);
      });
    });
    return [map = std::move(stmt2upstreams)](const StmtPattern* src,
                                             const StmtPattern* dst) {
      if (src == dst) return true;
      const auto iter = map.find(dst);
      if (iter == map.end()) return false;
      return iter->second.count(src) > 0;
    };
  }

  std::function<const StmtPattern*(const StmtPattern*)> MakeClusterRoot4Stmt(
      const common::TopoWalker<const StmtPattern*>& topo_walker,
      const std::vector<StmtPattern>& stmt_patterns) {
    std::unordered_map<const StmtPattern*, const StmtPattern*>
        stmt2cluster_root;
    VisitClusterStmts(topo_walker, stmt_patterns, [&](const auto& stmt_ptrs) {
      CHECK(!stmt_ptrs.empty());
      const auto* root = *stmt_ptrs.begin();
      for (const auto* stmt_ptr : stmt_ptrs) {
        CHECK(stmt2cluster_root.emplace(stmt_ptr, root).second);
      }
    });
    return [map = std::move(stmt2cluster_root)](const StmtPattern* stmt) {
      const auto& iter = map.find(stmt);
      CHECK(iter != map.end());
      return iter->second;
    };
  }

  template <typename DoEachComponentT>
  void VisitClusterStmts(const common::TopoWalker<const StmtPattern*>& walker,
                         const std::vector<StmtPattern>& stmt_patterns,
                         const DoEachComponentT& DoEachComponent) {
    std::vector<const StmtPattern*> stmt_ptrs = [&] {
      std::vector<const StmtPattern*> stmt_ptrs;
      stmt_ptrs.reserve(stmt_patterns.size());
      for (const auto& stmt : stmt_patterns) {
        stmt_ptrs.push_back(&stmt);
      }
      return stmt_ptrs;
    }();
    std::unordered_set<const StmtPattern*> visited;
    while (!stmt_ptrs.empty()) {
      VisitInferedClusterStmts(walker, stmt_ptrs, [&](const auto& component) {
        for (const auto* stmt_ptr : component) {
          CHECK(visited.emplace(stmt_ptr).second);
        }
        DoEachComponent(component);
      });
      stmt_ptrs = [&] {
        std::vector<const StmtPattern*> remainders;
        remainders.reserve(stmt_ptrs.size());
        for (const auto* stmt_ptr : stmt_ptrs) {
          if (visited.count(stmt_ptr)) continue;
          remainders.push_back(stmt_ptr);
        }
        return remainders;
      }();
    }
  }

  template <typename DoEachComponentT>
  void VisitInferedClusterStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<const StmtPattern*>& stmt_ptrs,
      const DoEachComponentT& DoEachComponent) {
    const auto ShardableAxes4Value = MakeInferedShardableAxes4Value(stmt_ptrs);
    const auto Fusible = [&](const auto* src, const auto* dst) {
      return clustering_policy_->IsEdgeFusible(ShardableAxes4Value, *src, *dst);
    };
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    const auto VisitNext = [&](const StmtPattern* stmt,
                               const NodeVisitor& DoEach) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* prev) {
        if (Fusible(prev, stmt)) {
          DoEach(prev);
        }
      });
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* next) {
        if (Fusible(stmt, next)) {
          DoEach(next);
        }
      });
    };
    common::BfsWalker<const StmtPattern*> cluster_walker(VisitNext);
    std::unordered_set<const StmtPattern*> visited;
    for (const auto* start : stmt_ptrs) {
      if (visited.count(start)) continue;
      if (!clustering_policy_->CanActAsSink(ShardableAxes4Value, *start))
        continue;
      std::vector<const StmtPattern*> collected_component;
      cluster_walker(start, [&](const auto* stmt_ptr) {
        collected_component.push_back(stmt_ptr);
        CHECK(visited.emplace(stmt_ptr).second);
      });
      DoEachComponent(collected_component);
    }
    CHECK(!visited.empty())
        << "no StmtPattern visited. please check if "
           "clustering_policy_->CanActAsSink() returns false all the time.";
  }

  using ShardableAxes4ValueT =
      std::function<std::optional<const ShardableAxes*>(pir::Value)>;
  ShardableAxes4ValueT MakeInferedShardableAxes4Value(
      const std::vector<const StmtPattern*>& stmt_ptrs) {
    const OpSetPtr ops = [&] {
      auto ops = std::make_shared<OpSet>();
      for (const auto* stmt_ptr : stmt_ptrs) {
        VisitStmtOp(*stmt_ptr, [&](const auto* op) { ops->insert(op); });
      }
      return ops;
    }();
    auto value2shardable_axes = shardable_axes_inferer_.InferShardableAxes(ops);
    return [map = std::move(value2shardable_axes)](
               pir::Value value) -> std::optional<const ShardableAxes*> {
      const auto& iter = map.find(value);
      if (iter == map.end()) return std::nullopt;
      return &iter->second;
    };
  }

  common::TopoWalker<const StmtPattern*> MakeTopoWalker(
      const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns) {
    using StmtPtrs = std::vector<const StmtPattern*>;
    using Op2OwnerStmtPtrs =
        std::unordered_map<const pir::Operation*, StmtPtrs>;
    auto op2owner_stmt_ptr = std::make_shared<Op2OwnerStmtPtrs>();
    for (const auto& stmt : stmt_patterns) {
      VisitStmtOp(stmt, [&](const pir::Operation* op) {
        (*op2owner_stmt_ptr)[op].push_back(&stmt);
      });
    }
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitInput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo.VisitInputOp(op, [&](const auto* input_op) {
          const auto& owners_iter = op2owner_stmt_ptr->find(input_op);
          if (owners_iter == op2owner_stmt_ptr->end()) return;
          if (owners_iter->second.size() != 1) return;
          const auto* owner_stmt = *owners_iter->second.begin();
          if (owner_stmt == stmt) return;
          DoEach(owner_stmt);
        });
      });
    };
    auto VisitOutput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      const auto* sink = GetStmtSoleSinkOp(*stmt);
      op_topo.VisitOutputOp(sink, [&](const pir::Operation* op) {
        const auto& owners_iter = op2owner_stmt_ptr->find(op);
        if (owners_iter == op2owner_stmt_ptr->end()) return;
        for (const StmtPattern* stmt : owners_iter->second) {
          DoEach(stmt);
        }
      });
    };
    const auto& TryPushBack = [](const auto* stmt, auto* stmts) {
      if (std::find(stmts->begin(), stmts->end(), stmt) == stmts->end()) {
        stmts->push_back(stmt);
      }
    };
    using EdgeCache =
        std::unordered_map<const StmtPattern*, std::vector<const StmtPattern*>>;
    auto stmt2inputs = std::make_shared<EdgeCache>();
    auto stmt2outputs = std::make_shared<EdgeCache>();
    for (const auto& stmt : stmt_patterns) {
      (void)(*stmt2inputs)[&stmt];
      VisitInput(&stmt, [&](const auto* input) {
        TryPushBack(input, &(*stmt2inputs)[&stmt]);
      });
      (void)(*stmt2outputs)[&stmt];
      VisitOutput(&stmt, [&](const auto* output) {
        TryPushBack(output, &(*stmt2outputs)[&stmt]);
      });
    }

    auto VisitCachedInput = [stmt2inputs](const auto* stmt,
                                          const NodeVisitor& DoEach) {
      const auto& map = (*stmt2inputs);
      const auto& iter = map.find(stmt);
      if (iter == map.end()) return;
      for (const auto* input : iter->second) {
        DoEach(input);
      }
    };
    auto VisitCachedOutput = [stmt2outputs](const auto* stmt,
                                            const NodeVisitor& DoEach) {
      const auto& map = (*stmt2outputs);
      const auto& iter = map.find(stmt);
      if (iter == map.end()) return;
      for (const auto* output : iter->second) {
        DoEach(output);
      }
    };
    return common::TopoWalker<const StmtPattern*>(VisitCachedInput,
                                                  VisitCachedOutput);
  }

  const std::vector<const pir::Operation*> ops_;
  const std::shared_ptr<ClusteringPolicy> clustering_policy_;
  ShardableAxesInferer shardable_axes_inferer_;
  const OpTopo op_topo_;
};

class LoopAlignableClusteringPolicy final : public ClusteringPolicy {
 public:
  explicit LoopAlignableClusteringPolicy(
      const pir::ShapeConstraintIRAnalysis* shape_analysis)
      : shape_analysis_(shape_analysis) {}

  bool CanActAsSink(const ShardableAxes4ValueT& ShardableAxes4Value,
                    const api::StmtPattern<FrontendPattern>& stmt) override {
    return IsSinkOpOutputFullyShardable(ShardableAxes4Value, stmt);
  }

  bool IsEdgeFusible(const ShardableAxes4ValueT& ShardableAxes4Value,
                     const api::StmtPattern<FrontendPattern>& src,
                     const api::StmtPattern<FrontendPattern>& dst) override {
    if (!IsSinkOpOutputFullyShardable(ShardableAxes4Value, src)) return false;
    if (!IsSinkOpOutputFullyShardable(ShardableAxes4Value, dst)) return false;
    if (!ReduceOpsSameShardable(ShardableAxes4Value, src, dst)) return false;
    if (!IsTotalLoopSizeEqual(src, dst)) return false;
    return true;
  }

  ClusteringResult MakeClusteringResult(
      const std::vector<StmtPatternPtrs>& stmts_list) {
    std::vector<LoopAlignableStmtsPattern> loop_alignable_list;
    for (const auto& stmt_ptrs : stmts_list) {
      loop_alignable_list.emplace_back(
          MakeLoopAlignableStmtsPattern(stmt_ptrs));
    }
    return ClusteringResult{
        .loop_alignable_list = std::move(loop_alignable_list),
    };
  }

 private:
  LoopAlignableStmtsPattern MakeLoopAlignableStmtsPattern(
      const std::vector<const StmtPattern*>& stmt_ptrs) {
    LoopAlignableStmtsPattern loop_alignable;
    loop_alignable.stmts.reserve(stmt_ptrs.size());
    for (const auto* stmt : stmt_ptrs) {
      loop_alignable.stmts.push_back(*stmt);
    }
    return loop_alignable;
  }

  bool IsTotalLoopSizeEqual(const StmtPattern& src, const StmtPattern& dst) {
    pir::Value src_value = GetStmtBigestShapeValue(src);
    pir::Value dst_value = GetStmtBigestShapeValue(dst);
    return shape_analysis_->IsProductEqual(
        src_value, 0, GetRank(src_value), dst_value, 0, GetRank(dst_value));
  }

  bool ReduceOpsSameShardable(const ShardableAxes4ValueT& ShardableAxes4Value,
                              const StmtPattern& src,
                              const StmtPattern& dst) {
    return std::visit(
        [&](const auto& src_impl, const auto& dst_impl) {
          return ReduceOpsSameShardableImpl(
              ShardableAxes4Value, src_impl, dst_impl);
        },
        src,
        dst);
  }

  template <typename SrcPatternT, typename DstPatternT>
  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const SrcPatternT& src,
      const DstPatternT& dst) {
    LOG(FATAL) << "Unimplemented. src_type: " << typeid(SrcPatternT).name()
               << ", dst_type: " << typeid(DstPatternT).name();
  }

  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const R& src,
      const PS& dst) {
    const auto* sink_op = src.reduce_op_pattern.reduce_op;
    pir::Value value =
        sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
    const auto& shardable_axes = ShardableAxes4Value(value);
    CHECK(shardable_axes.has_value());
    return IsStmtSinkOpOutputFullyShardableImpl(src, *shardable_axes.value());
  }

  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const R& src,
      const R& dst) {
    const auto GetSoleOutputValue = [&](const R& reduce_pattern) {
      const auto* sink_op = src.reduce_op_pattern.reduce_op;
      pir::Value value =
          sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
      return value;
    };
    const auto GetShardableAxes = [&](const R& reduce_pattern) {
      pir::Value value = GetSoleOutputValue(reduce_pattern);
      const auto& shardable_axes = ShardableAxes4Value(value);
      CHECK(shardable_axes.has_value());
      return shardable_axes.value();
    };
    const auto GetShardableAxesNames = [&](const R& reduce_pattern) {
      std::set<std::string> axis_names;
      for (const auto& shardable_axis : *GetShardableAxes(reduce_pattern)) {
        axis_names.insert(shardable_axis.axis_name);
      }
      return axis_names;
    };
    struct ShardibleAxisPair {
      std::optional<int> src_axis;
      std::optional<int> dst_axis;
    };
    const auto GetMatchedAxisPairs = [&]() {
      std::unordered_map<std::string, ShardibleAxisPair> matched_axis_pairs;
      for (const auto& src_sa : *GetShardableAxes(src)) {
        matched_axis_pairs[src_sa.axis_name].src_axis = src_sa.axis;
      }
      for (const auto& dst_sa : *GetShardableAxes(dst)) {
        matched_axis_pairs[dst_sa.axis_name].dst_axis = dst_sa.axis;
      }
      return matched_axis_pairs;
    };
    bool same_shardibility =
        (GetShardableAxesNames(src) == GetShardableAxesNames(dst));
    if (same_shardibility) {
      for (const auto& [axis_name, axis_pair] : GetMatchedAxisPairs()) {
        const auto& [src_axis, dst_axis] = axis_pair;
        CHECK(src_axis.has_value());
        CHECK(dst_axis.has_value());
        pir::Value src_value = GetSoleOutputValue(src);
        pir::Value dst_value = GetSoleOutputValue(dst);
        CHECK(shape_analysis_->IsProductEqual(
            src_value, {src_axis.value()}, dst_value, {dst_axis.value()}));
      }
    }
    return same_shardibility;
  }

  bool IsSinkOpOutputFullyShardable(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const StmtPattern& stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(stmt);
    pir::Value value =
        sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
    const auto& shardable_axes = ShardableAxes4Value(value);
    CHECK(shardable_axes.has_value());
    return IsStmtSinkOpOutputFullyShardable(stmt, *shardable_axes.value());
  }

  bool IsStmtSinkOpOutputFullyShardable(const StmtPattern& stmt,
                                        const ShardableAxes& shardable_axes) {
    return std::visit(
        [&](const auto& impl) {
          return IsStmtSinkOpOutputFullyShardableImpl(impl, shardable_axes);
        },
        stmt);
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const IS& injective_source, const ShardableAxes& shardable_axes) {
    return true;
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const PS& partial_shardable, const ShardableAxes& shardable_axes) {
    return true;
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const R& reduce_pattern, const ShardableAxes& shardable_axes) {
    const auto* reduce_op = reduce_pattern.reduce_op_pattern.reduce_op;
    if (reduce_op->isa<cinn::dialect::ReduceSumOp>()) {
      return IsCinnReduceSumOpOutputFullyShardable(reduce_op, shardable_axes);
    }
    LOG(FATAL) << "TODO(xiongkun). reduce_op name: " << reduce_op->name();
  }

  bool IsCinnReduceSumOpOutputFullyShardable(
      const pir::Operation* reduce_op, const ShardableAxes& shardable_axes) {
    const size_t input_rank = GetRank(reduce_op->operand_source(0));
    const auto& reduce_axes = GetReduceAxes(reduce_op);

    // no shardability if input reduced into one element.
    if (reduce_axes.empty()) return false;

    const auto& IsReduceAxis = [&](int axis) {
      return std::find(reduce_axes.begin(), reduce_axes.end(), axis) !=
             reduce_axes.end();
    };
    const auto& IsAxisSharded = [&](int axis) {
      const auto& Condition = [&](const auto& shardable_axis) {
        return shardable_axis.axis == axis;
      };
      return std::find_if(shardable_axes.begin(),
                          shardable_axes.end(),
                          Condition) != shardable_axes.end();
    };
    const bool keepdims = GetReduceOpKeepDims(reduce_op);
    if (keepdims) {
      const size_t output_rank = input_rank;
      CHECK(!reduce_axes.empty());
      for (int axis = 0; axis < output_rank; ++axis) {
        if (IsReduceAxis(axis)) continue;
        if (!IsAxisSharded(axis)) return false;
      }
      return true;
    } else {
      const int result_idx = GetOutputShardableAxesResultIdx(reduce_op);
      return GetRank(reduce_op->result(result_idx)) == shardable_axes.size();
    }
  }

  const pir::ShapeConstraintIRAnalysis* shape_analysis_;
};

}  // namespace

std::shared_ptr<ShardableAxesProvider> MakeDefaultShardableAxesProvider(
    const pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return std::make_shared<DefaultShardableAxesProvider>(shape_analysis);
}

std::shared_ptr<ClusteringPolicy> MakeLoopAlignableClusteringPolicy(
    const pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return std::make_shared<LoopAlignableClusteringPolicy>(shape_analysis);
}

ClusteringResult ClusterOps(
    const std::vector<const pir::Operation*>& ops,
    const std::shared_ptr<ShardableAxesProvider>& shardable_axes_provider,
    const std::shared_ptr<ClusteringPolicy>& clustering_policy) {
  VLOG(4) << "Initializing Inferer";
  ShardableAxesInferer inferer(shardable_axes_provider);
  VLOG(4) << "Initializing Clustering Engine";
  ClusteringEngine engine(ops, inferer, clustering_policy);
  VLOG(4) << "Engine calls ClusterOps()";
  return engine.ClusterOps();
}
}  // namespace cinn::frontend
