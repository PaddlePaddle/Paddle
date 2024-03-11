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
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#include <algorithm>
#include <optional>
#include <typeinfo>
#include <variant>

namespace cinn::frontend {

namespace {
using OpPatternKind = cinn::hlir::framework::OpPatternKind;

using IS = api::InjectiveSourcePattern<frontend::FrontendPattern>;
using R = api::ReductionPattern<frontend::FrontendPattern>;
using PS = api::PartialShardablePattern<frontend::FrontendPattern>;
using StmtPattern = api::StmtPattern<frontend::FrontendPattern>;
using StmtsPattern = api::StmtsPattern<frontend::FrontendPattern>;

using StmtPtr = StmtPattern*;
using OpVisitor = std::function<void(const pir::Operation*)>;
using NodeVisitor = std::function<void(StmtPtr)>;

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

void VisitInputOp(const pir::Operation* op, const OpVisitor& DoEach) {
  for (int i = 0; i < op->num_operands(); ++i) {
    const auto* input_op = op->operand_source(i).defining_op();
    DoEach(input_op);
  }
}

void VisitOutputOp(const pir::Operation* op, const OpVisitor& DoEach) {
  for (int i = 0; i < op->num_results(); ++i) {
    pir::Value output = op->result(i);
    for (auto consumer_it = output.use_begin(); consumer_it != output.use_end();
         ++consumer_it) {
      const auto* consumer_op = consumer_it->owner();
      if (consumer_op->isa<pir::YieldOp>()) continue;
      DoEach(consumer_op);
    }
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const IS& injective_source, const DoEachT& DoEach) {
  for (const auto* op : injective_source.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const R& reduce, const DoEachT& DoEach) {
  DoEach(reduce.reduction_op_pattern.reduce_op);
}

template <typename DoEachT>
void VisitStmtOpImpl(const PS& partial_shardable, const DoEachT& DoEach) {
  for (const auto* op : partial_shardable.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOp(const StmtPattern& stmt, const DoEachT& DoEach) {
  std::visit([&](const auto& impl) { VisitStmtOpImpl(impl, DoEach); }, stmt);
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInThisFusionOp(
    const std::vector<pir::Operation*>& ops) {
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
    const std::vector<pir::Operation*>& ops,
    const std::function<bool(const pir::Operation*)>& IsInThisOpList) {
  const auto& IsSource = [&](const pir::Operation* op) {
    std::size_t num_inputs = 0;
    VisitInputOp(op, [&](const pir::Operation* input) {
      if (IsInThisOpList(input)) {
        ++num_inputs;
      }
    });
    return num_inputs == 0;
  };

  const auto starts = [&] {
    std::list<const pir::Operation*> starts;
    for (const auto* op : ops) {
      if (!IsInThisOpList(op) && IsSource(op)) {
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
    VisitInputOp(op, [&](const pir::Operation* input) {
      if (IsInThisOpList(input)) {
        is_inputs_all_injective_source = (is_inputs_all_injective_source &&
                                          op_2_is_injective_source.at(input));
      }
    });
    return is_inputs_all_injective_source;
  };

  common::TopoWalker<const pir::Operation*> walker{VisitInputOp, VisitOutputOp};
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

size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

ShardableAxesSignature MakeShardableAxesSignature4ElementWiseOp(
    const pir::Operation* op) {
  CHECK(!op->isa<cinn::dialect::ReshapeOp>())
      << "reshape not supported. TODO(wuzhanfei).";
  const size_t rank = [&] {
    std::optional<size_t> rank;
    for (int i = 0; i < op->num_operands(); ++i) {
      if (rank.has_value()) {
        CHECK_EQ(rank.value(), GetRank(op->operand_source(i)));
      } else {
        rank = GetRank(op->operand_source(i));
      }
    }
    CHECK_EQ(op->num_results(), 1);
    if (rank.has_value()) {
      CHECK_EQ(rank.value(), GetRank(op->result(0)));
    } else {
      rank = GetRank(op->result(0));
    }
    CHECK(rank.has_value());
    return rank.value();
  }();
  const ShardableAxes output_shardable_axes =
      ShardableAxesUtil::GetFullyShardableAxes(rank);
  std::unordered_map<OpAndOperandIndex, ShardableAxes> input_shardable_axes;
  for (int i = 0; i < op->num_operands(); ++i) {
    input_shardable_axes[OpAndOperandIndex{op, i}] = output_shardable_axes;
  }
  return ShardableAxesSignature{
      .output_shardable_axes = output_shardable_axes,
      .input_shardable_axes = input_shardable_axes,
  };
}

ShardableAxesSignature MakeShardableAxesSignature4BroadcastOp(
    const pir::Operation* op) {
  LOG(FATAL) << "TODO(wuzhanfei).";
}

ShardableAxesSignature MakeShardableAxesSignature4Op(const pir::Operation* op) {
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  if (kind == hlir::framework::kElementWise) {
    return MakeShardableAxesSignature4ElementWiseOp(op);
  } else if (kind == hlir::framework::kBroadcast) {
    return MakeShardableAxesSignature4BroadcastOp(op);
  } else {
    LOG(FATAL)
        << "only kReduction, kElementWise, kBroadcast supported. op_name:"
        << op->name();
  }
  LOG(FATAL) << "Dead code";
}

template<typename InputIt>
std::unordered_map<pir::Value, ShardableAxes> ReversedInferShardableAxes(
    const common::TopoWalker<const pir::Operation*>& reversed_walker,
    InputIt sink_and_init_begin, InputIt sink_and_init_end) {
  std::unordered_map<pir::Value, ShardableAxes> value2shardable_axes;
  std::list<const pir::Operation*> sinks;
  for (auto iter = sink_and_init_begin; iter != sink_and_init_end; ++iter) {
    sinks.push_back(iter->first.defining_op());
    value2shardable_axes[iter->first] = iter->second;
  }
  const auto& UpdateValue2ShardableAxes = [&](pir::Value value, const ShardableAxes& sa) {
    auto iter = value2shardable_axes.find(value);
    if (iter != value2shardable_axes.end()) {
      iter->second =
          ShardableAxesUtil::GetCommonShardableAxes(iter->second, sa);
    } else {
      iter->second = sa;
    }
  };
  reversed_walker(sinks.begin(), sinks.end(), [&](const auto* op) {
    auto shardable_axes_sig = MakeShardableAxesSignature4Op(op);
    const auto& old2new = ShardableAxesUtil::GetOldName2NewName(
        shardable_axes_sig.output_shardable_axes,
        value2shardable_axes.at(op->result(0)));
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
  CHECK_EQ(sink->num_results(), 1);
  std::array<OpAndInitValue, 1> sinks{OpAndInitValue{sink->result(0), init_sa}};
  return ReversedInferShardableAxes(reversed_walker, sinks.begin(), sinks.end());
}

common::TopoWalker<const pir::Operation*> GetOpsTopoWalker(const std::unordered_set<const pir::Operation*>& ops) {
  const auto* ops_set = &ops;
  const auto VisitUpStreamInOps = [ops_set](const pir::Operation* op,
                                            const OpVisitor& DoEach) {
    VisitInputOp(op, [&](const auto* input) {
      if (ops_set->count(input) == 0) return;
      DoEach(input);
    });
  };
  const auto VisitDownStreamInOps = [ops_set](const pir::Operation* op,
                                              const OpVisitor& DoEach) {
    VisitOutputOp(op, [&](const auto* output) {
      if (ops_set->count(output) == 0) return;
      DoEach(output);
    });
  };
  common::TopoWalker<const pir::Operation*> reversed_walker(
      VisitDownStreamInOps, VisitUpStreamInOps);
  return reversed_walker;
}

std::list<const pir::Operation*> GetSinks(
    const std::unordered_set<const pir::Operation*>& ops) {
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

std::unordered_map<const pir::Operation*, ShardableAxesSignature>
GetOp2ShardableAxesSignature(const std::unordered_set<const pir::Operation*>& ops) {
  std::unordered_map<const pir::Operation*, ShardableAxesSignature> ret;
  for (const auto* op : ops) {
    ret[op] = MakeShardableAxesSignature4Op(op);
  }
  return ret;
}

std::map<std::string, std::vector<std::string>>
GetAxisName2BoundAxisName(
    const std::unordered_set<const pir::Operation*>& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>& op2shardable_axes_signature) {
  const auto GetInputShardableAxes = [&](const OpAndOperandIndex& op_and_idx) -> std::optional<const ShardableAxes*> {
    const auto& [op, idx] = op_and_idx;
    const auto* input_op = op->operand_source(idx).defining_op();
    if (ops.count(input_op) == 0) return std::nullopt;
    const auto& iter = op2shardable_axes_signature.find(input_op);
    if (iter == op2shardable_axes_signature.end()) return std::nullopt;
    const auto& output_sa = iter->second.output_shardable_axes;
    return &output_sa;
  };
  std::map<std::string, std::vector<std::string>> axis_name2bound_axis_name;
  const auto UpdateAxisName2BoundAxisName = [&](const ShardableAxes& input_sa, const ShardableAxes& sa) {
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
GetAxisName2UnionFindSetRoot(
    const std::unordered_set<const pir::Operation*>& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>& op2shardable_axes_signature) {
  const auto axis_name2bound_axis_name = GetAxisName2BoundAxisName(ops, op2shardable_axes_signature);
  using NodeVisitor = std::function<void(const std::string&)>;
  const auto VisitNext = [&](const std::string& axis_name, const NodeVisitor& DoEach) {
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
    walk(union_find_root, [&](const std::string& axis_name){
      CHECK(axis_name2root.emplace(axis_name, union_find_root).second);
    });
  }
  return axis_name2root;
}

std::unordered_map<pir::Value, ShardableAxes>
GetSinkAndInitShardableAxes(
    const std::list<const pir::Operation*>& sinks,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>& op2shardable_axes_signature,
    const std::unordered_map<std::string, std::string>& axis_name2union_find_set_root) {
  const auto& ConvertByBoundAxisName = [&](const ShardableAxes& sa) {
    ShardableAxes ret_sa;
    for (const auto& [axis, axis_name] : sa) {
      const auto& iter = axis_name2union_find_set_root.find(axis_name);
      CHECK(iter != axis_name2union_find_set_root.end());
      ret_sa.emplace_back(ShardableAxis{
        .axis=axis,
        .axis_name=iter->second,
      });
    }
    return ret_sa;
  };
  std::unordered_map<pir::Value, ShardableAxes> sink2sa;
  for (const auto* sink : sinks) {
    const auto& sig_iter = op2shardable_axes_signature.find(sink);
    CHECK(sig_iter != op2shardable_axes_signature.end());
    const auto& output_shardable_axes = sig_iter->second.output_shardable_axes;
    CHECK_EQ(sink->num_results(), 1);
    sink2sa[sink->result(0)] = ConvertByBoundAxisName(output_shardable_axes);
  }
  return sink2sa;
}

void RenameDuplicatedAxisName(std::unordered_map<pir::Value, ShardableAxes>* sink2sa) {
  const auto& RenameDuplicated = [&](ShardableAxes* sa) {
    std::set<std::string> existed_axis_name;
    for (auto& [_, axis_name] : *sa) {
      if (!existed_axis_name.emplace(axis_name).second) {
        axis_name = axis_name + "_" + std::to_string(ShardableAxis::UnqiueSeqNo());
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
    const std::unordered_set<const pir::Operation*>& ops,
    const std::list<const pir::Operation*>& sinks) {
  const auto& op2shardable_axes_signature = GetOp2ShardableAxesSignature(ops);
  const auto& axis_name2union_find_set_root = GetAxisName2UnionFindSetRoot(ops, op2shardable_axes_signature);
  std::unordered_map<pir::Value, ShardableAxes> sink_and_inits =
      GetSinkAndInitShardableAxes(sinks, op2shardable_axes_signature, axis_name2union_find_set_root);
  RenameDuplicatedAxisName(&sink_and_inits);
  return sink_and_inits;
}

class StmtFusionHelper {
 public:
  explicit StmtFusionHelper(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {
    this->IsInThisOpList = MakePredicatorIsInThisFusionOp(ops);
    this->IsInjectiveSource =
        MakePredicatorIsInjectiveSource(ops_, this->IsInThisOpList);
  }

  std::vector<StmtPattern> ConvertToStmtsPattern() {
    std::vector<StmtPattern> ret;
    for (const auto* op : ops_) {
      if (!IsInThisOpList(op)) continue;
      ret.emplace_back(ConvertToStmtPattern(op));
    }
    return ret;
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(
      std::vector<StmtPattern>* stmt_patterns) {
    const auto ConstructISPattern = [&](const auto& ops) { return IS{ops}; };
    return MultiFuse(IsISPattern, ConstructISPattern, stmt_patterns);
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns) {
    const auto ConstructPSPattern = [&](const auto& ops) {
      const auto shardable_axes_signature = GetShardableAxesSignature(ops);
      return PS{
          .ops = ops,
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
        std::vector<const pir::Operation*> ops;
        ops.insert(ops.end(), upstream.ops.begin(), upstream.ops.end());
        ops.insert(ops.end(), downstream.ops.begin(), downstream.ops.end());
        std::unique(ops.begin(), ops.end());
        return ops;
      }();
      const auto& shardable_axes_signature =
          MergeShardableAxesSignature(upstream, downstream);
      return StmtPattern(PS{
          .ops = ops,
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
            .ops = {downstream.reduction_op_pattern.reduce_op},
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
            .ops = {downstream.reduction_op_pattern.reduce_op},
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

 private:
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

  IS ConvertToIS(const pir::Operation* op) { return IS{{op}}; }

  R ConvertReductionOpToReductionPattern(const pir::Operation* op) {
    return R{{}, {op}};
  }

  PS ConvertOpToPS(const pir::Operation* op) {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    return PS{
        .ops = {op},
        .shardable_axes_signature = MakeShardableAxesSignature4Op(op),
    };
  }

  using StmtPtr4OpT =
      std::function<std::optional<StmtPtr>(const pir::Operation*)>;
  static StmtPtr4OpT MakeStmtFinderFromOp(std::vector<StmtPattern>* stmts) {
    std::unordered_map<const pir::Operation*, StmtPtr> op2stmt_ptr;
    for (auto& stmt : *stmts) {
      VisitStmtOp(stmt, [&](const auto* op) { op2stmt_ptr[op] = &stmt; });
    }
    return [map = std::move(op2stmt_ptr)](
               const pir::Operation* op) -> std::optional<StmtPtr> {
      const auto iter = map.find(op);
      if (iter == map.end()) return std::nullopt;
      return iter->second;
    };
  }

  std::function<size_t(const pir::Operation*)> MakeTopoOrderFinderOfOp(
      const std::vector<pir::Operation*>& ops) {
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

  template <typename IsChozenPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsChozenPatternT& IsChozenPattern,
      const ConstructPatternT& ConstructPattern,
      std::vector<StmtPattern>* stmts) {
    const auto StmtFinder = MakeStmtFinderFromOp(stmts);
    const auto VisitInputStmt = [&](StmtPtr stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            if (IsChozenPattern(*input_stmt.value())) {
              DoEach(input_stmt.value());
            }
          }
        });
      });
    };
    const auto VisitOutputStmt = [&](StmtPtr stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        VisitOutputOp(op, [&](const pir::Operation* output) {
          if (const auto& output_stmt = StmtFinder(output)) {
            if (IsChozenPattern(*output_stmt.value())) {
              DoEach(output_stmt.value());
            }
          }
        });
      });
    };
    const auto IsSinkPattern = [&](StmtPtr stmt) {
      if (!IsChozenPattern(*stmt)) return false;
      std::size_t num_injective_src_outputs = 0;
      VisitOutputStmt(stmt, [&](const auto& consumer) {
        num_injective_src_outputs += IsChozenPattern(*consumer);
      });
      return num_injective_src_outputs == 0;
    };
    const auto GetOrder = MakeTopoOrderFinderOfOp(ops_);
    const auto Cmp = [&](const auto* lhs, const auto& rhs) {
      return GetOrder(lhs) < GetOrder(rhs);
    };
    common::BfsWalker<StmtPtr> reverse_walker(VisitInputStmt);
    const auto& GetUpstreamOps = [&](const auto stmt_ptr) {
      std::vector<const pir::Operation*> visited_ops;
      reverse_walker(stmt_ptr, [&](const auto node) {
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
      ret_stmts.emplace_back(ConstructPattern(GetUpstreamOps(&stmt)));
    }
    *stmts = ret_stmts;
    return std::nullopt;
  }

  struct StmtIterPair {
    std::list<StmtPtr>::iterator upstream_iter;
    std::list<StmtPtr>::iterator downstream_iter;
  };

  bool IsConnected(const StmtPtr4OpT& StmtFinder,
                   const StmtPtr& upstream,
                   const StmtPtr& downstream) {
    const auto VisitInputStmt = [&](StmtPtr stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            DoEach(input_stmt.value());
          }
        });
      });
    };

    bool found = false;
    VisitInputStmt(downstream, [&](const StmtPtr& input_pattern) {
      if (input_pattern == upstream) {
        found = true;
      }
    });
    return found;
  }

  template <typename FuseTargetConditionT>
  std::optional<StmtIterPair> FindConnetedPattenPairWithCondition(
      const StmtPtr4OpT& StmtFinder,
      std::list<StmtPtr>* stmt_ptrs,
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
    std::list<StmtPtr> stmts_iters = [&] {
      std::list<StmtPtr> stmts_iters;
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

  ShardableAxesSignature GetShardableAxesSignature(
      const std::vector<const pir::Operation*>& ops) {
    std::unordered_set<const pir::Operation*> ops_set(ops.begin(), ops.end());
    const pir::Operation* sink = [&] {
      const auto& sinks = GetSinks(ops_set);
      CHECK_EQ(sinks.size(), 1) << "ops must have only one sink node.";
      return *sinks.begin();
    }();
    const auto& value2shardable_axes =
        InferShardableAxesFromSink(sink, ops_set);
    const auto& IsInputOpOperand = [&](const auto* op, int input_idx) {
      const auto& defining_op = op->operand_source(input_idx).defining_op();
      return IsInThisOpList(defining_op) && ops_set.count(defining_op) == 0;
    };
    const auto& input_op_operands = [&] {
      std::vector<OpAndOperandIndex> op_operands;
      for (const auto* op : ops) {
        for (int i = 0; i < op->num_operands(); ++i) {
          if (!IsInputOpOperand(op, i)) continue;
          op_operands.emplace_back(OpAndOperandIndex{op, i});
        }
      }
      return op_operands;
    }();
    const auto& shardable_axes_sig = [&] {
      ShardableAxesSignature signature;
      signature.output_shardable_axes =
          value2shardable_axes.at(sink->result(0));
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
  std::vector<pir::Operation*> ops_;
  std::function<bool(const pir::Operation*)> IsInThisOpList;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
};

GroupPattern FuseToGroupPattern(const std::vector<pir::Operation*>& ops) {
  StmtFusionHelper helper(ops);
  std::vector<StmtPattern> stmt_patterns = helper.ConvertToStmtsPattern();
  if (const auto& error = helper.Fuse_IS_x_IS_2_IS(&stmt_patterns))
    return error.value();
  if (const auto& error = helper.Fuse_PS_x_PS_2_PS(&stmt_patterns))
    return error.value();
  if (const auto& error = helper.Fuse_IS_x_PS_2_PS(&stmt_patterns))
    return error.value();
  if (const auto& error = helper.Fuse_IS_x_R_2_R(&stmt_patterns))
    return error.value();
  if (const auto& error = helper.Fuse_PS_x_R_2_R(&stmt_patterns))
    return error.value();
  return stmt_patterns;
}

class ClusteringHelper {
 public:
  ClusteringHelper(
      const pir::ShapeConstraintIRAnalysis* shape_analysis,
      const std::vector<pir::Operation*>& ops,
      const OpsClusteringSpec& clustering_spec)
    : shape_analysis_(shape_analysis), ops_(ops), clustering_spec_(clustering_spec) {
    this->IsInThisOpList = MakePredicatorIsInThisFusionOp(ops);
    this->IsInjectiveSource =
        MakePredicatorIsInjectiveSource(ops_, this->IsInThisOpList);
  }

  std::vector<ConditionalGroupPattern> ClusterIntoGroupPatterns() {
    LOG(FATAL) << "TODO(tianchao)";
  }

 private:
  const pir::ShapeConstraintIRAnalysis* shape_analysis_;
  const std::vector<pir::Operation*> ops_;
  const OpsClusteringSpec clustering_spec_;
  std::function<bool(const pir::Operation*)> IsInThisOpList;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
};

}  // namespace

std::vector<ConditionalGroupPattern> ClusterIntoGroupPatternsFromOpList(
    const pir::ShapeConstraintIRAnalysis* shape_analysis,
    const std::vector<pir::Operation*>& ops,
    const OpsClusteringSpec& clustering_spec) {
  ClusteringHelper helper(shape_analysis, ops, clustering_spec);
  return helper.ClusterIntoGroupPatterns();
}

GroupPattern GenerateGroupPatternFromOpList(
    const std::vector<pir::Operation*>& ops) {
  return FuseToGroupPattern(ops);
}

std::unordered_map<pir::Value, ShardableAxes> InferShardableAxesFromSink(
    const pir::Operation* sink,
    const std::unordered_set<const pir::Operation*>& ops) {
  auto reversed_walker = GetOpsTopoWalker(ops);
  CHECK_GT(ops.count(sink), 0);
  size_t rank = GetRank(sink->result(0));
  const auto& init_sa = ShardableAxesUtil::GetFullyShardableAxes(rank);
  return ReversedInferShardableAxes(reversed_walker, sink, init_sa);
}

std::unordered_map<pir::Value, ShardableAxes> InferShardableAxes(
    const std::unordered_set<const pir::Operation*>& ops) {
  auto reversed_walker = GetOpsTopoWalker(ops);
  const auto& sinks = GetSinks(ops);
  const auto& sink_and_init_value = GetSinkAndInitValues(reversed_walker, ops, sinks);
  return ReversedInferShardableAxes(reversed_walker, sink_and_init_value.begin(), sink_and_init_value.end());
}

}  // namespace cinn::frontend
