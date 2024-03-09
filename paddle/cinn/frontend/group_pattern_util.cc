#include "paddle/cinn/frontend/group_pattern_util.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/framework/op.h"
#include <optional>
#include <typeinfo>
#include <algorithm>
#include <variant>

namespace cinn::frontend {

namespace {
using OpPatternKind = cinn::hlir::framework::OpPatternKind;

using StmtIter = std::list<StmtPattern>::iterator;
using OpVisitor = std::function<void(const pir::Operation*)>;
using NodeVisitor = std::function<void(StmtIter)>;


OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

bool IsGeneralInjective(const pir::Operation* op) {
  hlir::framework::OpPatternKind op_pattern_kind = GetOpPatternKind(op);
  return op_pattern_kind == hlir::framework::kElementWise
    || op_pattern_kind == hlir::framework::kBroadcast
    || op_pattern_kind == hlir::framework::kInjective;
}

bool IsISPattern(StmtPattern& pattern){
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern){
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern){
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
    for (auto consumer_it = output.use_begin(); consumer_it != output.use_end(); ++consumer_it) {
      const auto* consumer_op = consumer_it->owner();
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
  DoEach(reduce.reduce_op);
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

std::function<bool(const pir::Operation*)> MakePredicatorIsInThisFusionOp(const cinn::dialect::FusionOp& fusion_op) {
  std::set<pir::Operation*> set;
  for (const pir::Operation* op : fusion_op.block()->ops()) {
    if (!op->isa<pir::YieldOp>()) {
      set.insert(op);
    }
  }
  return [set = std::move(set)](const pir::Operation* op) {
    return set.count(op) > 0;
  };
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const cinn::dialect::FusionOp& fusion_op,
    const std::function<bool(const pir::Operation*)>& IsInThisFusionOp) {

  const auto& IsSource = [&](const pir::Operation* op) {
    std::size_t num_inputs = 0;
    VisitInputOp(op, 
      [&](const pir::Operation* input) { 
        if(IsInThisFusionOp(input)){
          ++num_inputs;
        }
      }
    );
    return num_inputs == 0;
  };

  const auto starts = [&]{
    std::list<const pir::Operation*> starts;
    for (const auto* op : fusion_op.GetOperators()) {
      if (!IsInThisFusionOp(op) && IsSource(op)) {
        starts.push_back(op);
      } else {
        // do nothing.
      }
    }
    return starts;
  }();

  std::unordered_map<pir::Operation*, bool> op_2_is_injective_source;

  auto IsInputsAllInjectiveSource = [&](const pir::Operation* op) {
    bool is_inputs_all_injective_source = true;
    VisitInputOp(op, 
      [&](const pir::Operation* input){
        if (IsInThisFusionOp(input)){
          is_inputs_all_injective_source = (is_inputs_all_injective_source && op_2_is_injective_source.at(input));
        }
      }
    );
    return is_inputs_all_injective_source;
  };

  common::TopoWalker<const pir::Operation*> walker{VisitEachInput, VisitEachOutput};
  walker(starts, [&](const pir::Operation* op){
    op_2_is_injective_source[op] = (IsGeneralInjective(op) && IsInputsAllInjectiveSource(op));
  });
  return [map = std::move(op_2_is_injective_source)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

class StmtFusionHelper {
 public:
  explicit StmtFusionHelper(const cinn::dialect::FusionOp& fusion_op)
     : fusion_op_(fusion_op) {
    this->IsInThisFusionOp = MakePredicatorIsInThisFusionOp(fusion_op_);
    this->IsInjectiveSource = MakePredicatorIsInjectiveSource(fusion_op_, this->IsInThisFusionOp);
  }

  std::list<StmtPattern> ConvertToStmtsPattern() const {
    std::list<StmtPattern> ret;
    for (const auto* op : fusion_op_.GetOperators()) {
      if (!IsInThisFusionOp(op)) continue;
      ret.emplace_back(ConvertToStmtPattern(op));
    }
    return ret;
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(std::list<StmtPattern>* stmts) const {
    const auto ConstructISPattern = [&](const auto& ops) { return IS{ops}; };
    return MultiFuse(IsISPattern, ConstructISPattern, stmts);
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::list<StmtPattern>* stmt_patterns) const {
    const auto ConstructPSPattern = [&](const auto& ops) {
      const auto shardable_axes_signature = GetShardableAxesSignature(ops);
      return PS{
        .ops=ops,
        .shardable_axes_signature=shardable_axes_signature,
      };
    };
    return MultiFuse(IsPSPattern, ConstructISPattern, stmts);
  }

  struct FusePolicy_IS_x_PS_2_PS {
    static bool FuseCondition(const StmtPattern& upstream, const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsPSPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<IS>(upstream), std::get<PS>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream,
        const PS& downstream) {
      const auto& ops = [&]{
        std::vector<const pir::Operation*> ops;
        ops.insert(ops.end(), upstream.ops.begin(), upstream.ops.end());
        ops.insert(ops.end(), downstream.ops.begin(), downstream.ops.end());
        std::unique(ops.begin(), ops.end());
        return ops;
      }();
      const auto& shardable_axes_signature = MergeShardableAxesSignature(upstream, downstream);
      return PS{
        .ops=ops,
        .shardable_axes_signature=shardable_axes_signature,
      };
    }
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(std::list<StmtPattern>* stmt_patterns) const { 
    return FuseFilteredStmtPatterns<FusePolicy_IS_x_PS_2_PS>(stmt_patterns);
  }
  struct FusePolicy_IS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream, const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsRPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<IS>(upstream), std::get<R>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream,
        const R& downstream) {
      if (downstream.opt_inputs.has_value()) {
        return ErrorGroupPattern{
          .ops={downstream.reduction_op_pattern.reduce_op},
          .error_string="The input of reduce has been fused.",
        };
      }
      R new_pattern = R(downstream);
      new_pattern.opt_inputs = upstream;
      return new_pattern;
    }
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(std::list<StmtPattern>* stmt_patterns) const {
    return FuseFilteredStmtPatterns<FusePolicy_IS_x_R_2_R>(stmt_patterns);
  }

  struct FusePolicy_PS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream, const StmtPattern& downstream) {
      return IsISPattern(upstream) && IsRPattern(downstream);
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream) {
      return MergePatternImpl(std::get<PS>(upstream), std::get<R>(downstream));
    }
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const PS& upstream,
        const R& downstream) {
      if (downstream.opt_inputs.has_value()) {
        return ErrorGroupPattern{
          .ops={downstream.reduction_op_pattern.reduce_op},
          .error_string="The input of reduce has been fused.",
        };
      }
      R new_pattern = R(downstream);
      new_pattern.opt_inputs = upstream;
      return new_pattern;
    }
  };

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(std::list<StmtPattern>* stmt_patterns) const {
    return FuseFilteredStmtPatterns<FusePolicy_PS_x_R_2_R>(stmt_patterns);
  }

 private:

  StmtPattern ConvertToStmtPattern(const pir::Operation* op) const {
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
      LOG(FATAL) << "only kReduction, kElementWise, kBroadcast supported. op_name:" << op->op_name(); 
    }
    LOG(FATAL) << "Dead code";
  }

  IS ConvertToIS(const pir::Operation* op) const {
    return IS{{op}};
  }

  R ConvertReductionOpToReductionPattern(const pir::Operation* op) const {
    return R{{}, {op}};
  }

  PS ConvertOpToPS(const pir::Operation* op) const {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    return PS{
      .ops={op},
      .shardable_axes_signature=MakeShardableAxesSignature4Op(op),
    };
  }

  static std::function<std::optional<StmtIter>(const pir::Operation*)>
  MakeStmtFinderFromOp(std::list<StmtPattern>* stmts) {
    std::unordered_map<const pir::Operation*, StmtIter> op2stmt_iter;
    for (auto iter = stmts->begin(); iter != stmts->end(); ++iter) {
      VisitStmtOp(*iter, [&](const auto* op) { op2stmt_iter[op] = iter; });
    }
    return [map=std::move(op2stmt_iter)](const pir::Operation* op) -> std::optional<StmtIter> {
      const auto iter = map.find(op);
      if (iter == map.end()) return std::nullopt;
      return iter->second;
    };
  }

  std::function<size_t(const pir::Operation*)> MakeTopoOrderFinderOfOp(cinn::dialect::FusionOp& fusion_op) const {
    std::unordered_map<pir::Operation*, size_t> op2order_in_block;
    size_t order = 0;
    for (const pir::Operation* op : fusion_op.GetOperators()) {
      op2order_in_block[op] = ++order;
    }
    return [map=std::move(op2order_in_block)](const pir::Operation* op) {
      const auto& iter = map.find(op);
      CHECK(iter != map.end());
      return iter->second;
    };
  }

  template<typename IsDetailPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsDetailPatternT& IsDetailPattern,
      const ConstructPatternT& ConstructPattern,
      std::list<StmtPattern>* stmts) const {
    const auto StmtFinder = MakeStmtFinderFromOp(stmts);

    const auto VisitInputStmt = [&](StmtIter stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op){
        VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            if (IsDetailPattern(input_stmt->value())) {
              DoEach(input_stmt.value());
            }
          }
        });
      });
    };
    const auto VisitOutputStmt = [&](StmtIter stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op){
        VisitOutputOp(op, [&](const pir::Operation* output) {
          if (const auto& output_stmt = StmtFinder(output)) {
            if (IsDetailPattern(*output_stmt.value())) {
              DoEach(output_stmt.value());
            }
          }
        });
      });      
    };
    const auto IsSinkPattern = [&](StmtIter stmt) {
      if (!IsDetailPattern(*stmt)) return false;
      std::size_t num_injective_src_outputs = 0;
      VisitOutputStmt(node, [&](const auto& consumer) {
        num_injective_src_outputs += IsDetailPattern(*consumer);
      });
      return num_injective_src_outputs == 0;
    };
    const auto GetOrder = MakeTopoOrderFinderOfOp(fusion_op_);
    const auto Cmp = [&](const auto* lhs, const auto& rhs) {
      return GetOrder(lhs) < GetOrder(rhs);
    };
    common::BfsWalker<StmtIter> reverse_walker(VisitInputStmt);
    const auto& GetUpstreamOps = [&](const auto stmt_iter) {
      std::vector<const pir::Operation*> visited_ops;
      reverse_walker(start, [&](const auto node){
        VisitStmtOp(node, [&](const auto* op) { visited_ops.push_back(op); });
      });
      std::sort(visited_ops.begin(), visited_ops.end(), Cmp);
      return visited_ops;
    };
    std::list<StmtPattern> fused_stmts;
    for (auto stmt_iter = stmts->begin(); stmt_iter != stmts->end(); ++stmt_iter) {
      if (!IsSinkPattern(stmt_iter)) continue;
      fused_stmts.emplace_back(ConstructPattern(GetUpstreamOps(stmt_iter)));
    }
    for (auto stmt_iter = stmts->begin(); stmt_iter != start->end();) {
      if (IsDetailPattern(*stmt_iter)) {
        stmt_iter = stmts->erase(stmt_iter);
      } else {
        ++stmt_iter;
      }
    }
    stmts->splice(stmts->begin(), std::move(fused_stmts));
    return std::nullopt;
  }

  size_t GetRank(pir::Value value) const {
    return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
  };

  ShardableAxesSignature MakeShardableAxesSignature4Op(const pir::Operation* op) const {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (kind == hlir::framework::kElementWise) {
      return MakeShardableAxesSignature4ElementWiseOp(op);
    } else if (kind == hlir::framework::kBroadcast) {
      return MakeShardableAxesSignature4BroadcastOp(op);
    } else {
      LOG(FATAL) << "only kReduction, kElementWise, kBroadcast supported. op_name:" << op->op_name(); 
    }
    LOG(FATAL) << "Dead code";
  }

  ShardableAxesSignature MakeShardableAxesSignature4ElementWiseOp(const pir::Operation* op) const {
    CHECK(!op->isa<cinn::dialect::ReshapeOp>()) << "reshape not supported. TODO(wuzhanfei).";
    const size_t rank = [&]{
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
    const ShardableAxes shardable_axes = ShardableAxesUtil::GetFullyShardableAxes(rank);
    std::unordered_map<OpOperand, ShardableAxes> input_shardable_axes;
    for (int i = 0; i < op->num_operands(); ++i) {
      input_shardable_axes[std::pair(op, i)] = shardable_axes;
    }
    return ShardableAxesSignature{
      .output_shardable_axes,
      .input_shardable_axes=input_shardable_axes,
    };
  }

  ShardableAxesSignature MakeShardableAxesSignature4BroadcastOp(const pir::Operation* op) const {
    LOG(FATAL) << "TODO(wuzhanfei).";
  }

  struct StmtIterPair {
    StmtIter upstream_iter;
    StmtIter downstream_iter;
  };

  bool IsConnected(const StmtIter& upstream, const StmtIter& downstream){
    const auto StmtFinder = MakeStmtFinderFromOp({*upstream, *downstream});
    const auto VisitInputStmt = [&](StmtIter stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op)){
        VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            if (IsDetailPattern(input_stmt->value())) {
              DoEach(input_stmt.value());
            }
          }
        });
      };
    };

    auto downstream_input_patterns = std::unordered_set<StmtIter>();
    VisitInputStmt(*downstream, [&](const StmtIter& input_pattern){
      downstream_input_patterns.insert(input_pattern);
    })

    return downstream_input_patterns.count(upstream) > 0;
  }

  template <typename FuseTargetConditionT>
  std::optional<StmtIterPair> FindConnetedPattenPairWithCondition(
      std::list<StmtPattern>* stmt_patterns,
      const FuseTargetConditionT& FuseTargetCondition) const {
    for (auto dst_iter = stmt_patterns->begin(); dst_iter != stmt_patterns->end(); ++dst_iter) {
      for (auto src_iter = stmt_patterns->begin(); src_iter != stmt_patterns->end(); ++src_iter) {
        if (src_iter == dst_iter) continue;
        if (!IsConnected(*src_iter, *dst_iter)) continue;
        if (FuseTargetCondition(*src_iter, *dst_iter)) {
          return StmtPattern{
            .upstream_iter=src_iter,
            .downstream_iter=dst_iter,
          }
        }
      }
    }
    return std::nullopt;
  }

  template <typename FusionPolicy>
  std::optional<ErrorGroupPattern> FuseFilteredStmtPatterns(
      std::list<StmtPattern>* stmt_patterns) const{
    while(true){
      const auto& pattern_pair = FindConnetedPattenPairWithCondition(
        stmt_patterns, &FusionPolicy::FuseCondition);
      if (!pattern_pair.value()) break;
      const std::variant<StmtPattern, ErrorGroupPattern>& new_pattern = 
        FusionPolicy::MergePattern(*pattern_pair.value().upstream_iter, *pattern_pair.value().downstream_iter);

      if (std::holds_alternative<ErrorGroupPattern>(new_pattern)){
        return std::get<ErrorGroupPattern>(new_pattern);
      }
      stmt_patterns->erase(pattern_pair.value().upstream_iter);
      stmt_patterns->erase(pattern_pair.value().downstream_iter);
      stmt_patterns->emplace_back(std::get<StmtPattern>(new_pattern));
    }
    return std::nullopt;
  }

  ShardableAxesSignature GetShardableAxesSignature(const std::vector<const pir::Operation*>& ops) const {
    std::unordered_set<const pir::Operation*> ops_set(ops.begin(), ops.end());
    const auto VisitUpStreamInOps = [&](const pir::Operation* op, const OpVisitor& DoEach) {
      VisitInputOp(op, [&](const auto* input){
        if (ops_set.count(input) == 0) return;
        DoEach(input);
      });
    };
    const auto VisitDownStreamInOps = [&](const pir::Operation* op, const OpVisitor& DoEach) {
      VisitOutputOp(op, [&](const auto* output){
        if (ops_set.count(output) == 0) return;
        DoEach(output);
      });
    };
    const auto IsSinkOp = [&](const pir::Operation* op) {
      size_t num_donwstreams = 0;
      VisitDownStreamInOps(op, [&](const auto*){  ++num_donwstreams; });
      return num_donwstreams == 0;
    };
    const pir::Operation* sink = [&]{
      std::optional<const pir::Operation*> sink;
      for (const auto* op : ops) {
        if (IsSinkOp(op)) {
          CHECK(!sink.has_value()) << "only one sink node.";
        }
        sink = op;
      }
      CHECK(sink.has_value());
      return sink.value();
    }();
    const auto& value2shardable_axes = [&]{
      common::TopoWalker<const pir::Operation*> reversed_walker(VisitDownStreamInOps, VisitUpStreamInOps);
      size_t rank = GetRank(sink->result(0));
      const auto& init_sa = ShardableAxesUtil::GetFullyShardableAxes(rank);
      return ReversedInferShardableAxes(reversed_walker, sink, init_sa);
    }();
    const auto& IsInputOpOperand = [&](const auto* op, int input_idx) {
      const auto& defining_op = op->operand_source(input_idx)->defining_op();
      return IsInThisFusionOp(defining_op) && ops_set.count(defining_op) == 0;
    };
    using OpOperandT = std::pair<const std::Operation*, /*input index*/int>;
    const auto& input_op_operands = [&]{
      std::vector<OpOperandT> op_operands;
      for (const auto* op : ops) {
        for (int i = 0; i < op->num_operands(); ++i) {
          if (!IsInputOpOperand(op, i)) continue;
          op_operands.emplace_back({op, i});
        }
      }
      return op_operands;
    }();
    const auto& shardable_axes_sig = [&]{
      ShardableAxesSignature signature;
      ShardableAxesSignature.output_shardable_axes = value2shardable_axes.at(sink->result(0));
      for (const auto& pair : input_op_operands) {
        const auto& [op, idx] = pair;
        pir::Value input = op->operand_source(idx);
        ShardableAxesSignature.input_shardable_axes[pair] = value2shardable_axes.at(input);
      }
    }();
    return shardable_axes_sig;
  }

  std::unordered_map<pir::Value, ShardableAxes> ReversedInferShardableAxes(
      common::TopoWalker<const pir::Operation*>& reversed_walker,
      const pir::Operation* sink,
      const ShardableAxes& init_sa) const {
    std::unordered_map<pir::Value, ShardableAxes> value2shardable_axes{
      {sink->result(0), init_sa}
    };
    const auto& UpdateValue2ShardableAxes = [&](pir::Value value, const ShardableAxes& sa) {
      auto iter = value2shardable_axes.find(value);
      if (iter != value2shardable_axes.end()) {
        iter->second = ShardableAxesUtil::GetCommonShardableAxes(iter->second, sa);
      } else {
        iter->second = sa;
      }
    };
    reversed_walker(sink, [&](const auto* op){
      auto shardable_axes_sig = MakeShardableAxesSignature4Op(op);
      const auto& old2new = ShardableAxesUtil::GetOldName2NewName(shardable_axes_sig.output_shardable_axes,
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

 private:
  cinn::dialect::FusionOp fusion_op_;
  std::function<bool(const pir::Operation*)> IsInThisFusionOp;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
};

GroupPattern FuseToGroupPattern(const cinn::dialect::FusionOp& fusion_op) {
  StmtFusionHelper helper(fusion_op);
  std::list<StmtPattern> stmt_patterns = helper.ConvertToStmtsPattern();
  if (const auto& error = helper.Fuse_IS_x_IS_2_IS(&stmt_patterns)) return error.value();
  if (const auto& error = helper.Fuse_PS_x_PS_2_PS(&stmt_patterns)) return error.value();
  if (const auto& error = helper.Fuse_IS_x_PS_2_PS(&stmt_patterns)) return error.value();
  if (const auto& error = helper.Fuse_IS_x_R_2_R(&stmt_patterns)) return error.value();
  if (const auto& error = helper.Fuse_PS_x_R_2_R(&stmt_patterns)) return error.value();
  return stmt_patterns;
}

}

GroupPattern GenerateGroupPatternFromFusionOp(const cinn::dialect::FusionOp& fusion_op) {
  return FuseToGroupPattern(fusion_op);
}

}