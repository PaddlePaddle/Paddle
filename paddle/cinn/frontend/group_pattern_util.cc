#include "paddle/cinn/frontend/group_pattern_util.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/framework/op.h"
#include <optional>

namespace cinn::frontend {

namespace {

using IS = api::InjectiveSourcePattern<FrontendPattern>;
using R = api::ReductionPattern<FrontendPattern>;
using PS = api::PartialShardablePattern<FrontendPattern>;
using StmtPattern = api::StmtPattern<FrontendPattern>;
using OpPatternKind = cinn::hlir::framework::OpPatternKind;

OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

std::function<size_t(const pir::Operation*)> MakeGetterOrderValue4Op(const cinn::dialect::FusionOp& fusion_op) {
  std::unordered_map<pir::Operation*, size_t> op2order_in_block;
  size_t order = 0;
  for (const pir::Operation* op : fusion_op.block()->ops()) {
    op2order_in_block[op] = ++order;
  }
  return [map=std::move(op2order_in_block)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}


bool IsISPattern(const StmtPattern& pattern){
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern){
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern){
  return std::holds_alternative<R>(pattern);
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

bool IsGeneralInjective(const pir::Operation* op) {
  hlir::framework::OpPatternKind op_pattern_kind = GetOpPatternKind(op);
  return op_pattern_kind == hlir::framework::kElementWise
    || op_pattern_kind == hlir::framework::kBroadcast
    || op_pattern_kind == hlir::framework::kInjective;
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const cinn::dialect::FusionOp& fusion_op,
    const std::function<bool(const pir::Operation*)>& IsInThisFusionOp) {
  using NodeVisitor = std::function<void(pir::Operation*)>;
  const auto VisitEachInput = [&](const pir::Operation* op, const NodeVisitor& DoEach) {
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (IsInThisFusionOp(input_op)) {
        DoEach(input_op);
      }
    }
  };
  const auto VisitEachOutput = [&](const pir::Operation* op, const NodeVisitor& DoEach) {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin(); consumer_it != output.use_end(); ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (IsInThisFusionOp(consumer_op)) {
          DoEach(consumer_op);
        }
      }
    }
  };

  const auto starts = [&]{
    const auto& IsSource = [&](const pir::Operation* op) {
      std::size_t num_inputs = 0;
      VisitEachInput([&](const pir::Operation*) { ++num_inputs});
      return num_inputs == 0;
    };
    std::list<const pir::Operation*> starts;
    for (const auto* op : fusion_op.GetOperators()) {
      if (!IsInThisFusionOp(op)) continue;
      if (IsSource(op)) {
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
    VisitEachInput(op, [&](const pir::Operation* input){
      is_inputs_all_injective_source = (is_inputs_all_injective_source && op_2_is_injective_source.at(input));
    });
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
    for (const auto* op : fusion_op_.block()->ops()) {
      if (!IsInThisFusionOp(op)) continue;
      ret.emplace_back(ConvertToStmtPattern(op));
    }
    return ret;
  }

  using StmtIter = std::list<StmtPattern>::iterator;

  static std::function<std::optional<StmtIter>(const pir::Operation*)>
  MakeGetterStmt4Op(std::list<StmtPattern>* stmts) const {
    std::unordered_map<const pir::Operation*, StmtIter> op2stmt_iter;
    for (auto iter = stmts->begin(); iter != stmts->end(); ++iter) {
      op2stmt_iter[GetSoleOp(*iter)] = iter;
    }
    return [map=std::move(op2stmt_iter)](const pir::Operation* op) -> std::optional<StmtIter> {
      const auto iter = map.find(op);
      if (iter == map.end()) return std::nullopt;
      return iter->second;
    };
  }

  const pir::Operation* GetSoleOpImpl(const IS& injective_source) const {
    CHECK_EQ(injective_source.ops.size(), 1);
    return injective_source.ops.at(0);
  }

  const pir::Operation* GetSoleOpImpl(const R& reduce) const {
    return reduce.reduce_op;
  }

  const pir::Operation* GetSoleOpImpl(const PS& partial_shardable) const {
    CHECK_EQ(partial_shardable.ops.size(), 1);
    return partial_shardable.ops.at(0);
  }

  const pir::Operation* GetSoleOp(const StmtPattern& stmt) const {
    return std::visit([&](const auto& impl) {
      return GetSoleOpImpl(impl);
    }, stmt);
  }

  template<typename IsDetailPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsDetailPatternT& IsDetailPattern,
      const ConstructPatternT& ConstructPattern,
      std::list<StmtPattern>* stmts) const {
    const auto StmtIter4Op = MakeGetterStmt4Op(stmts);
    using NodeVisitor = std::function<void(StmtIter)>;
    const auto VisitInputStmt = [&](StmtIter stmt, const NodeVisitor& DoEach) {
      const pir::Operation* op = GetSoleOp(*stmt);
      VisitEachInputOp(op, [&](const pir::Operation* input) {
        if (const auto& input_stmt = StmtIter4Op(input)) {
          if (IsDetailPattern(*input_stmt.value())) {
            DoEach(input_stmt.value());
          }
        }
      });
    };
    const auto VisitOutputStmt = [&](StmtIter stmt, const NodeVisitor& DoEach) {
      const pir::Operation* op = GetSoleOp(*stmt);
      VisitEachOutputOp(op, [&](const pir::Operation* output) {
        if (const auto& output_stmt = StmtIter4Op(output)) {
          if (IsDetailPattern(*output_stmt.value())) {
            DoEach(output_stmt.value());
          }
        }
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
    const auto GetOrder = MakeGetterOrderValue4Op(fusion_op_);
    const auto Cmp = [&](const auto* lhs, const auto& rhs) {
      return GetOrder(lhs) < GetOrder(rhs);
    };
    const auto& GetVisitedOps = [&](const auto stmt_iter) {
      std::vector<const pir::Operation*> visited_ops;
      reverse_walker(start, [&](const auto node){
        visited_ops.push_back(GetSoleOp(node));
      });
      std::sort(visited_ops.begin(), visited_ops.end(), Cmp);
      return visited_ops;
    };
    common::BfsWalker<StmtIter> reverse_walker(VisitInputStmt);
    std::list<StmtPattern> fused_stmts;
    for (auto stmt_iter = stmts->begin(); stmt_iter != stmts->end(); ++stmt_iter) {
      if (!IsSinkPattern(stmt_iter)) continue;
      fused_stmts.emplace_back(ConstructPattern(GetVisitedOps(stmt_iter)));
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
  
  using OpVisitor = std::function<void(const pir::Operation*)>;

  void VisitInputOp(const pir::Operation* op, const OpVisitor& DoEach) const {
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (IsInThisFusionOp(input_op)) {
        DoEach(input_op);
      }
    }
  }

  void VisitOutputOp(const pir::Operation* op, const OpVisitor& DoEach) const {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin(); consumer_it != output.use_end(); ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (IsInThisFusionOp(consumer_op)) {
          DoEach(consumer_op);
        }
      }
    }
  }

  StmtPattern ConvertToStmtPattern(const pir::Operation* op) const {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (IsInjectiveSource(op)) {
      return ConvertToIS(op);
    } else if (kind == hlir::framework::kReduction) {
      return ConvertReductionOpToReductionPattern(op);
    } else if (kind == hlir::framework::kElementWise) {
      return ConvertElementwiseOpToPS(op);
    } else if (kind == hlir::framework::kBroadcast) {
      return ConvertBroadcastOpToPS(op);
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

  PS ConvertElementwiseOpToPS(const pir::Operation* op) const {
    CHECK(!op->isa<cinn::dialect::ReshapeOp>()) << "reshape not supported. TODO(wuzhanfei).";
    const auto& GetRank = [](pir::Value value) -> size_t {
      return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
    };
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
    const auto& shardable_axes_signature = [&]{
      const ShardableAxes shardable_axes = GetElementwiseOpShardableAxes(rank);
      std::unordered_map<OpOperand, ShardableAxes> input_shardable_axes;
      for (int i = 0; i < op->num_operands(); ++i) {
        input_shardable_axes[std::pair(op, i)] = shardable_axes;
      }
      return ShardableAxesSignature{
        .output_shardable_axes,
        .input_shardable_axes=input_shardable_axes,
      };
    }();
    return PS{
      .ops={op},
      .shardable_axes_signature=shardable_axes_signature,
    };
  }

  ShardableAxes GetElementwiseOpShardableAxes(size_t rank) const {
    ShardableAxes ret;
    for (int i = 0; i < rank; ++i) {
      ret.emplace_back(ShardableAxis{
        .axis=i,
        .axis_name=std::string("D") + std::to_string(ShardableAxis::UnqiueSeqNo())
      });
    }
    return ret;
  }

  PS ConvertBroadcastOpToPS(const pir::Operation* op) const {
    LOG(FATAL) << "TODO(wuzhanfei).";
  }

  std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const IS& upstream,
      const PS& downstream){
    PS new_pattern = PS(downstream);
    new_pattern.ops.insert(new_pattern.end(), upstream.begin(), upstream.end());
    return new_pattern;
  }

  std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const PS& upstream,
      const PS& downstream){
    PS new_pattern = PS(downstream);
    new_pattern.ops.insert(new_pattern.end(), upstream.begin(), upstream.end());
    return new_pattern
  }

  std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const IS& upstream,
      const R& downstream){
    R new_pattern = R(downstream);
    new_pattern.opt_inputs = IS(upstream);
    return new_pattern;
  }

  std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const PS& upstream,
      const R& downstream){
    R new_pattern = R(downstream);
    new_pattern.opt_inputs = PS(upstream);
    return new_pattern;
  }

  std::optional<std::pair<StmtPattern, StmtPattern>> FindConnetedPattenPairWithCondition(
      std::list<StmtPattern>* stmt_patterns,
      std::function<bool(const StmtPattern& upstream, const StmtPattern& downstream)>& FuseTargetCondition) const {
    for (int i=0; i<stmt_patterns.size(); i++){
      for (int j=i+1; j<stmt_patterns.size(); j++){
        bool i_used_j = FirstIsUpstreamOfSecond(stmt_patterns[j], stmt_patterns[i]);
        bool j_used_i = FirstIsUpstreamOfSecond(stmt_patterns[i], stmt_patterns[j]);

        if (i_used_j && FuseTargetCondition(stmt_patterns[j], stmt_patterns[i])){
          return std::make_pair(stmt_patterns[j], stmt_patterns[i]);
        }else if(j_used_i && FuseTargetCondition(stmt_patterns[i], stmt_patterns[j])){
          return std::make_pair(stmt_patterns[i], stmt_patterns[j]);
        }else{
          continue;
        }
      }
    }
    return std::nullopt;
  }

  std::optional<ErrorGroupPattern> FuseIternalPattenPrototype(
      std::list<StmtPattern>* stmt_patterns,
      std::function<bool(const StmtPattern&, const StmtPattern&)>& FuseTargetCondition) const{

    while(true){
      const auto& pattern_pair = FindConnetedPattenPairWithCondition(
        stmt_patterns, FuseTargetCondition
      );
      if (!pattern_pair.value()){
        break;
      }
      const std::variant<StmtPattern, ErrorGroupPattern>& new_pattern = 
        MergePattern(pattern_pair.first, pattern_pair.second);

      if (IsErrorGroupPattern(new_pattern)){
        return new_pattern;
      }

      iternal_patterns.erase(pattern_pair.first);
      iternal_patterns.erase(pattern_pair.second);
      stmt_patterns->emplace_back(new_pattern);
    }
    return {};
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(std::list<StmtPattern>* stmts) const {
    const auto ConstructISPattern = [&](const auto& ops) { return IS{ops}; };
    return MultiFuse(IsISPattern, ConstructISPattern, stmts);
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(std::list<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsISPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

/*
  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::list<StmtPattern>* stmt_patterns) const {
    const auto shardable_axes_signature = [&](const auto& ops) {

    };
    const auto ConstructPSPattern = [&](const auto& ops) {
      const auto shardable_axes_signature = GetShardableAxesSignature(ops);
      return PS{
        .ops=ops,
        .shardable_axes_signature=shardable_axes_signature,
      };
    };
    return MultiFuse(IsPSPattern, ConstructISPattern, stmts);
  }
*/

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::list<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsPSPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(std::list<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsISPattern(upstream) && IsRPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(std::list<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsPSPattern(upstream) && IsRPattern(downstream);
      }
    );
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