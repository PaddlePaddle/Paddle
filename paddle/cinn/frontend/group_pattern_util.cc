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

  std::vector<StmtPattern> FuseISAndConvertRemainder() const {
    std::vector<StmtPattern> ret;
    FuseInjectiveSourceThenAppend(fusion_op_, &ret);
    for (const auto* op : fusion_op_.block()->ops()) {
      if (!IsInThisFusionOp(op)) continue;
      if (IsInjectiveSource(op)) continue;
      ret.emplace_back(ConvertNonInjectiveSourceToStmtPattern(op));
    }
    return ret;
  }

  void FuseInjectiveSourceThenAppend(
      std::vector<StmtPattern>* ret) const {
    auto GetOrder = MakeGetterOrderValue4Op(fusion_op_);
    auto Cmp = [&](const auto* lhs, const auto& rhs) {
      return GetOrder(lhs) < GetOrder(rhs);
    };
    VisitConnectedInjectiveSource([&](std::vector<const pir::Operation*>&& ops){
      std::sort(ops.begin(), ops.end(), Cmp);
      ret->emplace_back(IS{ops});
    });
  }

  template <typename DoEachT>
  void VisitConnectedInjectiveSource(
      const DoEachT& DoEach) const {
    const auto VisitNext = [&](const pir::Operation* node, const OpVisitor& DoEach) {
      VisitInputInjectiveSource(node, DoEach);
      VisitOutputInjectiveSource(node, DoEach);
    };
    common::BfsWalker<const pir::Operation*> bfs_walker(VisitNext);
    std::unordered_set<const pir::Operation*> visisted_ops;
    for (const auto* start : fusion_op_.block()->ops()) {
      if (!IsInThisFusionOp(start)) continue;
      if (!IsInjectiveSource(start)) continue;
      if (visisted_ops.count(start) > 0) continue;
      std::vector<const pir::Operation*> current_visited_ops;
      bfs_walker(start, [&](const pir::Operation* op){
        CHECK(visisted_ops.emplace(op).second);
        current_visited_ops.push_back(op);
      });
      DoEach(std::move(current_visited_ops));
    }
  }
  
  using OpVisitor = std::function<void(const pir::Operation*)>;

  void VisitInputInjectiveSource(const pir::Operation* op, const OpVisitor& DoEach) const {
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (IsInThisFusionOp(input_op) && IsInjectiveSource(input_op)) {
        DoEach(input_op);
      }
    }
  }

  void VisitOutputInjectiveSource(const pir::Operation* op, const OpVisitor& DoEach) const {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin(); consumer_it != output.use_end(); ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (IsInThisFusionOp(consumer_op) && IsInjectiveSource(input_op)) {
          DoEach(consumer_op);
        }
      }
    }
  }

  StmtPattern ConvertNonInjectiveSourceToStmtPattern(const pir::Operation* op) const {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (kind == hlir::framework::kReduction) {
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
      std::vector<StmtPattern>* stmt_patterns,
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
      std::vector<StmtPattern>* stmt_patterns,
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

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsISPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsPSPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const StmtPattern& downstream){
        return IsISPattern(upstream) && IsRPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(std::vector<StmtPattern>* stmt_patterns) const {
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
  std::vector<StmtPattern> stmt_patterns = helper.FuseISAndConvertRemainder();
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