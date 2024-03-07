#include "paddle/cinn/frontend/group_pattern_util.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/hlir/framework/op.h"
#include <optional>

namespace cinn::frontend {

namespace {

using IS = api::InjectiveSourcePattern<FrontendPattern>;
using R = api::ReductionPattern<FrontendPattern>;
using PS = api::PartialShardablePattern<FrontendPattern>;
using OpPatternKind = cinn::hlir::framework::OpPatternKind;

hlir::framework::OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

std::function<bool(const pir::Operation*)> MakeGetterIsInThisFusionOp(const cinn::dialect::FusionOp& fusion_op) {
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

std::function<bool(const pir::Operation*)> MakeGetterIsInjectiveSource(
    const cinn::dialect::FusionOp& fusion_op,
    const std::function<bool(const pir::Operation*)>& IsInThisFusionOp) {
  using NodeVisitor = std::function<void(pir::Operation*)>;
  const auto VisitEachInput = [&](const pir::Operation* node, const NodeVisitor& DoEach) {
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (IsInThisFusionOp(input_op)) {
        DoEach(input_op);
      }
    }
  };
  const auto VisitEachOutput = [&](const pir::Operation* node, const NodeVisitor& DoEach) {
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
    for (const auto* op : fusion_op.block().ops()) {
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

struct StmtFusionHelper {
  const std::function<bool(const pir::Operation*)> IsInThisFusionOp;
  const std::function<bool(const pir::Operation*)> IsInjectiveSource;

  std::vector<StmtPattern> FuseISAndConvertRemainder(const cinn::dialect::FusionOp& fusion_op) const {
    const auto& [injective_source_ops, remainder_ops] = SplitInjectiveSourceOps(fusion_op);
    std::vector<StmtPattern> ret;
    FuseInjectiveSourceThenAppend(injective_source_ops, &ret);
    for (const auto& op : remainder_ops) {
      ret.emplace_back(ConvertNonInjectiveSourceToStmtPattern(op));
    }
    return ret;
  }

  void FuseInjectiveSourceThenAppend(
      const std::list<const pir::Operation*>& injective_source_ops,
      std::vector<StmtPattern>* ret) {
    using IterType = std::list<const pir::Operation*>::iterator;
    TODO();
  }

  StmtPattern ConvertNonInjectiveSourceToStmtPattern(const pir::Operation* op) {
    const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
    if (kind == hlir::framework::kReduction) {
      return ConvertReductionOpToStmtPattern(op);
    } else if (kind == hlir::framework::kElementWise) {
      return ConvertElementwiseOpToStmtPattern(op);
    } else if (kind == hlir::framework::kBroadcast) {
      return ConvertBroadcastOpToStmtPattern(op);
    } else {
      LOG(FATAL) << "only kReduction, kElementWise, kBroadcast supported. op_name:" << op->op_name(); 
    }
    LOG(FATAL) << "Dead code";
  }

  StmtPattern ConvertReductionOpToStmtPattern(const pir::Operation* op) {
    return R{{}, {op}};
  }

  StmtPattern ConvertElementwiseOpToStmtPattern(const pir::Operation* op) {
    CHECK(!op->isa<cinn::dialect::ReshapeOp>()) << "reshape not supported.";
    TODO();
  }

  StmtPattern ConvertBroadcastOpToStmtPattern(const pir::Operation* op) {
    LOG(FATAL) << "TODO(wuzhanfei)";
  }

  std::variant<IternalPattern, ErrorGroupPattern> MergePattern(
      const IS& upstream,
      const PS& downstream){
    PS new_pattern = CopyPattern(downstream);
    new_pattern.ops.insert(new_pattern.end(), upstream.begin(), upstream.end());
    return new_pattern;
  }

  std::variant<IternalPattern, ErrorGroupPattern> MergePattern(
      const PS& upstream,
      const PS& downstream){
    PS new_pattern = CopyPattern(downstream);
    new_pattern.ops.insert(new_pattern.end(), upstream.begin(), upstream.end());
    new_pattern.shardable_axes_signature.output_shardable_axes.insert(
      new_pattern.shardable_axes_signature.output_shardable_axes.end(), 
      upstream.shardable_axes_signature.output_shardable_axes.begin(), 
      upstream.shardable_axes_signature.output_shardable_axes.end()
    );
    new_pattern.shardable_axes_signature.input_shardable_axes.insert(
      upstream.shardable_axes_signature.input_shardable_axes.begin(), 
      upstream.shardable_axes_signature.input_shardable_axes.end()
    );
    return new_pattern
  }

  std::variant<IternalPattern, ErrorGroupPattern> MergePattern(
      const IS& upstream,
      const R& downstream){
    R new_pattern = CopyPattern(downstream);
    new_pattern.opt_inputs = CopyPattern(upstream);
    return new_pattern;
  }

  std::variant<IternalPattern, ErrorGroupPattern> MergePattern(
      const PS& upstream,
      const R& downstream){
    R new_pattern = CopyPattern(downstream);
    new_pattern.opt_inputs = CopyPattern(upstream);
    return new_pattern;
  }

  SplitedOps SplitInjectiveSourceOps(const cinn::dialect::FusionOp& fusion_op) {
    SplitedOps ret;
    for (const auto& op : fusion_op.block().ops()) {
      if (!IsInThisFusionOp(op)) continue;
      if (IsInjectiveSource(op)) {
        ret.injective_source_ops.push_back(op);
      } else {
        ret.remainder_ops.push_back(op);
      }
    }
    return ret;
  }

  struct SplitedOps {
    std::list<const pir::Operation*> injective_source_ops;
    std::list<const pir::Operation*> remainder_ops;
  }

  std::optional<std::pair<StmtPattern, StmtPattern>> FindConnetedPattenPairWithCondition(
      std::vector<StmtPattern>* stmt_patterns,
      std::function<bool(const IternalPattern& upstream, const IternalPattern& downstream)>& FuseTargetCondition) const {
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
      std::function<bool(const IternalPattern&, const IternalPattern&)>& FuseTargetCondition) const{

    while(true){
      const auto& pattern_pair = FindConnetedPattenPairWithCondition(
        stmt_patterns, FuseTargetCondition
      );
      if (!pattern_pair.value()){
        break;
      }
      const std::variant<IternalPattern, ErrorGroupPattern>& new_pattern = 
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
      [](const StmtPattern& upstream, const IternalPattern& downstream){
        return IsISPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const IternalPattern& downstream){
        return IsPSPattern(upstream) && IsPSPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const IternalPattern& downstream){
        return IsISPattern(upstream) && IsRPattern(downstream);
      }
    );
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(std::vector<StmtPattern>* stmt_patterns) const {
    return FuseIternalPattenPrototype(
      stmt_patterns,
      [](const StmtPattern& upstream, const IternalPattern& downstream){
        return IsPSPattern(upstream) && IsRPattern(downstream);
      }
    );
  }

};

GroupPattern FuseToGroupPattern(const cinn::dialect::FusionOp& fusion_op) {
  const auto& IsInThisFusionOp = MakeGetterIsInThisFusionOp(fusion_op);
  const auto& IsInjectiveSource = MakeGetterIsInjectiveSource(fusion_op, IsInThisFusionOp);
  StmtFusionHelper helper{IsInThisFusionOp, IsInjectiveSource};
  std::vector<StmtPattern> stmt_patterns = helper.FuseISAndConvertRemainder(fusion_op);
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