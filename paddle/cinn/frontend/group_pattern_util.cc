#include "paddle/cinn/frontend/group_pattern_util.h"
#include "paddle/cinn/common/topo_walker.h"
#include <optional>

namespace cinn::frontend {

namespace {

using IS = api::InjectiveSourcePattern<FrontendPattern>;
using R = api::ReductionPattern<FrontendPattern>;
using PS = api::PartialShardablePattern<FrontendPattern>;
using InternalPattern = std::variant<IS, R, PS>;


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

void InitInternalFusions(const std::optional<IS> injective_source, std::vector<InternalPattern>* ret) {
  if (injective_source.has_value()) {
    ret->emplace_back(InternalPattern{injective_source.value()});
  }
}

struct InternalFusionHelper {
  const std::function<bool(const pir::Operation*)> IsInThisFusionOp;
  const std::function<bool(const pir::Operation*)> IsInjectiveSource;

  std::vector<InternalPattern> FuseISAndConvertRemainder(const cinn::dialect::FusionOp& fusion_op) const {
    TODO();
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(std::vector<InternalPattern>* internal_patterns) const {
    TODO();
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(std::vector<InternalPattern>* internal_patterns) const {
    TODO();
  }

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(std::vector<InternalPattern>* internal_patterns) const {
    TODO();
  }

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(std::vector<InternalPattern>* internal_patterns) const {
    TODO();
  }

};

std::variant<std::vector<InternalPattern>, ErrorGroupPattern> InternalFusion(const cinn::dialect::FusionOp& fusion_op) {
  const auto& IsInThisFusionOp = MakeGetterIsInThisFusionOp(fusion_op);
  const auto& IsInjectiveSource = MakeGetterIsInjectiveSource(fusion_op, IsInThisFusionOp);
  InternalFusionHelper helper{IsInThisFusionOp, IsInjectiveSource};
  std::vector<InternalPattern> internal_patterns = helper.FuseISAndConvertRemainder(fusion_op);
  if (const auto& opt_error = helper.Fuse_IS_x_PS_2_PS(&internal_patterns)) return opt_error.value();
  if (const auto& opt_error = helper.Fuse_PS_x_PS_2_PS(&internal_patterns)) return opt_error.value();
  if (const auto& opt_error = helper.Fuse_IS_x_R_2_R(&internal_patterns)) return opt_error.value();
  if (const auto& opt_error = helper.Fuse_PS_x_R_2_R(&internal_patterns)) return opt_error.value();
  return internal_patterns;
}

std::optional<IS> ConvertToSoleIS(const std::vector<InternalPattern>& internal_patterns) {
  std::optional<IS> injective_source;
  for (const auto& pattern : internal_patterns) {
    if (std::holds_alternative<IS>(pattern)) {
      if (injective_source.has_value()) {
        LOG(FATAL) << "zero or one InjectiveSource allowed.";
      }
      injective_source = std::get<IS>(pattern);
    }
  }
  return injective_source;
}

struct ConvertInternalPatternToPSOrR {
  std::variant<PS, R> operator()(const IS& pattern) {
    LOG(FATAL) << "dead code";
  }
  std::variant<PS, R> operator()(const PS& pattern) {
    return pattern;
  }
  std::variant<PS, R> operator()(const R& pattern) {
    return pattern;
  }
}

api::ShardableReductionsPattern<FrontendPattern> LiftToShardableReductionsPattern(
    const std::vector<InternalPattern>& internal_patterns) {
  api::ShardableReductionsPattern<FrontendPattern> ret;
  for (const auto& pattern : internal_patterns) {
    ret.emplace_back(std::visit(ConvertInternalPatternToPSOrR{}, pattern));
  }
  return ret;
}


GroupPattern LiftToGroupPattern(const std::vector<InternalPattern>& internal_patterns) {
  if (const auto& opt_injective_src = ConvertToSoleIS(internal_patterns)) return opt_injective_src.value();
  return LiftToShardableReductionsPattern(internal_patterns);
}

struct SafeLiftToGroupPattern {
  std::variant<GroupPattern, ErrorGroupPattern> operator()(const ErrorGroupPattern& error) const {
    return error;
  }

  std::variant<GroupPattern, ErrorGroupPattern> operator()(const std::vector<InternalPattern>& patterns) const {
    return LiftToGroupPattern(patterns);
  }
};

}

std::variant<GroupPattern, ErrorGroupPattern> GenerateGroupPatternFromFusionOp(const cinn::dialect::FusionOp& fusion_op) {
  return std::visit(SafeLiftToGroupPattern{}, InternalFusion(fusion_op));
}

}