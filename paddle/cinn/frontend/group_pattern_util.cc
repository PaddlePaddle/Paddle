#include "paddle/cinn/frontend/group_pattern_util.h"

namespace cinn::frontend {

namespace {

using IS = InjectiveSourcePattern<FrontendPattern>;
using R = ReductionPattern<FrontendPattern>;
using PS = PartialShardablePattern<FrontendPattern>;
using InternalPattern = std::variant<IS, R, PS>;


std::function<bool(const pir::Operation*)> MakeGetterIsInThisFusionOp(const pir::FusionOp& fusion_op) {
  TODO();
}

std::function<bool(const pir::Operation*)> MakeGetterIsInjectiveSource(
    const pir::FusionOp& fusion_op,
    const std::function<bool(const pir::Operation*)>& IsInThisFusionOp) {
  TODO();
}

void InitInternalFusions(const std::optional<IS> injective_source, std::vector<InternalPattern>* ret) {
  if (injective_source.has_value()) {
    ret->emplace_back(InternalPattern{injective_source.value()});
  }
}

struct InternalFusionHelper {
  const std::function<bool(const pir::Operation*)> IsInThisFusionOp;
  const std::function<bool(const pir::Operation*)> IsInjectiveSource;

  std::vector<InternalPattern> FuseISAndConvertRemainder(const pir::FusionOp& fusion_op) const {
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

std::variant<std::vector<InternalPattern>, ErrorGroupPattern> InternalFusion(const pir::FusionOp& fusion_op) {
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

std::variant<GroupPattern, ErrorGroupPattern> LiftToGroupPattern(const std::vector<InternalPattern>& internal_patterns) {
  TODO();
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

std::variant<GroupPattern, ErrorGroupPattern> GenerateGroupPatternFromFusionOp(const pir::FusionOp& fusion_op) {
  return std::visit(SafeLiftToGroupPattern{}, InternalFusion(fusion_op));
}

}