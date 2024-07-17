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

#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

std::vector<ir::Var> ExprVec2VarVec(const std::vector<ir::Expr>& in) {
  std::vector<ir::Var> out;
  for (auto& expr : in) {
    out.push_back(expr.as_var_ref());
  }
  return out;
}

std::vector<ir::Expr> VarVec2ExprVec(const std::vector<ir::Var>& in) {
  return std::vector<ir::Expr>(in.begin(), in.end());
}

std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Expr& body,
                                            const ir::Tensor& tensor) {
  VLOG(4) << "GetEachTensorLoadExpr: " << tensor;
  std::vector<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesInOrder(
      body, [&tensor](const Expr* expr) {
        return expr->As<ir::Load>() && expr->As<ir::Load>()->is_addr_tensor() &&
               expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                   tensor->name;
      });
  for (auto& t : load_exprs) {
    VLOG(4) << "GetEachTensorLoadExpr Found: " << t << " " << t.ptr();
  }
  return load_exprs;
}

MappingTargetExprToDestExprMutator::MappingTargetExprToDestExprMutator(
    const ir::Expr& source, const ir::Expr& dest)
    : source_(source), dest_(dest) {}

void MappingTargetExprToDestExprMutator::operator()(Expr* expr) {
  IRMutator::Visit(expr, expr);
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Load* load, Expr* op) {
  if (load == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(load, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::For* for_node,
                                               Expr* op) {
  if (for_node == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(for_node, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Store* store,
                                               Expr* op) {
  if (store == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(store, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Reduce* reduce,
                                               Expr* op) {
  if (reduce == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(reduce, op);
  }
}

bool CheckIterEq(const std::vector<ir::Var>& up_iter,
                 const std::vector<ir::Var>& down_iter) {
  if (up_iter.size() != down_iter.size()) return false;

  for (int i = 0; i < up_iter.size(); ++i) {
    const ir::Var& up_iter_var = up_iter[i];
    const ir::Var& down_iter_var = down_iter[i];

    if (up_iter_var != down_iter_var) return false;
    if (up_iter_var->lower_bound.as_int64() !=
        down_iter_var->lower_bound.as_int64())
      return false;
    if (up_iter_var->upper_bound.as_int64() !=
        down_iter_var->upper_bound.as_int64())
      return false;
  }
  return true;
}

ir::Expr CopyedReplaceExpr(const Expr& source,
                           const std::vector<Var>& replaced,
                           const std::vector<Expr>& candidates) {
  VLOG(4) << "CopyedReplaceExpr Start";
  VLOG(4) << "Replace Body : " << source;
  VLOG(4) << "Replace From : " << cinn::utils::Join(replaced, " ");
  VLOG(4) << "Replace To   : " << cinn::utils::Join(candidates, " ");

  PADDLE_ENFORCE_EQ(
      replaced.size(),
      candidates.size(),
      ::common::errors::InvalidArgument(
          "In ReplaceExpr, the size of Vars to be replaced must be equal to "
          "the size of cadidate Exprs! Please check."));
  auto copyed_source = ir::ir_utils::IRCopy(source);
  if (replaced.empty()) return copyed_source;
  std::map<Var, Expr, ir::CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  ir::MappingVarToExprMutator mapper(replacing_map);
  mapper(&copyed_source);
  VLOG(4) << "CopyedReplaceExpr Result: " << copyed_source;
  return copyed_source;
}

void SubstitudeTargetExprWithDestExpr(const ir::Expr& source,
                                      const ir::Expr& dest,
                                      ir::Expr* body) {
  VLOG(4) << "SubstitideExpr Start";
  VLOG(4) << "Substitide Body : " << *body;
  VLOG(4) << "Substitide From : " << source;
  VLOG(4) << "Substitide To   : " << dest;
  MappingTargetExprToDestExprMutator mapper(source, dest);
  mapper(body);
  VLOG(4) << "SubstitideExpr Result: " << *body;
}

ir::Expr SubstitudeIndexVector(const Expr& source,
                               const std::vector<Var>& load_vars,
                               const std::vector<ir::Expr>& indices) {
  return CopyedReplaceExpr(source, load_vars, indices);
}
}  // namespace ComposeUtils

namespace ExprSetFinderUtils {

using ExprSet = std::vector<ir::Expr>;
using Expr2ExprSet = std::function<ExprSet(const ir::Expr& x)>;
ExprSetFinder::ExprSetFinder(Expr2ExprSet f, std::string s) {
  f_ = f;
  name = s;
}
ExprSet ExprSetFinder::operator()(const ir::Expr& x) const { return f_(x); }
ir::Expr ExprSetFinder::GetSingle(const ir::Expr& x) const {
  ExprSetFinder call = (*this) * ExprSetFinder::GetIdentity();
  const auto& o = call.operator()(x);
  if (o.size() != 1) {
    PADDLE_THROW("Try to get single result, but we get %d.", o.size());
  }
  return *o.begin();
}

ExprSetFinder ExprSetFinder::operator*(ExprSetFinder x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ExprSet {
    const auto& rs = self.f_(e);
    VLOG(6) << "ExprSetFinder Info : " << self.name;
    VLOG(6) << "        Inputs  :" << e;
    for (const auto& r : rs) {
      VLOG(6) << "      Outputs : \n" << r;
    }
    std::vector<ir::Expr> res;
    for (const auto& r : rs) {
      const auto& x_res = x.f_(r);
      res.insert(res.end(), x_res.begin(), x_res.end());
    }
    return res;
  };
  return ExprSetFinder(std::function(new_f), x.name + "*" + this->name);
}

ExprSetFinder ExprSetFinder::GetIdentity() {
  return ExprSetFinder(
      [](const ir::Expr& e) { return std::vector<ir::Expr>{e}; }, "identity");
}

ExprSetFinder Identity = ExprSetFinder::GetIdentity();

ExprSetFinder Store2Value = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::Store>()) {
        return {e.As<ir::Store>()->value};
      }
      return {};
    },
    "Store2Value");

ExprSetFinder Realizer2ScheduleBlock = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlockRealize>()) {
        return {e.As<ir::ScheduleBlockRealize>()->schedule_block};
      }
      return {};
    },
    "Realizer2ScheduleBlock");

ExprSetFinder ScheduleBlock2Body = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlock>()) {
        return {e.As<ir::ScheduleBlock>()->body};
      }
      return {};
    },
    "ScheduleBlock2Body");

ExprSetFinder ScheduleBlockRealizeNotRoot = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("root") == std::string::npos);
    },
    "ScheduleBlockRealizeNotRoot");

ExprSetFinder ScheduleBlockRealizeIsRoot = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("root") != std::string::npos);
    },
    "ScheduleBlockRealizeIsRoot");

ExprSetFinder ScheduleBlockRealizeIsNotInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") == std::string::npos);
    },
    "ScheduleBlockRealizeIsNotInit");

ExprSetFinder ScheduleBlockRealizeIsInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") != std::string::npos);
    },
    "ScheduleBlockRealizeIsInit");

ExprSetFinder IsFor = FilterMaker(
    [](const ir::Expr& e) -> bool { return e.As<ir::For>(); }, "IsFor");

ExprSetFinder ChildScheduleBlocks =
    Collector([](const ir::Expr* e) { return e->As<ir::ScheduleBlock>(); },
              "ChildScheduleBlocks");

ExprSetFinder ChildScheduleBlockRealizes =
    Collector(
        [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); },
        "ChildScheduleBlockRealizes") *
    ScheduleBlockRealizeNotRoot;

ExprSetFinder ChildRootScheduleBlockRealizes =
    Collector(
        [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); },
        "ChildScheduleBlockRealizes") *
    ScheduleBlockRealizeIsRoot;

ExprSetFinder IsForIterVar(const ir::Var& var) {
  return FilterMaker(
      [var = var](const ir::Expr& e) -> bool {
        return e.As<ir::For>() && e.As<ir::For>()->loop_var == var;
      },
      "IsForIterVar");
}

ExprSetFinder For2Min = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->min}; },
    "For2Min");

ExprSetFinder For2Max = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->extent}; },
    "For2Max");

ExprSetFinder ChildStores = Collector(
    [](const ir::Expr* e) { return e->As<ir::Store>(); }, "ChildStores");

ExprSetFinder ChildTensorLoads = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Load>() && e->As<ir::Load>()->is_addr_tensor();
    },
    "ChildLoads");

ExprSetFinder ChildTensorStores = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Store>() && e->As<ir::Store>()->is_addr_tensor();
    },
    "ChildTensorStores");

ExprSetFinder FilterLoadByTensor(const ir::Tensor& tensor) {
  return FilterMaker(
      [tensor = tensor](const ir::Expr& e) -> bool {
        return e.As<ir::Load>() &&
               e.As<ir::Load>()->tensor.as_tensor_ref()->name == tensor->name;
      },
      "FilterLoadByTensor(" + tensor->name + ")");
}

ExprSetFinder ChildFors =
    Collector([](const ir::Expr* e) { return e->As<ir::For>(); }, "ChildFors");

ExprSetFinder FindFather(const ir::Expr& root) {
  const auto& f = [&](const auto& child) -> ExprSet {
    ExprSetFinder find_child =
        Collector([child](const ir::Expr* e) { return *e == child; });
    const auto& father_collector = Collector([&](const ir::Expr* current) {
      auto res = (*current != child) && !find_child(*current).empty();
      return res;
    });
    return father_collector(root);
  };
  return ExprSetFinder(f, "FindFather");
}
}  // namespace ExprSetFinderUtils

namespace ExprTransformerUtils {
using ExprTransformFunc = std::function<ir::Expr(ir::Expr)>;

ExprTransformer::ExprTransformer(ExprTransformFunc f) { f_ = f; }
ir::Expr ExprTransformer::operator()(const ir::Expr& x) const { return f_(x); }
ExprTransformer ExprTransformer::operator*(const ExprTransformer& x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ir::Expr {
    const auto& rs = self.f_(e);
    return x.f_(rs);
  };
  return ExprTransformer(std::function(new_f));
}

ExprTransformer Identity = ExprTransformer([](const ir::Expr& e) { return e; });
ExprTransformer WrapForTransformer(const ir::Var& v) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    auto block = e;
    if (!block.As<ir::Block>()) {
      block = ir::Block::Make({e});
    }
    return ir::For::Make(v,
                         v->lower_bound,
                         v->upper_bound,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         block);
  };
  return ExprTransformer(f);
}

ExprTransformer WrapForsTransformer(const std::vector<ir::Var>& vs) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    ExprTransformer t = Identity;
    for (const auto& v : vs) {
      t = WrapForTransformer(v) * t;
    }
    return t(e);
  };
  return ExprTransformer(f);
}

ExprTransformer UnsqueezeForTransformer(
    const ExprSetFinderUtils::ExprSetFinder& followed_finder,
    const ir::Var& to_append_var) {
  const auto& suqueeze_for_func = [&](const ir::Expr& e) -> ir::Expr {
    auto copied_e = ir::ir_utils::IRCopy(e);
    ir::Expr followed_expr = followed_finder.GetSingle(copied_e);
    // (ExprSetFinderUtils::ChildFors *
    // ExprSetFinderUtils::IsForIterVar(following_for_iter_var)).GetSingle(copied_e);
    VLOG(6) << "UnsqueezeForTransformer: for insert after " << followed_expr;
    if (followed_expr.As<ir::For>()) {
      followed_expr.As<ir::For>()->body = ir::Block::Make({WrapForTransformer(
          to_append_var)(followed_expr.As<ir::For>()->body)});
    } else if (followed_expr.As<ir::ScheduleBlockRealize>()) {
      const auto& schedule_block = followed_expr.As<ir::ScheduleBlockRealize>()
                                       ->schedule_block.As<ir::ScheduleBlock>();
      schedule_block->body =
          WrapForTransformer(to_append_var)(schedule_block->body);
    } else {
      PADDLE_THROW(
          "UnsqueezeForTransformer: only support insert after a (For / "
          "ScheduleBlockRealizer): %s",
          followed_expr);
    }
    VLOG(6) << "UnsqueezeForTransformer: After changed: " << copied_e;
    return copied_e;
  };
  return ExprTransformer(suqueeze_for_func);
}

ExprTransformer ChangeTensorLoadTransformer(const ir::Tensor& tensor,
                                            const ir::Expr& dst_load) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    auto copied_e = ir::ir_utils::IRCopy(e);
    const auto& load = (ExprSetFinderUtils::ChildTensorLoads *
                        ExprSetFinderUtils::FilterLoadByTensor(tensor))
                           .GetSingle(copied_e);
    ComposeUtils::MappingTargetExprToDestExprMutator(load, dst_load)(&copied_e);
    return copied_e;
  };
  return ExprTransformer(f);
}

void ReplaceTarget(ir::Expr* e, const ir::Expr& t, const ir::Expr dst) {
  ComposeUtils::MappingTargetExprToDestExprMutator(t, dst)(e);
}

ExprTransformer WrapStoreTransformer(const ir::Tensor& tensor,
                                     const std::vector<ir::Expr>& indices) {
  const auto& MakeStoreNode = [=](const ir::Expr& e) -> ir::Expr {
    return ir::Store::Make(tensor, e, indices);
  };
  return ExprTransformer(MakeStoreNode);
}

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars) {
  int i = 0;
  std::vector<ir::Var> vars;
  for (const auto& v : block_vars) {
    vars.emplace_back("inner_block_" + std::to_string(i++));
    vars.back()->is_reduce_axis = v->is_reduce_axis;
  }
  return vars;
}

ExprTransformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                     const std::vector<ir::Var>& dest_vars) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopyedReplaceExpr(
        e,
        target_vars,
        std::vector<ir::Expr>(dest_vars.begin(), dest_vars.end()));
  };
  return ExprTransformer(f);
}

ExprTransformer ReplaceVarTransformer(const std::vector<ir::Var>& target_vars,
                                      const std::vector<ir::Expr>& dest_expr) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopyedReplaceExpr(e, target_vars, dest_expr);
  };
  return ExprTransformer(f);
}

bool IsReduceBool(const ir::Expr& lhs, const ir::Expr& rhs) {
  return lhs.type().is_bool() || rhs.type().is_bool();
}

ExprTransformer WrapReduceOperation(const ir::Reduce::ReduceType& reduce_type,
                                    const ir::Tensor& tensor,
                                    const std::vector<ir::Expr>& axis_exprs) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    switch (reduce_type) {
      case ir::Reduce::kSum:
        if (IsReduceBool(tensor(axis_exprs), e)) {
          return ir::Store::Make(tensor, tensor(axis_exprs) || e, axis_exprs);
        }
        return ir::Store::Make(tensor, tensor(axis_exprs) + e, axis_exprs);
      case ir::Reduce::kMul:
        if (IsReduceBool(tensor(axis_exprs), e)) {
          return ir::Store::Make(tensor, tensor(axis_exprs) && e, axis_exprs);
        }
        return ir::Store::Make(tensor, tensor(axis_exprs) * e, axis_exprs);
      case ir::Reduce::kMax:
        return ir::Store::Make(
            tensor, ir::Max::Make(tensor(axis_exprs), e), axis_exprs);
      case ir::Reduce::kMin:
        return ir::Store::Make(
            tensor, ir::Min::Make(tensor(axis_exprs), e), axis_exprs);
      case ir::Reduce::kAll:
        return ir::Store::Make(tensor, tensor(axis_exprs) && e, axis_exprs);
      case ir::Reduce::kAny:
        return ir::Store::Make(tensor, tensor(axis_exprs) || e, axis_exprs);
      default:
        CINN_NOT_IMPLEMENTED
    }
  };
  return ExprTransformer(f);
}

ExprTransformer SubstitudeByScheduleBlockRealize(const ir::Expr& realize) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    const auto& iter_values =
        realize.As<ir::ScheduleBlockRealize>()->iter_values;
    const auto& iter_vars = realize.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars;
    return ExprTransformerUtils::ChangeVarTransformer(
        iter_vars, ComposeUtils::ExprVec2VarVec(iter_values))(e);
  };
  return ExprTransformer(f);
}

ExprTransformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                     const std::string& tensor_name) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    if (e.As<ir::ScheduleBlock>()) {
      PADDLE_THROW("please input a non-schedule block expr.");
    }
    const auto& inner_block_var = CreateInnerBlockVars(block_vars);
    const auto& replaced_e =
        ChangeVarTransformer(block_vars, inner_block_var)(e);
    const auto& schedule_block = ir::ScheduleBlock::Make(
        inner_block_var, {}, {}, tensor_name, replaced_e);
    const auto& schedule_realizer = ir::ScheduleBlockRealize::Make(
        std::vector<ir::Expr>(block_vars.begin(), block_vars.end()),
        schedule_block);
    return schedule_realizer;
  };
  return ExprTransformer(f);
}
}  // namespace ExprTransformerUtils

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops) {
  const auto& op_pattern_map =
      Operator::GetAttrs<cinn::hlir::framework::OpPatternKind>("OpPattern");
  std::vector<OpPatternKind> op_patterns;
  const auto ConvertToPattern = [&op_pattern_map](const ::pir::Operation* op) {
    const std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    return op_pattern_map[cinn_op];
  };
  std::transform(ops.begin(),
                 ops.end(),
                 std::back_inserter(op_patterns),
                 ConvertToPattern);
  return op_patterns;
}

bool IsTrivialKind(OpPatternKind kind) {
  return kind == OpPatternKind::kElementWise ||
         kind == OpPatternKind::kBroadcast || kind == OpPatternKind::kInjective;
}

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns) {
  if (VLOG_IS_ON(4)) {
    for (const auto& func : op_compute_bodies) {
      VLOG(4) << "FuncBody is :" << func;
    }
    for (const auto& op_ptn : op_patterns) {
      VLOG(4) << "OpPattern is :" << op_ptn;
    }
  }
  VLOG(4) << "      op_patterns.size() = " << op_compute_bodies.size();
  VLOG(4) << "op_compute_bodies.size() = " << op_patterns.size();
  PADDLE_ENFORCE_EQ(
      op_patterns.size(), op_compute_bodies.size(), "ops and  size not equal");
}

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
