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

void MappingTargetExprToDestExprMutator::Visit(
    const ir::ScheduleBlockRealize* r, Expr* op) {
  if (r == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(r, op);
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

void MappingTargetExprToDestExprMutator::Visit(const ir::Block* block_node,
                                               Expr* op) {
  if (block_node == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(block_node, op);
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
  VLOG(5) << "Substitide Body : " << *body;
  ir::Expr new_dest = dest;
  if (source.type() != dest.type()) {
    VLOG(4) << "Cast the dest" << dest << " to type" << source.type();
    new_dest = ir::Cast::Make(source.type(), dest);
  }
  VLOG(4) << "Substitide From : " << source;
  VLOG(4) << "Substitide To   : " << new_dest;
  MappingTargetExprToDestExprMutator mapper(source, new_dest);
  mapper(body);
  VLOG(5) << "SubstitideExpr Result: " << *body;
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
  PADDLE_ENFORCE_EQ(o.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "Try to get single result, but we get %d.", o.size()));
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

ExprSetFinder Realizer2IterValues = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlockRealize>()) {
        return e.As<ir::ScheduleBlockRealize>()->iter_values;
      }
      return {};
    },
    "Realizer2IterValues");

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

ExprSetFinder ScheduleBlockRealizeIsSplitTransform = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("_split_transform") != std::string::npos);
    },
    "ScheduleBlockRealizeIsSplitTransform");

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
  const auto& f = [root](const auto& child) -> ExprSet {
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

ExprSetFinder DirectlyFather(const ir::Expr& root) {
  const auto& f = [root](const auto& child) -> ExprSet {
    ExprSet result = FindFather(root)(child);
    // VLOG(4) << "Direcly Father of \n" << child << "\nIn root: \n" << root <<
    // "\n is : "; for (const auto& r: result){ VLOG(4) << "\n  RESULT: " << r;
    //}
    return {result[result.size() - 1]};
  };
  return ExprSetFinder(f, "DirectlyFather");
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
      PADDLE_THROW(::common::errors::PreconditionNotMet(
          "UnsqueezeForTransformer: only support insert after a (For / "
          "ScheduleBlockRealizer)"));
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

ExprTransformer RemoveVarInScheduleBlockRealize(const ir::Var& target_vars,
                                                const ir::Expr& replaced_expr) {
  /*
   * remove var in schedule block realize, replace it with replaced_expr and
   * remove it in axes.bind()
   */
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    VLOG(4) << "Start RemoveVarInScheduleBlockRealize(" << target_vars << ", "
            << replaced_expr << ")";
    VLOG(4) << "      Input is " << e;
    PADDLE_ENFORCE_NE(
        e.As<ir::ScheduleBlockRealize>(),
        nullptr,
        ::common::errors::InvalidArgument(
            "RemoveVarInScheduleBlockRealize: input expr is not a "
            "ScheduleBlockRealize."));
    auto copied_ir = ir::ir_utils::IRCopy(e);
    auto schedule_block_iter_vars =
        copied_ir.As<ir::ScheduleBlockRealize>()->iter_values;
    auto block_bound_vars = copied_ir.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars;
    for (const auto& i_var : schedule_block_iter_vars) {
      PADDLE_ENFORCE_EQ(
          i_var.is_var(),
          true,
          ::common::errors::InvalidArgument("RemoveVarInScheduleBlockRealize: "
                                            "axes.bind rhs is is not a Var."));
    }
    // find replace idx
    int target_idx = -1;
    for (int i = 0; i < schedule_block_iter_vars.size(); ++i) {
      VLOG(4) << "RemoveVarInScheduleBlockRealize: compare with "
              << schedule_block_iter_vars[i] << " vs " << target_vars
              << ", and equality is: "
              << (schedule_block_iter_vars[i].as_var()->name ==
                  target_vars->name);
      if (schedule_block_iter_vars[i].as_var()->name == target_vars->name) {
        target_idx = i;
      }
    }
    if (target_idx == -1) {
      return copied_ir;  // do nothing, can't find target vars;
    }
    return ir::ScheduleBlockRealize::Make(
        fusion::GatherVectorExcept<ir::Expr, int>(schedule_block_iter_vars,
                                                  {target_idx}),
        ir::ScheduleBlock::Make(
            fusion::GatherVectorExcept<ir::Var, int>(block_bound_vars,
                                                     {target_idx}),
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->read_buffers,
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->write_buffers,
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name,
            ComposeUtils::CopyedReplaceExpr(
                copied_ir.As<ir::ScheduleBlockRealize>()
                    ->schedule_block.As<ir::ScheduleBlock>()
                    ->body,
                {block_bound_vars[target_idx]},
                {replaced_expr})));
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
    PADDLE_ENFORCE_EQ(e.As<ir::ScheduleBlock>(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "please input a non-schedule block expr."));
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

ExprTransformer RemoveOneTransformer(int one) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    ir::Expr copied = ir::ir_utils::IRCopy(root);
    const auto& iters = GetAllLoopVars(copied);
    // Find target expr and replace with for->body.
    const ir::Expr& target_for = (ExprSetFinderUtils::ChildFors *
                                  ExprSetFinderUtils::IsForIterVar(iters[one]))
                                     .GetSingle(copied);
    const ir::Expr& target_block =
        ExprSetFinderUtils::DirectlyFather(copied).GetSingle(target_for);
    VLOG(4) << "RemoveOneTransformer: directly target_block of for is "
            << target_block;
    if (target_block.As<ir::ScheduleBlockRealize>() != nullptr) {
      VLOG(4) << "RemoveOneTransformer: father block is root realize";
      ir::Expr shedule_block =
          target_block.As<ir::ScheduleBlockRealize>()->schedule_block;
      PADDLE_ENFORCE_EQ(
          shedule_block.As<ir::ScheduleBlock>()->body,
          target_for,
          ::common::errors::InvalidArgument(
              "Root realize body should be equal to target for."));
      const auto for_body = target_for.As<ir::For>()->body;
      const auto for_body_stmts = for_body.As<ir::Block>()->stmts;
      if (for_body_stmts.size() == 1 &&
          for_body_stmts[0].As<ir::For>() != nullptr) {
        shedule_block.As<ir::ScheduleBlock>()->body = for_body_stmts[0];
      } else {
        shedule_block.As<ir::ScheduleBlock>()->body = for_body;
      }
    } else if (target_block.As<ir::Block>() != nullptr) {
      VLOG(4) << "RemoveOneTransformer: father block is Block";
      std::vector<ir::Expr> new_bodies;
      for (const auto& expr : target_block.As<ir::Block>()->stmts) {
        if (expr != target_for) {
          new_bodies.push_back(expr);
        } else {
          for (const auto& origin_for :
               target_for.As<ir::For>()->body.As<ir::Block>()->stmts) {
            new_bodies.push_back(origin_for);
          }
        }
      }
      ir::Expr to_replace_block = ir::Block::Make(new_bodies);
      ComposeUtils::MappingTargetExprToDestExprMutator(
          target_block, to_replace_block)(&copied);
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "RemoveOneTransformer: target for father should be a ir::Block or "
          "ir::ScheduleBlockRealize."));
    }
    VLOG(4) << "Remove Var to 0 in ScheduleBlockRealizer: " << copied;
    // Remove var to 0 in ScheduleBlockRealizer
    InplaceMutateSingleExpr(
        &copied,
        (ExprSetFinderUtils::ChildScheduleBlockRealizes *
         ExprSetFinderUtils::ScheduleBlockRealizeIsNotInit),
        RemoveVarInScheduleBlockRealize(iters[one], ir::Expr(0)));
    InplaceMutateSingleExpr(
        &copied,
        (ExprSetFinderUtils::ChildScheduleBlockRealizes *
         ExprSetFinderUtils::ScheduleBlockRealizeIsInit),
        RemoveVarInScheduleBlockRealize(iters[one], ir::Expr(0)));
    return copied;
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveOnesTransformer(const std::vector<int32_t>& ones) {
  ExprTransformer f = Identity;
  for (const auto& one : ones) {
    f = RemoveOneTransformer(one) * f;
  }
  return ExprTransformer(f);
}

ExprTransformer TransposeForsTransformer(const std::vector<int32_t>& perm) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    const auto& iters = GetNonReduceLoopVars(root);
    PADDLE_ENFORCE_EQ(
        iters.size(),
        perm.size(),
        ::common::errors::InvalidArgument(
            "Transposed iters size and perm size should be equal."));
    for (size_t i = 0; i < perm.size(); ++i) {
      if (iters[i]->is_reduce_axis) {
        PADDLE_ENFORCE_EQ(i,
                          perm[i],
                          ::common::errors::InvalidArgument(
                              "Can only transpose non reduce iters."));
      }
    }
    const auto transposed_iters = cinn::fusion::TransposeVector(iters, perm);
    const auto non_reduce_iters = cinn::fusion::FilterVector(
        transposed_iters, [](const ir::Var& v) { return !v->is_reduce_axis; });
    const auto body_block = GetBodyBlock(root);
    return ir::Block::Make({(WrapForsTransformer(non_reduce_iters) *
                             WrapScheduleRealizer({}, "root"))(body_block)});
  };
  return ExprTransformer(f);
}

ExprTransformer InsertForsTransformer(const std::vector<int32_t>& axis,
                                      const std::vector<ir::Var>& vars) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    auto iters = GetNonReduceLoopVars(root);
    PADDLE_ENFORCE_EQ(
        axis.size(),
        vars.size(),
        ::common::errors::InvalidArgument(
            "The number of axis to insert and vars should be equal."));
    const size_t reduce_size =
        std::count_if(iters.begin(), iters.end(), [](const ir::Var& v) {
          return v->is_reduce_axis;
        });
    for (size_t i = 0; i < axis.size(); ++i) {
      PADDLE_ENFORCE_LE(axis[i],
                        iters.size() - reduce_size,
                        ::common::errors::OutOfRange(
                            "Insert axis should not be behind reduce axis."));
      iters.insert(iters.begin() + axis[i], vars[i]);
    }
    const auto non_reduce_iters =
        cinn::fusion::SliceVector(iters, 0, iters.size() - reduce_size);
    const auto body_block = GetBodyBlock(root);
    return ir::Block::Make({(WrapForsTransformer(non_reduce_iters) *
                             WrapScheduleRealizer({}, "root"))(body_block)});
  };
  return ExprTransformer(f);
}

ExprTransformer InsertIfForAppendVarsTransformer() {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    const auto vars = GetNonReduceLoopVars(root);
    std::vector<ir::Expr> conditions;
    for (const auto& var : vars) {
      if (var->name.find("append") == std::string::npos) {
        continue;
      }
      VLOG(4) << "Insert If for append loop: " << var;
      conditions.push_back(ir::EQ::Make(var, var->lower_bound));
    }
    std::reverse(conditions.begin(), conditions.end());

    auto realizes = (ExprSetFinderUtils::ChildScheduleBlockRealizes *
                     ExprSetFinderUtils::ScheduleBlockRealizeNotRoot)(root);
    for (auto realize : realizes) {
      auto schedule_block =
          realize.As<ir::ScheduleBlockRealize>()->schedule_block;
      auto new_body = schedule_block.As<ir::ScheduleBlock>()->body;
      for (const auto& cond : conditions) {
        new_body = ir::IfThenElse::Make(cond, new_body, ir::Expr());
      }
      schedule_block.As<ir::ScheduleBlock>()->body = new_body;
    }

    return root;
  };
  return ExprTransformer(f);
}

int InplaceMutateSingleExpr(ir::Expr* root,
                            const ExprSetFinderUtils::ExprSetFinder& finder,
                            const ExprTransformer& transformer) {
  // NOTE!!!
  // source ir::node type must be supported in
  // MappingTargetExprToDestExprMutator.
  const auto& source = finder(*root);
  if (source.empty()) {
    return 0;
  }
  PADDLE_ENFORCE_EQ(
      source.size(),
      1,
      ::common::errors::InvalidArgument("Only one expr should be found."));
  const auto& target = transformer(source[0]);
  ComposeUtils::MappingTargetExprToDestExprMutator(source[0], target)(root);
  return 1;  // operation number.
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
      op_patterns.size(),
      op_compute_bodies.size(),
      ::common::errors::InvalidArgument(
          "The number of op_compute_bodies and op_patterns should be equal."));
}

std::vector<ir::Var> AppendBound(const std::vector<ir::Var> vars,
                                 const ir::Expr& root) {
  return ExprSetFinderUtils::MapVector<ir::Var>(
      vars, [&](const auto& v) -> ir::Var {
        VLOG(4) << "Start Append Bound for " << v;
        VLOG(4) << "AppendBound for " << v << ", lower: "
                << (ExprSetFinderUtils::ChildFors *
                    ExprSetFinderUtils::IsForIterVar(v) *
                    ExprSetFinderUtils::For2Min)
                       .GetSingle(root)
                << ", upper: "
                << (ExprSetFinderUtils::ChildFors *
                    ExprSetFinderUtils::IsForIterVar(v) *
                    ExprSetFinderUtils::For2Max)
                       .GetSingle(root);
        return ir::Var(
            (ExprSetFinderUtils::ChildFors *
             ExprSetFinderUtils::IsForIterVar(v) * ExprSetFinderUtils::For2Min)
                .GetSingle(root),
            (ExprSetFinderUtils::ChildFors *
             ExprSetFinderUtils::IsForIterVar(v) * ExprSetFinderUtils::For2Max)
                .GetSingle(root),
            v->name,
            v->is_reduce_axis);
      });
}

std::vector<ir::Var> GetNonReduceLoopVars(const ir::Expr& root) {
  const auto& fors_expr = (ExprSetFinderUtils::ChildFors)(root);
  std::vector<ir::Var> loop_vars;
  for (const auto& for_expr : fors_expr) {
    loop_vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  const auto non_reduce_loop_vars = cinn::fusion::FilterVector(
      loop_vars, [](const ir::Var& v) { return !v->is_reduce_axis; });
  return AppendBound(non_reduce_loop_vars, root);
}

std::vector<ir::Var> GetAllLoopVars(const ir::Expr& root) {
  const auto& fors_expr = (ExprSetFinderUtils::ChildFors)(root);
  std::vector<ir::Var> loop_vars;
  for (const auto& for_expr : fors_expr) {
    loop_vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  return AppendBound(loop_vars, root);
}

ir::Expr GetBodyBlock(const ir::Expr& root) {
  const auto& iters = GetNonReduceLoopVars(root);
  const size_t reduce_size =
      std::count_if(iters.begin(), iters.end(), [](const ir::Var& v) {
        return v->is_reduce_axis;
      });
  PADDLE_ENFORCE_LT(reduce_size,
                    iters.size(),
                    ::common::errors::InvalidArgument(
                        "The reduce size should be less than the total size."));
  return (ExprSetFinderUtils::ChildFors *
          ExprSetFinderUtils::IsForIterVar(
              iters[iters.size() - reduce_size - 1]))
      .GetSingle(root)
      .As<ir::For>()
      ->body;
}

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
