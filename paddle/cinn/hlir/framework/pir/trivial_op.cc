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

#include "paddle/cinn/hlir/framework/pir/trivial_op.h"

#include <variant>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

// #include "paddle/cinn/frontend/group_pattern_util.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

struct MappingTargetExprToDestExprMutator : public ir::IRMutator<> {
  explicit MappingTargetExprToDestExprMutator(const ir::Expr& source,
                                              const ir::Expr& dest)
      : source_(source), dest_(dest) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* load, Expr* op) override {
    if (load == source_.ptr()) {
      VLOG(4) << "substitude find!";
      *op = dest_;
    } else {
      IRMutator::Visit(load, op);
    }
  }
  void Visit(const ir::Store* store, Expr* op) override {
    if (store == source_.ptr()) {
      VLOG(4) << "substitude find!";
      *op = dest_;
    } else {
      IRMutator::Visit(store, op);
    }
  }
  void Visit(const ir::Reduce* reduce, Expr* op) override {
    if (reduce == source_.ptr()) {
      VLOG(4) << "substitude find!";
      *op = dest_;
    } else {
      IRMutator::Visit(reduce, op);
    }
  }

 private:
  ir::Expr source_;
  ir::Expr dest_;
};

std::set<Expr> GetStoreFromBody(const ir::Expr& body) {
  std::set<Expr> store_tensor_exprs =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          body, [](const Expr* expr) {
            return expr->As<ir::Store>() &&
                   expr->As<ir::Store>()->is_addr_tensor();
          });

  return store_tensor_exprs;
}

std::vector<ir::Var> GetOutputIters(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> vars;
  std::transform(indices.begin(),
                 indices.end(),
                 std::back_inserter(vars),
                 [](const ir::Expr& expr) { return expr.as_var_ref(); });
  return vars;
}

bool CheckIterEq(std::vector<ir::Var> up_iter, std::vector<ir::Var> down_iter) {
}

static ir::Expr CopyedReplaceExpr(const Expr& source,
                                  const std::vector<Var>& replaced,
                                  const std::vector<Expr>& candidates) {
  VLOG(4) << "Copyed Replace Expr Start";
  CHECK_EQ(replaced.size(), candidates.size())
      << "In ReplaceExpr, the size of Vars to be replaced must be equal to "
         "the "
         "size of cadidate Exprs! Please check.";
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
  VLOG(4) << "Copyed Replace Expr End";
  return copyed_source;
}

static void SubstitudeTargetExprWithDestExpr(const ir::Expr& source,
                                             const ir::Expr& dest,
                                             ir::Expr* body) {
  VLOG(4) << "Start SubstitudeTargetExprWithDestExpr";
  MappingTargetExprToDestExprMutator mapper(source, dest);
  mapper(body);
  VLOG(4) << "End SubstitudeTargetExprWithDestExpr";
}

static ir::Expr SubstitudeIndexVector(const Expr& source,
                                      const std::vector<Var>& load_vars,
                                      const std::vector<ir::Expr>& indices) {
  return CopyedReplaceExpr(source, load_vars, indices);
}

template <typename FusionOp>
static void ReplaceDownstreamLoadExprWithUpstreamComputeBody(
    const FusionOp& upstream,
    const ir::Expr& downstream_load_expr,
    ir::Expr* downstream_body) {
  ComposeUtils::SubstitudeTargetExprWithDestExpr(
      downstream_load_expr,
      ComposeUtils::SubstitudeIndexVector(
          upstream.GetStoreValue(),
          upstream.GetOutputIters(),
          downstream_load_expr.As<ir::Load>()->indices),
      downstream_body);
}

}  // namespace ComposeUtils

namespace SearchUtils {

// 1. search by type. DONE
// 2. search by value. DONE
// 3. search by father. TODO

using ExprSet = std::vector<ir::Expr>;
using Func = std::function<ExprSet(const ir::Expr& x)>;
struct Mapping {
  Func f_;
  explicit Mapping(Func f) { f_ = f; }
  ExprSet operator()(const ir::Expr& x) const { return f_(x); }
  ir::Expr GetSingle(const ir::Expr& x) const {
    const auto& o = this->operator()(x);
    if (o.size() != 1) {
      PADDLE_THROW("Try to get single result, but we get %d.", o.size());
    }
    return *o.begin();
  }
  Mapping operator*(Mapping x) {
    auto new_f = [=](const ir::Expr& e) -> ExprSet {
      const auto& rs = this->f_(e);
      ExprSet res;
      for (const auto& r : rs) {
        const auto& x_res = x.f_(r);
        res.insert(res.begin(), x_res.begin(), x_res.end());
      }
      return res;
    };
    return Mapping(std::function(new_f));
  }
};

Mapping Identity =
    Mapping([](const ir::Expr& e) { return std::vector<ir::Expr>{e}; });

template <typename Teller>
Func Collector(Teller t) {
  return [=](const ir::Expr& x) -> ExprSet {
    const auto& rs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(x, t);
    return std::vector(rs.begin(), rs.end());
  };
}

template <typename FilterFunc>
Func FilterMaker(FilterFunc t) {
  return [=](const ir::Expr& x) -> ExprSet {
    if (t(x)) {
      return {x};
    }
    return {};
  };
}

Mapping Store2Value = Mapping([](const ir::Expr& e) -> ExprSet {
  if (e.As<ir::Store>()) {
    return {e.As<ir::Store>()->value};
  }
  return {};
});

Mapping ScheduleBlock2Body = Mapping([](const ir::Expr& e) -> ExprSet {
  if (e.As<ir::ScheduleBlock>()) {
    return {e.As<ir::ScheduleBlock>()->body};
  }
  return {};
});

Mapping ScheduleBlockIsInit = FilterMaker([](const ir::Expr& e) -> bool {
  return e.As<ir::ScheduleBlock>() && e.As<ir::ScheduleBlock>()->name.find(
                                          "_reduce_init") == std::string::npos;
});

Mapping ScheduleBlockIsNotInit = FilterMaker([](const ir::Expr& e) -> bool {
  return e.As<ir::ScheduleBlock>() && e.As<ir::ScheduleBlock>()->name.find(
                                          "_reduce_init") != std::string::npos;
});

Mapping ChildScheduleBlocks =
    Collector([](const ir::Expr* e) { return e->As<ir::ScheduleBlock>(); });

Mapping ChildScheduleBlockRealizes = Collector(
    [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); });

Mapping ChildStores =
    Collector([](const ir::Expr* e) { return e->As<ir::Store>(); });

Mapping ChildLoads =
    Collector([](const ir::Expr* e) { return e->As<ir::Load>(); });

template <class Transformer>
void FindAndReplace(ir::Expr* body,
                    const SearchUtils::Mapping& map,
                    const Transformer& transformer,
                    bool force_single_target = true) {
  ExprSet found_targets;
  if (force_single_target && found_targets.size() != 1) {
    found_targets = {map.GetSingle(*body)};
  } else {
    found_targets = map(*body);
  }
  for (const auto& expr : found_targets) {
    MappingTargetExprToDestExprMutator(expr, transformer(expr))(body);
  }
}

Mapping ChildFors =
    Collector([](const ir::Expr* e) { return e->As<ir::For>(); });

Mapping FindFather(const ir::Expr& root) {
  const auto& f = [&](const auto& child) -> ExprSet {
    Mapping find_child =
        Collector([child](const ir::Expr* e) { return *e == child; });
    const auto& father_collector = Collector(
        [&](const ir::Expr* current) { return !find_child(*current).empty(); });
    return father_collector(root);
  };
  return Mapping(f);
}

}  // namespace SearchUtils

namespace TransformerUtils {
using TransformFunc = std::function<ir::Expr(ir::Expr)>;
struct Transformer {
  TransformFunc f_;
  explicit Transformer(TransformFunc f) { f_ = f; }
  ir::Expr operator()(const ir::Expr& x) const { return f_(x); }
  Transformer operator*(const Transformer& x) {
    auto new_f = [=](const ir::Expr& e) -> ir::Expr {
      const auto& rs = this->f_(e);
      return x.f_(rs);
    };
    return Transformer(std::function(new_f));
  }
};

Transformer Identity = Transformer([](const ir::Expr& e) { return e; });
Transformer WrapForTransformer(const ir::Var& v) {
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
  return Transformer(f);
}

Transformer WrapForsTransformer(const std::vector<ir::Var>& vs) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    Transformer t = Identity;
    for (auto v = vs.rbegin(); v != vs.rend(); v++) {
      t = WrapForTransformer(*v) * t;
    }
    return t(e);
  };
  return Transformer(f);
}

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars) {
  int i = 0;
  std::vector<ir::Var> vars;
  for (const auto& v : block_vars) {
    vars.emplace_back("inner_block_" + std::to_string(i++));
  }
  return vars;
}

Transformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                 const std::vector<ir::Var>& dest_vars) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopyedReplaceExpr(
        e,
        target_vars,
        std::vector<ir::Expr>(dest_vars.begin(), dest_vars.end()));
  };
  return Transformer(f);
}

Transformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                 const ir::Tensor& tensor) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    if (e.As<ir::ScheduleBlock>()) {
      PADDLE_THROW("please input a non-schedule block expr.");
    }
    const auto& inner_block_var = CreateInnerBlockVars(block_vars);
    const auto& replaced_e =
        ChangeVarTransformer(block_vars, inner_block_var)(e);
    const auto& schedule_block =
        ir::ScheduleBlock::Make(block_vars, {}, {}, tensor->name, replaced_e);
    const auto& schedule_realizer = ir::ScheduleBlockRealize::Make(
        std::vector<ir::Expr>(block_vars.begin(), block_vars.end()),
        schedule_block);
    return schedule_realizer;
  };
  return Transformer(f);
}

}  // namespace TransformerUtils

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

template <class A, class C, class Func>
void SequenceMutator(const std::vector<A>& as, C* acc, const Func& mutator) {
  VLOG(4) << "SequenceTransform Init: " << acc;
  for (int i = 0; i < as.size(); ++i) {
    mutator(as[i], acc);
    VLOG(4) << "SequenceTransform Iter: " << acc;
  }
}

static bool IsAdjecent(const ir::Expr& upstream, const ir::Expr& downstream) {
  // 1. Get inputs / output from Expr, then we can tell whether they are
  // adjecent.
  std::set<Expr> upstream_stores =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          upstream, [](const Expr* expr) {
            return expr->As<ir::Store>() &&
                   expr->As<ir::Store>()->is_addr_tensor();
          });
  // don't support multi-output yet.
  PADDLE_ENFORCE(upstream_stores.size() == 1,
                 "The expr of injective should have only one store");

  std::set<Expr> downstream_loads =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          downstream, [](const Expr* expr) {
            return expr->As<ir::Load>() &&
                   expr->As<ir::Load>()->is_addr_tensor();
          });

  for (const auto& upstream_store : upstream_stores) {
    for (const auto& downstream_load : downstream_loads) {
      if (upstream_store.As<ir::Store>()->tensor.As<ir::_Tensor_>()->name ==
          downstream_load.As<ir::Load>()->tensor.As<ir::_Tensor_>()->name) {
        return true;
      }
    }
  }
  return false;
}

inline bool IsTrivialKind(OpPatternKind kind) {
  return kind == OpPatternKind::kElementWise ||
         kind == OpPatternKind::kBroadcast || kind == OpPatternKind::kInjective;
}

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns) {
  if (VLOG_IS_ON(4)) {
    for (const auto& func : op_compute_bodies) {
      VLOG(4) << "TrivialOpFusion: {FuncBody is} :" << func;
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

struct TrivialOp {
 public:
  explicit TrivialOp(const ir::Expr& origin_func_body) {
    func_body = ir::ir_utils::IRCopy(origin_func_body);
  }

  TrivialOp(const TrivialOp& trivial_op) {
    func_body = trivial_op.GetFuncBody();
  }

  ir::Expr* GetFuncBodyPointer() { return &func_body; }

  ir::Expr GetStoreValue() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  std::vector<ir::Var> GetOutputIters() const {
    return ComposeUtils::GetOutputIters(
        GetSingleStoreExpr(func_body).As<ir::Store>()->indices);
  }

  ir::Expr* GetStoreValuePointer() const {
    return &GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  ir::Expr GetFuncBody() const { return func_body; }

  ir::Tensor GetOutputTensor() const {
    return GetSingleStoreExpr(func_body)
        .As<ir::Store>()
        ->tensor.as_tensor_ref();
  }

  std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Tensor& tensor) const {
    VLOG(4) << "Start GetEachTensorLoadExpr: " << tensor;
    std::set<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
        GetStoreValue(), [&tensor](const Expr* expr) {
          return expr->As<ir::Load>() &&
                 expr->As<ir::Load>()->is_addr_tensor() &&
                 expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     tensor->name;
        });
    for (auto& t : load_exprs) {
      VLOG(4) << "GetEachTensorLoadExpr: " << t << " " << t.ptr();
    }
    return std::vector(load_exprs.begin(), load_exprs.end());
  }

  ir::Expr GetComputeExpr() const { return GetStoreValue(); }

 private:
  ir::Expr func_body;

  ir::Expr GetSingleStoreExpr(const ir::Expr& body) const {
    const auto& store_tensor_exprs = ComposeUtils::GetStoreFromBody(body);
    PADDLE_ENFORCE(store_tensor_exprs.size() == 1,
                   "TrivialOp must store for output only once.");
    return *(store_tensor_exprs.begin());
  }
};

struct ReduceOp {
 public:
  explicit ReduceOp(const ir::Expr& origin_func_body) {
    func_body = ir::ir_utils::IRCopy(origin_func_body);
  }

  ReduceOp(const ReduceOp& reduce_op) { func_body = reduce_op.GetFuncBody(); }

  ir::Expr GetStoreValue() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  ir::Expr* GetStoreValuePointer() const {
    return &GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  std::vector<ir::Var> GetOutputIters() const {
    return ComposeUtils::GetOutputIters(
        GetSingleStoreExpr(func_body).As<ir::Store>()->indices);
  }

  ir::Expr GetFuncBody() const { return func_body; }

  ir::Expr* GetFuncBodyPointer() { return &func_body; }

  ir::Tensor GetOutputTensor() const {
    return GetSingleStoreExpr(func_body)
        .As<ir::Store>()
        ->tensor.as_tensor_ref();
  }

  std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Tensor& tensor) const {
    VLOG(4) << "Start GetEachTensorLoadExpr: " << tensor;
    std::set<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
        GetStoreValue(), [&tensor](const Expr* expr) {
          return expr->As<ir::Load>() &&
                 expr->As<ir::Load>()->is_addr_tensor() &&
                 expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     tensor->name;
        });
    for (auto& t : load_exprs) {
      VLOG(4) << "GetEachTensorLoadExpr: " << t << " " << t.ptr();
    }
    return std::vector(load_exprs.begin(), load_exprs.end());
  }

  ir::Expr GetComputeExpr() const {
    VLOG(4) << "GetComputeExpr";
    return (SearchUtils::ChildScheduleBlocks *
            SearchUtils::ScheduleBlockIsNotInit * SearchUtils::ChildStores *
            SearchUtils::Store2Value)
        .GetSingle(GetFuncBody());
  }
  ir::Expr GetInitExpr() const {
    return (SearchUtils::ChildScheduleBlocks *
            SearchUtils::ScheduleBlockIsInit * SearchUtils::ChildStores *
            SearchUtils::Store2Value)
        .GetSingle(GetFuncBody());
  }

  std::vector<ir::Var> GetAllIterVars() const {
    ir::Expr compute_schedule_block_realize =
        (SearchUtils::ChildScheduleBlock * SearchUtils::ScheduleBlockIsNotInit *
         SearchUtils::FindFather(GetFuncBody()) *
         FilterMaker([](const ir::Expr& e) -> bool {
           return e.As<ir::ScheduleBlockRealize>()
         })).GetSingle(GetFuncBody());

    const std::vector<Expr>& all_iter_expr =
        compute_schedule_block_realize.As<ir::ScheduleBlockRealize>()
            ->iter_values;
    std::vector<ir::Var> all_iter_vars;

    std::transform(all_iter_expr.begin(),
                   all_iter_expr.end(),
                   all_iter_vars.begin(),
                   [](const Expr& expr) { return expr.as_var_ref(); });

    return all_iter_vars;
  }

  std::vector<ir::Var> GetReduceIterVars() const {
    ir::Expr init_schedule_block_realize =
        (SearchUtils::ChildScheduleBlock * SearchUtils::ScheduleBlockIsInit *
         SearchUtils::FindFather(GetFuncBody()) *
         FilterMaker([](const ir::Expr& e) -> bool {
           return e.As<ir::ScheduleBlockRealize>()
         })).GetSingle(GetFuncBody());

    // Get Iter Vars from initial schedule block
    const std::vector<Expr>& init_iter_expr =
        init_schedule_block_realize.As<ir::ScheduleBlockRealize>()->iter_values;
    std::vector<ir::Var> init_iter_vars;
    std::transform(init_iter_expr.begin(),
                   init_iter_expr.end(),
                   init_iter_vars.begin(),
                   [](const Expr& expr) { return expr.as_var_ref(); });

    // Iter Vars not appearing in init_iter_vars are pushed into
    // reduce_iter_vars
    std::vector<ir::Var> all_iter_vars = GetAllIterVars();
    std::vector<ir::Var> reduce_iter_vars;
    for (auto& iter_var : all_iter_vars) {
      if (!(std::find(init_iter_vars.begin(), init_iter_vars.end(), iter_var) !=
            init_iter_vars.end())) {
        reduce_iter_vars.push_back(iter_var);
      }
    }

    return reduce_iter_vars;
  }

 private:
  ir::Expr func_body;

  ir::Expr GetSingleStoreExpr(const ir::Expr& body) const {
    VLOG(4) << "GetComputeExpr";
    return (SearchUtils::ChildScheduleBlocks *
            SearchUtils::ScheduleBlockIsNotInit * SearchUtils::ChildStores)
        .GetSingle(GetFuncBody());
  }
};

using FusibleOp = std::variant<ReduceOp, TrivialOp>;

ir::Expr GetExpr(const FusibleOp& op) {
  return std::visit([](auto&& arg) { return arg.GetFuncBody(); }, op);
}

std::vector<ir::Expr> GetEachTensorLoadExpr(const FusibleOp& op,
                                            const ir::Tensor& tensor) {
  return std::visit(
      [&](auto&& arg) { return arg.GetEachTensorLoadExpr(tensor); }, op);
}

ir::Tensor GetOutputTensor(const FusibleOp& op) {
  return std::visit([&](auto&& arg) { return arg.GetOutputTensor(); }, op);
}

std::vector<ir::Var> GetOutputIters(const FusibleOp& op) {
  return std::visit([&](auto&& arg) { return arg.GetOutputIters(); }, op);
}

ir::Expr* GetFuncBodyPointer(FusibleOp op) {
  return std::visit([&](auto&& arg) { return arg.GetFuncBodyPointer(); }, op);
}

ir::Expr CopyReduceBody(const FusibleOp& downstream, const ReduceOp& upstream) {
  struct Visitor {
    ir::Expr operator()(const ReduceOp& op) {
      return ir::ir_utils::IRCopy(op.GetFuncBody());
    }
    ir::Expr operator()(const TrivialOp& op) {
      PADDLE_THROW("TrivialOp cannot be copied.");
    }
  };
  return std::visit(Visitor(), downstream);
}

ir::Expr CreateReduceExpr(
    const std::vector<ir::Var>& output_iters,
    const std::vector<ir::Var>& reduce_iters,
    const ir::Expr& init_body,    // relay on output_iters
    const ir::Expr& reduce_body,  // relay on output_iters + reduce_iters
    const ir::Tensor& new_write_tensor) {
  const auto& init_schedule_block = TransformerUtils::WrapScheduleRealizer(
      output_iters, new_write_tensor)(init_body);
  const auto& reduce_schedule_block =
      (TransformerUtils::WrapScheduleRealizer(output_iters, new_write_tensor) *
       TransformerUtils::WrapForsTransformer(reduce_iters))(reduce_body);
  const auto& gather_body = ir::Block::Make(
      std::vector<ir::Expr>({init_schedule_block, reduce_schedule_block}));
  return TransformerUtils::WrapForsTransformer(output_iters)(gather_body);
}

struct FusionNode {
  FusibleOp fusible_op;
  ::pir::Operation* expr_related_op;

  std::unordered_map<FusionNode*, ::pir::Value> upstream;
  std::unordered_map<FusionNode*, ::pir::Value> downstream;

  explicit FusionNode(FusibleOp fusible_op) : fusible_op(fusible_op) {}

  static std::string GetTensorCounter() {
    static int i = 0;
    return std::to_string(i++);
  }

  void replace_topo_structure_of_fused_nodes(FusionNode* fused_up_node,
                                             FusionNode* fused_down_node) {
    upstream.insert(fused_up_node->upstream.begin(),
                    fused_up_node->upstream.end());
    upstream.insert(fused_down_node->upstream.begin(),
                    fused_down_node->upstream.end());
    upstream.erase(fused_up_node);

    downstream.insert(fused_up_node->downstream.begin(),
                      fused_up_node->downstream.end());
    downstream.insert(fused_down_node->downstream.begin(),
                      fused_down_node->downstream.end());
    downstream.erase(fused_down_node);

    expr_related_op = fused_down_node->expr_related_op;

    for (const auto& pair_data : upstream) {
      FusionNode* upstream_node = pair_data.first;
      ::pir::Value related_value = pair_data.second;
      if (upstream_node->downstream.find(fused_up_node) !=
          upstream_node->downstream.end()) {
        upstream_node->downstream.erase(fused_up_node);
      }
      if (upstream_node->downstream.find(fused_down_node) !=
          upstream_node->downstream.end()) {
        upstream_node->downstream.erase(fused_down_node);
      }
      upstream_node->downstream[this] = related_value;
    }

    for (const auto& pair_data : downstream) {
      FusionNode* downstream_node = pair_data.first;
      ::pir::Value related_value = pair_data.second;
      if (downstream_node->upstream.find(fused_up_node) !=
          downstream_node->upstream.end()) {
        downstream_node->upstream.erase(fused_up_node);
      }
      if (downstream_node->upstream.find(fused_down_node) !=
          downstream_node->upstream.end()) {
        downstream_node->upstream.erase(fused_down_node);
      }
      downstream_node->upstream[this] = related_value;
    }
  }

  bool IsTrivial() const {
    return std::holds_alternative<TrivialOp>(fusible_op);
  }
};

template <class DownStreamOp>
DownStreamOp TrivalxOther_Fusion(TrivialOp upstream, DownStreamOp downstream) {
  VLOG(4) << "Trivial x OtherFusion begin.";

  const auto& replaced_tensor = upstream.GetOutputTensor();
  VLOG(4) << "upstream is " << upstream.GetFuncBody();
  VLOG(4) << "downstream is " << downstream.GetFuncBody();

  DownStreamOp fused(ir::ir_utils::IRCopy(downstream.GetFuncBody()));
  SequenceMutator(
      fused.GetEachTensorLoadExpr(replaced_tensor),
      fused.GetStoreValuePointer(),
      [&](const ir::Expr& downstream_load_expr, ir::Expr* downstream_body) {
        ComposeUtils::ReplaceDownstreamLoadExprWithUpstreamComputeBody(
            upstream, downstream_load_expr, downstream_body);
      });

  VLOG(4) << "After mutate, store_value is: " << fused.GetFuncBody();
  VLOG(4) << "TTFusion end:\n" << fused.GetFuncBody();
  return DownStreamOp(fused.GetFuncBody());
}

bool CheckAllLoopRangeEq(ReduceOp reduce_upper, TrivialOp trivial_down) {}

std::vector<FusibleOp> TransformReduceLoopRange(ReduceOp upstream,
                                                FusibleOp downstream) {
  VLOG(4) << "RRTransform begin";
  VLOG(4) << "Upstream is " << upstream.GetFuncBody();
  // CHECK(ComposeUtils::CheckIterEq(upstream.GetReduceIters(),
  // downstream.GetReduceIters()));
  const auto& load_upstream_expr =
      GetEachTensorLoadExpr(downstream, upstream.GetOutputTensor());
  VLOG(4) << "step 1";
  std::vector<FusibleOp> results;
  ir::Tensor downstream_output_tensor = GetOutputTensor(downstream);
  VLOG(4) << "step 1";
  const auto create_new_tensor = [&](const ir::Tensor& downstream_load_tensor) {
    return ir::Tensor(
        downstream_load_tensor->name + FusionNode::GetTensorCounter(),
        downstream_load_tensor->type(),
        downstream_output_tensor.self()->sym_shape,
        downstream_load_tensor.self()->sym_domain,
        downstream_load_tensor.self()->operation,
        downstream_output_tensor.self()->reduce_axis);
  };
  VLOG(4) << "step 1";

  for (const auto& load_tensor : load_upstream_expr) {
    VLOG(4) << "step 1";
    const auto& new_tensor =
        create_new_tensor(load_tensor.As<ir::Load>()->tensor.as_tensor_ref());
    VLOG(4) << "step 1";
    ir::Expr new_reduce = CreateReduceExpr(
        GetOutputIters(downstream),
        upstream.GetReduceIters(),
        upstream.GetInitExpr(),
        ComposeUtils::CopyedReplaceExpr(upstream.GetComputeExpr(),
                                        upstream.GetOutputIters(),
                                        load_tensor.As<ir::Load>()->indices),
        new_tensor);
    ComposeUtils::MappingTargetExprToDestExprMutator(
        load_tensor.As<ir::Load>()->tensor,
        Expr(new_tensor))(GetFuncBodyPointer(downstream));
    results.emplace_back(ReduceOp(new_reduce));
  }
  return results;
}

FusibleOp TrivialFusion(FusionNode* upstream, FusionNode* downstream) {
  CHECK(upstream->IsTrivial());
  if (downstream->IsTrivial()) {
    return TrivalxOther_Fusion(std::get<TrivialOp>(upstream->fusible_op),
                               std::get<TrivialOp>(downstream->fusible_op));
  } else {
    return TrivalxOther_Fusion(std::get<TrivialOp>(upstream->fusible_op),
                               std::get<ReduceOp>(downstream->fusible_op));
  }
}

ir::Expr ExtendFor(ir::Expr target, std::vector<ir::Expr> extended_fors) {
  ir::Expr loop_body = target.As<ir::For>()->body;
  for (auto for_expr = extended_fors.rbegin(); for_expr != extended_fors.rend();
       for_expr++) {
    loop_body = TransformerUtils::WrapForTransformer(
        (*for_expr).As<ir::For>()->loop_var)(loop_body);
  }
  return TransformerUtils::WrapForTransformer(target.As<ir::For>()->loop_var)(
      loop_body);
}

FusibleOp SinkTrivialLoopAlign(TrivialOp trivial_op, ReduceOp reduce_op) {
  ir::Expr new_trivial_body = ir::ir_utils::IRCopy(trivial_op.GetFuncBody());

  ir::Expr reduce_init = reduce_op.GetInitExpr();  // Mapping.
  std::vector<ir::Expr> reduce_for =
      (SearchUtils::ChildFors *
       SearchUtils::FindFather(reduce_init))(reduce_op.GetFuncBody());
  ir::Expr trivial_last_for = SearchUtils::ChildFors(new_trivial_body).back();

  ComposeUtils::SubstitudeTargetExprWithDestExpr(
      trivial_last_for,
      ExtendFor(trivial_last_for, reduce_for),
      &new_trivial_body);

  return TrivialOp(new_trivial_body);
}

std::vector<FusibleOp> ReduceTransformRecursive(FusibleOp root_op,
                                                FusionNode* fusion_tree) {
  VLOG(4) << "ReduceTransformRecursive: " << *GetFuncBodyPointer(root_op);
  std::vector<FusibleOp> result;
  for (auto& pair : fusion_tree->upstream) {
    auto transformed_nodes = TransformReduceLoopRange(
        std::get<ReduceOp>(pair.first->fusible_op), root_op);
    for (auto& node : transformed_nodes) {
      auto child_flatten = ReduceTransformRecursive(node, pair.first);
      result.insert(result.end(), child_flatten.begin(), child_flatten.end());
    }
  }
  result.push_back(
      std::holds_alternative<TrivialOp>(root_op)
          ? SinkTrivialLoopAlign(
                std::get<TrivialOp>(root_op),
                std::get<ReduceOp>(
                    fusion_tree->upstream.begin()->first->fusible_op))
          : root_op);
  return result;
}

std::vector<FusibleOp> ReduceTransform(FusionNode* downstream) {
  if (downstream->IsTrivial() && downstream->upstream.empty()) {
    return {downstream->fusible_op};
  }
  auto reduces = ReduceTransformRecursive(downstream->fusible_op, downstream);
  return reduces;
}

FusibleOp CreateFusibleOp(ir::Expr compute_body, OpPatternKind op_pattern) {
  if (IsTrivialKind(op_pattern)) {
    return TrivialOp(compute_body);
  } else {
    return ReduceOp(compute_body);
  }
}

struct FusionGraph {
  explicit FusionGraph(const std::vector<::pir::Operation*>& ops,
                       const std::vector<ir::Expr>& op_compute_bodies) {
    // shardable_axes_ = InferShardableAxes(ops);
    VLOG(4) << "CreateFusionGraph";

    const auto& op_patterns = GetOpPatternKindVector(ops);
    CheckFusionInputValid(op_compute_bodies, op_patterns);

    std::unordered_map<::pir::Operation*, FusionNode*> op_to_node_map;

    for (int i = 0; i < ops.size(); ++i) {
      FusionNode* node =
          new FusionNode(CreateFusibleOp(op_compute_bodies[i], op_patterns[i]));
      op_to_node_map[ops[i]] = node;
      all_fusion_nodes_.emplace(node);
      node->expr_related_op = ops[i];
    }

    for (::pir::Operation* op : ops) {
      FusionNode* cur_node = op_to_node_map[op];

      // add upstream nodes
      for (int i = 0; i < op->num_operands(); ++i) {
        ::pir::Value related_value = op->operand_source(i);
        ::pir::Operation* input_op = related_value.defining_op();
        if (op_to_node_map.find(input_op) != op_to_node_map.end()) {
          FusionNode* upstream_node = op_to_node_map[input_op];
          cur_node->upstream[upstream_node] = related_value;
          upstream_node->downstream[cur_node] = related_value;
        }
      }

      // add downstream nodes
      for (int i = 0; i < op->num_results(); ++i) {
        ::pir::Value related_value = op->result(i);
        for (auto consumer_it = related_value.use_begin();
             consumer_it != related_value.use_end();
             ++consumer_it) {
          ::pir::Operation* output_op = consumer_it->owner();
          if (op_to_node_map.find(output_op) != op_to_node_map.end()) {
            FusionNode* downstream_node = op_to_node_map[output_op];
            cur_node->downstream[downstream_node] = related_value;
            downstream_node->upstream[cur_node] = related_value;
          }
        }
      }

      if (cur_node->upstream.empty()) {
        entrance_nodes_.emplace(cur_node);
      }

      if (cur_node->downstream.empty()) {
        exit_nodes_.emplace(cur_node);
      }
    }

    VLOG(4) << "FusionGraph Created, fusion node size: "
            << all_fusion_nodes_.size();
  }

  ~FusionGraph() {
    for (FusionNode* node : all_fusion_nodes_) {
      delete node;
    }
  }

  std::vector<ir::Expr> DoFusion() {
    VLOG(4) << "Start Trivial Fusion";
    DoTrivialFusion();
    VLOG(4) << "Start R + T and R + R Fusion";
    ReduceLoopTranform();
    return GetExprResults();
  }

 private:
  FusionNode* FindTrivialFusibleNode() {
    for (FusionNode* node : all_fusion_nodes_) {
      if (node->IsTrivial() && !node->downstream.empty()) {
        return node;
      }
    }
    return nullptr;
  }

  void DoTrivialFusion() {
    FusionNode* upstream = nullptr;
    // use funcion to get upstream and downstream is save here
    // cause we might delete Nodes in this process
    while ((upstream = FindTrivialFusibleNode()) != nullptr) {
      std::unordered_map<FusionNode*, ::pir::Value> fusion_candidate =
          upstream->downstream;
      upstream->downstream.clear();
      for (const auto& pair_data : fusion_candidate) {
        FusionNode* downstream = pair_data.first;
        FusionNode* new_node =
            new FusionNode(TrivialFusion(upstream, downstream));
        new_node->replace_topo_structure_of_fused_nodes(upstream, downstream);
        AppendNode(new_node);
        RemoveNode(downstream);
      }
      RemoveNode(upstream);
    }
  }

  void ReduceLoopTranform() {
    for (FusionNode* node : exit_nodes_) {
      auto fusion_nodes = ReduceTransform(node);
      fusion_results_.insert(
          fusion_results_.end(), fusion_nodes.begin(), fusion_nodes.end());
    }
  }

  std::vector<ir::Expr> GetExprResults() {
    std::vector<ir::Expr> output_exprs;
    for (const auto& node : fusion_results_) {
      output_exprs.emplace_back(GetExpr(node));
    }
    return output_exprs;
  }

  void RemoveNode(FusionNode* node) {
    if (all_fusion_nodes_.find(node) != all_fusion_nodes_.end()) {
      all_fusion_nodes_.erase(node);
    }
    if (entrance_nodes_.find(node) != entrance_nodes_.end()) {
      entrance_nodes_.erase(node);
    }
    if (exit_nodes_.find(node) != exit_nodes_.end()) {
      exit_nodes_.erase(node);
    }
    delete node;
  }

  void AppendNode(FusionNode* node) {
    all_fusion_nodes_.emplace(node);
    if (node->upstream.empty()) {
      entrance_nodes_.emplace(node);
    }

    if (node->downstream.empty()) {
      exit_nodes_.emplace(node);
    }
  }

  FusionNode* FindReduceUpstream(FusionNode* node) {
    for (const auto& pair_data : node->upstream) {
      FusionNode* upstream = pair_data.first;
      if (!upstream->IsTrivial()) {
        return upstream;
      }
    }
    return nullptr;
  }

 private:
  std::unordered_set<FusionNode*> all_fusion_nodes_;
  std::vector<FusibleOp> fusion_results_;
  std::unordered_set<FusionNode*> entrance_nodes_;
  std::unordered_set<FusionNode*> exit_nodes_;

  // std::unordered_map<::pir::Value, ShardableAxes> shardable_axes_;
};

}  // namespace trivial_fusion_detail

std::vector<ir::Expr> OperationFusion(
    const std::vector<::pir::Operation*>& ops,
    const std::vector<ir::Expr>& op_compute_bodies) {
  trivial_fusion_detail::FusionGraph graph =
      trivial_fusion_detail::FusionGraph(ops, op_compute_bodies);
  auto output = graph.DoFusion();
  VLOG(4) << "Fusion Result: output size is " << output.size();
  for (const auto& expr : output) {
    VLOG(4) << expr;
  }
  return output;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
