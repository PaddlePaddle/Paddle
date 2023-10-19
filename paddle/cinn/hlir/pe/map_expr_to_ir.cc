// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"

#include <unordered_map>
#include <vector>

#include "paddle/cinn/adt/inline_translator.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_map_expr_enable_inline);
namespace cinn::adt {

namespace {

using Node2LoweredFuncs =
    std::unordered_map<hlir::framework::Node*, std::vector<ir::LoweredFunc>>;

using TensorIteratorExpr4TensorT =
    std::function<adt::List<adt::TensorIteratorExpr>(const adt::Tensor&)>;
using LoopDescriptor4LoopIteratorT =
    std::function<adt::LoopDescriptor(const adt::Iterator&)>;

class MapExprToIrTranslator {
 public:
  explicit MapExprToIrTranslator(const MapExpr& map_expr,
                                 const Node2LoweredFuncs& node2lowered_funcs)
      : map_expr_(map_expr), node2lowered_funcs_(&node2lowered_funcs) {
    const auto& [anchored_map_stmts, _0, _1] = map_expr.tuple();
    CHECK_EQ(anchored_map_stmts->size(), 1);
    TensorIteratorExpr4Tensor = std::get<4>(anchored_map_stmts->at(0).tuple());
    LoopDescriptor4LoopIterator =
        std::get<5>(anchored_map_stmts->at(0).tuple());
  }

  ir::Expr Translate() const {
    VLOG(1) << "Translate MapExpr: ";
    VLOG(1) << ToTxtString(map_expr_, "");
    return ir::Block::Make({Translate(map_expr_)});
  }

 private:
  ir::Expr GetStoreExprForOp(const hlir::framework::Node* op) const {
    const auto& iter =
        node2lowered_funcs_->find(const_cast<hlir::framework::Node*>(op));
    CHECK(iter != node2lowered_funcs_->end());
    const auto& lowered_funcs = iter->second;
    CHECK_EQ(lowered_funcs.size(), 1);
    std::optional<ir::Expr> ret{std::nullopt};
    VisitEachStoreExpr(lowered_funcs.at(0), [&](const ir::Expr& expr) {
      CHECK(!ret.has_value());
      ret = expr;
    });
    CHECK(ret.has_value());
    return ret.value();
  }

  ir::Expr GetStoreExprForOp(
      tReduceInit<const hlir::framework::Node*> op) const {
    const auto& iter = node2lowered_funcs_->find(
        const_cast<hlir::framework::Node*>(op.value()));
    CHECK(iter != node2lowered_funcs_->end());
    const auto& lowered_funcs = iter->second;
    CHECK_EQ(lowered_funcs.size(), 1);
    std::vector<ir::Expr> stores{};
    VisitEachStoreExpr(lowered_funcs.at(0), [&](const ir::Expr& expr) {
      stores.emplace_back(expr);
    });
    CHECK_EQ(stores.size(), 2);
    return stores.at(0);
  }

  ir::Expr GetStoreExprForOp(
      tReduceAcc<const hlir::framework::Node*> op) const {
    const auto& iter = node2lowered_funcs_->find(
        const_cast<hlir::framework::Node*>(op.value()));
    CHECK(iter != node2lowered_funcs_->end());
    const auto& lowered_funcs = iter->second;
    CHECK_EQ(lowered_funcs.size(), 1);
    std::vector<ir::Expr> stores{};
    VisitEachStoreExpr(lowered_funcs.at(0), [&](const ir::Expr& expr) {
      stores.emplace_back(expr);
    });
    CHECK_EQ(stores.size(), 2);
    return stores.at(1);
  }

  std::optional<ir::Expr> GetStoreExprImpl(
      const OpCall<OpExpr>& op_expr) const {
    const auto& [op, _] = op_expr.tuple();
    return std::visit([&](const auto& impl) { return GetStoreExprForOp(impl); },
                      op.variant());
  }

  std::optional<ir::Expr> GetStoreExprImpl(const Load<Tensor>& load) const {
    return std::nullopt;
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::optional<ir::Expr> GetStoreExpr(const OpExpr& op_expr) const {
    return std::visit([&](const auto& impl) { return GetStoreExprImpl(impl); },
                      op_expr.variant());
  }

  template <typename DoEachT>
  void VisitEachStoreExpr(const ir::Expr& expr, const DoEachT& DoEach) const {
    switch (expr.node_type()) {
      case ir::IrNodeTy::_LoweredFunc_:
        VisitEachStoreExpr(expr.as_lowered_func()->body, DoEach);
        break;
      case ir::IrNodeTy::Block:
        for (const auto& stmt : expr.As<ir::Block>()->stmts) {
          VisitEachStoreExpr(stmt, DoEach);
        }
        break;
      case ir::IrNodeTy::ScheduleBlockRealize:
        VisitEachStoreExpr(expr.As<ir::ScheduleBlockRealize>()->schedule_block,
                           DoEach);
        break;
      case ir::IrNodeTy::ScheduleBlock:
        VisitEachStoreExpr(expr.As<ir::ScheduleBlock>()->body, DoEach);
        break;
      case ir::IrNodeTy::For:
        VisitEachStoreExpr(expr.As<ir::For>()->body, DoEach);
        break;
      case ir::IrNodeTy::Store:
        DoEach(expr);
        break;
      default:
        LOG(FATAL) << "Visit node_type = " << expr.node_type()
                   << ", not supported!";
        break;
    }
  }

  template <typename DoEachT>
  void VisitEachStmt(const List<Stmt>& stmts, const DoEachT& DoEach) const {
    for (const auto& stmt : *stmts) {
      DoEach(stmt);
    }
  }

  ir::Expr Translate(const MapExpr& map_expr) const {
    const auto& [anchored_map_stmts, _0, _1] = map_expr.tuple();
    CHECK_EQ(anchored_map_stmts->size(), 1);
    return Translate(anchored_map_stmts->at(0));
  }

  ir::Expr Translate(const AnchoredMapStmt& anchored_map_stmt) const {
    const MapStmt<Stmt>& map_stmt = std::get<0>(anchored_map_stmt.tuple());
    ir::Expr ret = Translate(map_stmt);
    ret = ir::ScheduleBlock::Make({}, {}, {}, "root", ret);
    ret = ir::ScheduleBlockRealize::Make({}, ret);
    return ret;
  }

  using InternalLeafStmt = Store<Tensor, OpCall<Load<Tensor>>>;
  using InternalStmt = Tree<MapStmt, InternalLeafStmt>;

  InternalStmt ConvertToInternalStmtImpl(const OpStmt& op_stmt) const {
    const auto& [op, inputs, outputs] = op_stmt.tuple();
    CHECK_EQ(outputs.value()->size(), 1);
    List<Load<Tensor>> loads{};
    for (const auto& in : *inputs.value()) {
      loads->emplace_back(Load<Tensor>{in});
    }
    OpCall<Load<Tensor>> op_call{op, loads};
    return InternalLeafStmt{outputs.value()->at(0), op_call};
  }

  InternalStmt ConvertToInternalStmtImpl(const MapStmt<Stmt>& map_stmt) const {
    const auto& [iterators, stmts] = map_stmt.tuple();
    List<InternalStmt> children{};
    for (const auto& stmt : *stmts) {
      children->emplace_back(ConvertToInternalStmt(stmt));
    }
    return MapStmt<InternalStmt>{iterators, children};
  }

  InternalStmt ConvertToInternalStmt(const Stmt& stmt) const {
    return std::visit(
        [&](const auto& impl) { return ConvertToInternalStmtImpl(impl); },
        stmt.variant());
  }

  InlineStmt ConvertToInlineStmt(const InternalStmt& internal_stmt) const {
    return InlineTranslator<MapStmt, OpCall, Tensor>::Call(internal_stmt);
  }

  std::optional<ir::Expr> TranslateImpl(
      const Load<Tensor>& load,
      const std::optional<Tensor>& opt_output_tensor) const {
    return std::nullopt;
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const OpCall<OpExpr>& op_call) const {
    LOG(FATAL) << "Dead code, no TensorIndexExpr for OpCall";
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const Load<Tensor>& op_call) const {
    const auto& [tensor] = op_call.tuple();
    return Translate(TensorIteratorExpr4Tensor(tensor));
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::vector<ir::Expr> TranslateTensorIndex(const OpExpr& op_expr) const {
    return std::visit(
        [&](const auto& impl) { return TranslateTensorIndexImpl(impl); },
        op_expr.variant());
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const hlir::framework::Node* op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;

    CHECK_EQ(store_rvalue->operands.size(), op_expr_children->size());
    for (int i = 0; i < op_expr_children->size(); ++i) {
      const auto& opt_operant =
          Translate(op_expr_children->at(i), std::nullopt);
      if (opt_operant.has_value()) {
        store_rvalue->operands.at(i) = opt_operant.value();
      } else {
        store_rvalue->operands.at(i).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(i));
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const tReduceInit<const hlir::framework::Node*>& op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    VLOG(1) << "tReduceInit store_rvalue:\n" << store_rvalue;

    CHECK_EQ(store_rvalue->operands.size(), 0);
    CHECK_EQ(op_expr_children->size(), 0);
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const tReduceAcc<const hlir::framework::Node*>& op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    VLOG(1) << "tReduceAcc store_rvalue:\n" << store_rvalue;

    CHECK_EQ(store_rvalue->operands.size(), 2);
    CHECK_EQ(op_expr_children->size(), 1);
    CHECK(opt_output_tensor.has_value());
    store_rvalue->operands.at(0).As<ir::Load>()->indices =
        Translate(TensorIteratorExpr4Tensor(opt_output_tensor.value()));
    {
      const auto& opt_operant =
          Translate(op_expr_children->at(0), std::nullopt);
      if (opt_operant.has_value()) {
        store_rvalue->operands.at(1) = opt_operant.value();
      } else {
        store_rvalue->operands.at(1).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(0));
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateImpl(
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor) const {
    const auto& [op, op_expr_children] = op_expr.tuple();
    return std::visit(
        [&](const auto& impl) {
          return TranslateOpCallImpl(impl, op_expr, opt_output_tensor);
        },
        op.variant());
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::optional<ir::Expr> Translate(
      const OpExpr& op_expr,
      const std::optional<Tensor>& opt_output_tensor) const {
    return std::visit(
        [&](const auto& impl) {
          return TranslateImpl(impl, opt_output_tensor);
        },
        op_expr.variant());
  }

  ir::Expr TranslateImpl(const OpExprStmt& op_expr_stmt) const {
    return Translate(op_expr_stmt);
  }

  // using OpExprStmt = Store<Tensor, OpExpr>;
  ir::Expr Translate(const OpExprStmt& op_expr_stmt) const {
    const auto& [output_tensor, op_expr] = op_expr_stmt.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    std::optional<Tensor> opt_output_tensor = output_tensor;
    const auto& opt_rvalue = Translate(op_expr, opt_output_tensor);
    CHECK(opt_rvalue.has_value());

    const auto& output_expr =
        ir::Store::Make(store_expr.value().As<ir::Store>()->tensor,
                        opt_rvalue.value(),
                        Translate(TensorIteratorExpr4Tensor(output_tensor)));

    std::vector<ir::Expr> fake_values = {ir::Var("fake_v_0"),
                                         ir::Var("fake_v_1")};
    std::vector<ir::Var> fake_vars = {ir::Var("fake_i_0"), ir::Var("fake_i_1")};
    ir::Expr ret = ir::ScheduleBlock::Make(
        fake_vars,
        {},
        {},
        output_expr.As<ir::Store>()->tensor.as_tensor()->name,
        output_expr);
    ret = ir::ScheduleBlockRealize::Make(fake_values, ret);
    return ret;
  }

  ir::Expr Translate(const List<InlineStmt>& stmts) const {
    std::vector<ir::Expr> exprs;
    for (const auto& stmt : *stmts) {
      exprs.emplace_back(Translate(stmt));
    }
    return ir::Block::Make(exprs);
  }

  ir::Expr TranslateImpl(const MapStmt<InlineStmt>& map_stmt) const {
    const auto& [iterators, stmts] = map_stmt.tuple();
    ir::Expr ret = Translate(stmts);
    CHECK_GT(iterators->size(), 0);
    for (int i = iterators->size() - 1; i >= 0; --i) {
      const auto& iterator = iterators->at(i);
      const auto& ld = LoopDescriptor4LoopIterator(iterator);
      ir::Var var{"v_" + std::to_string(iterator.value().unique_id())};
      ir::Expr min{std::int32_t(0)};
      ir::Expr extent = GetLoopSize(ld);
      ir::ForType for_type = GetForType(ld);
      ir::DeviceAPI device_api = GetDeviceApi();
      ret = ir::For::Make(var, min, extent, for_type, device_api, ret);
    }
    return ret;
  }

  ir::Expr Translate(const InlineStmt& inline_stmt) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      inline_stmt.variant());
  }

  ir::Expr Translate(const MapStmt<Stmt>& map_stmt) const {
    if (FLAGS_cinn_map_expr_enable_inline) {
      Stmt stmt = map_stmt;
      InternalStmt internal_stmt = ConvertToInternalStmt(stmt);
      InlineStmt inline_stmt = ConvertToInlineStmt(internal_stmt);
      return Translate(inline_stmt);
    } else {
      const auto& [iterators, stmts] = map_stmt.tuple();
      ir::Expr ret = Translate(stmts);
      CHECK_GT(iterators->size(), 0);
      for (int i = iterators->size() - 1; i >= 0; --i) {
        const auto& iterator = iterators->at(i);
        const auto& ld = LoopDescriptor4LoopIterator(iterator);
        ir::Var var{"v_" + std::to_string(iterator.value().unique_id())};
        ir::Expr min{std::int32_t(0)};
        ir::Expr extent = GetLoopSize(ld);
        ir::ForType for_type = GetForType(ld);
        ir::DeviceAPI device_api = GetDeviceApi();
        ret = ir::For::Make(var, min, extent, for_type, device_api, ret);
      }
      ret = ir::ScheduleBlock::Make({}, {}, {}, "root", ret);
      ret = ir::ScheduleBlockRealize::Make({}, ret);
      return ret;
    }
  }

  ir::DeviceAPI GetDeviceApi() const { return ir::DeviceAPI::Host; }

  ir::Expr GetLoopSize(const LoopDescriptor& ld) const {
    const auto& [_, loop_size] = ld.tuple();
    CHECK(loop_size.Has<std::int64_t>());
    return ir::Expr{std::int32_t(loop_size.Get<std::int64_t>())};
  }

  ir::ForType GetForTypeImpl(const S0x& loop_type) const {
    return ir::ForType::GPUBlock;
  }

  ir::ForType GetForTypeImpl(const S0y& loop_type) const {
    return ir::ForType::GPUBlock;
  }

  ir::ForType GetForTypeImpl(const S0z& loop_type) const {
    return ir::ForType::GPUBlock;
  }

  ir::ForType GetForTypeImpl(const S1x& loop_type) const {
    return ir::ForType::GPUThread;
  }

  ir::ForType GetForTypeImpl(const S1y& loop_type) const {
    return ir::ForType::GPUThread;
  }

  ir::ForType GetForTypeImpl(const S1z& loop_type) const {
    return ir::ForType::GPUThread;
  }

  ir::ForType GetForTypeImpl(const Temporal& loop_type) const {
    return ir::ForType::Serial;
  }

  ir::ForType GetForTypeImpl(const Vectorize& loop_type) const {
    return ir::ForType::Vectorized;
  }

  ir::ForType GetForTypeImpl(const Unroll& loop_type) const {
    return ir::ForType::Unrolled;
  }

  ir::ForType GetForType(const LoopDescriptor& ld) const {
    const auto& [loop_type, _] = ld.tuple();
    return ir::ForType::Serial;
    // TODO(Hongyu Jia): Support different ForType here
    // return std::visit([&](const auto& impl) { return GetForTypeImpl(impl); },
    //            loop_type.variant());
  }

  ir::Expr TranslateImpl(const MapStmt<Stmt>& map_stmt) const {
    return Translate(map_stmt);
  }

  ir::Expr TranslateImpl(const Undefined& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const Ok& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const Iterator& iterator) const {
    return ir::Var("v_" + std::to_string(iterator.value().unique_id()));
  }
  ir::Expr TranslateImpl(const Constant& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const List<Value>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const IndexDotValue<Value, Constant>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const IndexUnDotValue<Value, Constant>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const ConstantAdd<Value>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const ConstantDiv<Value>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const ConstantMod<Value>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const ListGetItem<Value, Constant>& value) const {
    LOG(FATAL) << "Not Supported";
  }
  ir::Expr TranslateImpl(const PtrGetItem<Value>& value) const {
    LOG(FATAL) << "Not Supported";
  }

  ir::Expr Translate(const TensorIteratorExpr& iterator_expr) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      iterator_expr.variant());
  }

  std::vector<ir::Expr> Translate(
      const List<TensorIteratorExpr>& iterator_exprs) const {
    std::vector<ir::Expr> ret{};
    for (const auto& iterator_expr : *iterator_exprs) {
      ret.emplace_back(Translate(iterator_expr));
    }
    return ret;
  }

  ir::Expr GetStoreExprForOpStmt(const OpStmt& op_stmt) const {
    const auto& [op, _0, _1] = op_stmt.tuple();
    return std::visit([&](const auto& impl) { return GetStoreExprForOp(impl); },
                      op.variant());
  }

  ir::Expr MakeStoreExprForOpStmt(const OpStmt& op_stmt) const {
    const auto& store_expr = GetStoreExprForOpStmt(op_stmt);

    const auto& [_0, inputs, outputs] = op_stmt.tuple();
    CHECK_EQ(outputs.value()->size(), 1);

    ir::Expr store_rvalue = store_expr.As<ir::Store>()->value;
    CHECK_EQ(store_rvalue->operands.size(), inputs.value()->size());
    for (int i = 0; i < inputs.value()->size(); ++i) {
      const auto& index_exprs =
          Translate(TensorIteratorExpr4Tensor(inputs.value()->at(i)));
      store_rvalue->operands.at(i).As<ir::Load>()->indices = index_exprs;
    }
    return ir::Store::Make(
        store_expr.As<ir::Store>()->tensor,
        store_rvalue,
        Translate(TensorIteratorExpr4Tensor(outputs.value()->at(0))));
  }

  ir::Expr Translate(const OpStmt& op_stmt) const {
    const auto& output_expr = MakeStoreExprForOpStmt(op_stmt);

    std::vector<ir::Expr> fake_values = {ir::Var("fake_v_0"),
                                         ir::Var("fake_v_1")};
    std::vector<ir::Var> fake_vars = {ir::Var("fake_i_0"), ir::Var("fake_i_1")};
    ir::Expr ret = ir::ScheduleBlock::Make(
        fake_vars,
        {},
        {},
        output_expr.As<ir::Store>()->tensor.as_tensor()->name,
        output_expr);
    ret = ir::ScheduleBlockRealize::Make(fake_values, ret);
    return ret;
  }

  ir::Expr TranslateImpl(const OpStmt& op_stmt) const {
    return Translate(op_stmt);
  }

  ir::Expr Translate(const List<Stmt>& stmts) const {
    std::vector<ir::Expr> exprs;
    for (const auto& stmt : *stmts) {
      exprs.emplace_back(Translate(stmt));
    }
    return ir::Block::Make(exprs);
  }

  ir::Expr Translate(const Stmt& stmt) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      stmt.variant());
  }

  MapExpr map_expr_;
  const Node2LoweredFuncs* node2lowered_funcs_;
  TensorIteratorExpr4TensorT TensorIteratorExpr4Tensor;
  LoopDescriptor4LoopIteratorT LoopDescriptor4LoopIterator;
};

}  // namespace

ir::Expr MapExprToIr(const MapExprCtx& map_expr_ctx) {
  const auto& expr = MapExprToIrTranslator(map_expr_ctx.map_expr(),
                                           map_expr_ctx.node2lowered_funcs())
                         .Translate();
  VLOG(1) << "Finish MapExprToIr\n" << expr;
  return expr;
}

}  // namespace cinn::adt
