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
using Node2IrExpr = std::unordered_map<hlir::framework::Node*, ir::Expr>;
using Tensor2Store = std::unordered_map<Tensor, ir::Expr>;
using Tensor2Load = std::unordered_map<Tensor, ir::Expr>;

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
    InitTensor2LoadAndStore();
  }

  ir::Expr Translate() const {
    VLOG(1) << "Translate MapExpr: ";
    VLOG(1) << ToTxtString(map_expr_, "");
    return ir::Block::Make({Translate(map_expr_)});
  }

 private:
  void InitTensor2LoadAndStore() { InitTensor2LoadAndStore(map_expr_); }

  void InitTensor2LoadAndStore(const MapExpr& map_expr) {
    const auto& [anchored_map_stmts, _0, _1] = map_expr.tuple();
    CHECK_EQ(anchored_map_stmts->size(), 1);
    InitTensor2LoadAndStore(anchored_map_stmts->at(0));
  }

  void InitTensor2LoadAndStore(const AnchoredMapStmt& anchored_map_stmts) {
    MapStmt<Stmt> map_stmts = std::get<0>(anchored_map_stmts.tuple());
    InitTensor2LoadAndStore(map_stmts);
  }

  void InitTensor2LoadAndStore(const MapStmt<Stmt>& map_stmt) {
    const auto& [_, stmts] = map_stmt.tuple();
    InitTensor2LoadAndStore(stmts);
  }

  void InitTensor2LoadAndStore(const List<Stmt>& stmts) {
    for (const auto& stmt : *stmts) {
      InitTensor2LoadAndStore(stmt);
    }
  }

  void InitTensor2LoadAndStore(const Stmt& stmt) {
    std::visit([&](const auto& impl) { InitTensor2LoadAndStoreImpl(impl); },
               stmt.variant());
  }

  void InitTensor2LoadAndStoreImpl(const MapStmt<Stmt>& map_stmt) {
    return InitTensor2LoadAndStore(map_stmt);
  }

  void InitTensor2LoadAndStoreImpl(const OpStmt& op_stmt) {
    const auto& [op, inputs, outputs] = op_stmt.tuple();
    CHECK_EQ(outputs.value()->size(), 1);
    CHECK(op.Has<const hlir::framework::Node*>());
    const hlir::framework::Node* op_node =
        op.Get<const hlir::framework::Node*>();
    CHECK(node2lowered_funcs_ != nullptr);
    const auto& iter =
        node2lowered_funcs_->find(const_cast<hlir::framework::Node*>(op_node));
    CHECK(iter != node2lowered_funcs_->end());
    const auto& lowered_func = iter->second;

    VLOG(1) << "Origin Lowered_Func :\n" << lowered_func.at(0);
    CHECK_EQ(lowered_func.size(), 1);
    int i = 0;
    const auto& output_values = outputs.value();
    const auto& input_values = inputs.value();
    VisitEachExprBody(lowered_func.at(0), [&](const ir::Expr& expr) {
      if (expr.node_type() == ir::IrNodeTy::Store) {
        if (tensor2store_.find(output_values->at(0)) == tensor2store_.end()) {
          CHECK(tensor2store_.emplace(output_values->at(0), expr).second);
        }
        CHECK(node2ir_expr_
                  .emplace(const_cast<hlir::framework::Node*>(op_node),
                           expr.As<ir::Store>()->value)
                  .second);
      } else if (expr.node_type() == ir::IrNodeTy::Load) {
        if (tensor2load_.find(input_values->at(i)) == tensor2load_.end()) {
          CHECK(tensor2load_.emplace(input_values->at(i), expr).second);
          ++i;
        }
      } else {
        // Do nothing
      }
    });
  }

  template <typename DoEachT>
  void VisitEachExprBody(const ir::Expr& expr, const DoEachT& DoEach) const {
    DoEach(expr);
    switch (expr.node_type()) {
      case ir::IrNodeTy::_LoweredFunc_:
        VisitEachExprBody(expr.as_lowered_func()->body, DoEach);
        break;
      case ir::IrNodeTy::Block:
        for (const auto& stmt : expr.As<ir::Block>()->stmts) {
          VisitEachExprBody(stmt, DoEach);
        }
        break;
      case ir::IrNodeTy::ScheduleBlockRealize:
        VisitEachExprBody(expr.As<ir::ScheduleBlockRealize>()->schedule_block,
                          DoEach);
        break;
      case ir::IrNodeTy::ScheduleBlock:
        VisitEachExprBody(expr.As<ir::ScheduleBlock>()->body, DoEach);
        break;
      case ir::IrNodeTy::For:
        VisitEachExprBody(expr.As<ir::For>()->body, DoEach);
        break;
      case ir::IrNodeTy::Store:
        VisitEachExprBody(expr.As<ir::Store>()->value, DoEach);
        break;
      case ir::IrNodeTy::Add:
        VisitEachExprBody(expr.As<ir::Add>()->a(), DoEach);
        VisitEachExprBody(expr.As<ir::Add>()->b(), DoEach);
        break;
      case ir::IrNodeTy::Call:
        for (const auto& arg : expr.As<ir::Call>()->write_args) {
          VLOG(1) << "Call write_arg = " << arg;
          VisitEachExprBody(arg, DoEach);
        }
        for (const auto& arg : expr.As<ir::Call>()->read_args) {
          VLOG(1) << "Call read_arg = " << arg;
          VisitEachExprBody(arg, DoEach);
        }
        break;
      case ir::IrNodeTy::Load:
        // Do nothing
        break;
      default:
        VLOG(1) << "Visit node_type = " << expr.node_type() << ", do nothing";
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
    VLOG(1) << "Translate AnchoredMapStmt";
    const MapStmt<Stmt>& map_stmt = std::get<0>(anchored_map_stmt.tuple());
    return Translate(map_stmt);
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

  ir::Expr TranslateImpl(const OpCall<OpExpr>& op_call) const {
    const auto& [op, op_exprs] = op_call.tuple();
    CHECK(op.Has<const hlir::framework::Node*>());
    const hlir::framework::Node* op_node =
        op.Get<const hlir::framework::Node*>();
    const auto& ir_expr =
        node2ir_expr_.at(const_cast<hlir::framework::Node*>(op_node));
    CHECK_EQ(ir_expr->operands.size(), op_exprs->size());
    for (int i = 0; i < op_exprs->size(); ++i) {
      ir_expr->operands.at(i) = Translate(op_exprs->at(i));
    }
    return ir_expr;
  }

  ir::Expr TranslateImpl(const Load<Tensor>& load) const {
    const auto& [tensor] = load.tuple();
    return ir::Load::Make(tensor2load_.at(tensor).As<ir::Load>()->tensor,
                          Translate(TensorIteratorExpr4Tensor(tensor)));
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  ir::Expr Translate(const OpExpr& op_expr) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      op_expr.variant());
  }

  ir::Expr TranslateImpl(const OpExprStmt& op_expr_stmt) const {
    return Translate(op_expr_stmt);
  }

  // using OpExprStmt = Store<Tensor, OpExpr>;
  ir::Expr Translate(const OpExprStmt& op_expr_stmt) const {
    const auto& [output, op_expr] = op_expr_stmt.tuple();

    const auto& output_expr =
        ir::Store::Make(tensor2store_.at(output).As<ir::Store>()->tensor,
                        Translate(op_expr),
                        Translate(TensorIteratorExpr4Tensor(output)));

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
    ret = ir::ScheduleBlock::Make({}, {}, {}, "root", ret);
    ret = ir::ScheduleBlockRealize::Make({}, ret);
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

  std::vector<ir::Expr> Translate(const tIn<List<Arg>>& inputs) const {
    std::vector<ir::Expr> ret{};
    for (std::size_t i = 0; i < inputs.value()->size(); ++i) {
      const auto& input = inputs.value()->at(i);
      const auto& load = tensor2load_.at(input);
      ret.emplace_back(
          ir::Load::Make(load.As<ir::Load>()->tensor,
                         Translate(TensorIteratorExpr4Tensor(input))));
    }
    return ret;
  }

  ir::Expr TranslateImpl(const hlir::framework::Node* op_node,
                         const std::vector<ir::Expr>& inputs) const {
    const auto& op_expr =
        node2ir_expr_.at(const_cast<hlir::framework::Node*>(op_node));
    CHECK_EQ(op_expr->operands.size(), inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      op_expr->operands.at(i) = inputs.at(i);
    }
    return op_expr;
  }

  ir::Expr TranslateImpl(
      const tReduceInit<const hlir::framework::Node*>& op_node,
      const std::vector<ir::Expr>& inputs) const {
    ADT_TODO();
  }

  ir::Expr TranslateImpl(
      const tReduceAcc<const hlir::framework::Node*>& op_node,
      const std::vector<ir::Expr>& inputs) const {
    ADT_TODO();
  }

  ir::Expr Translate(const Op& op, const std::vector<ir::Expr>& inputs) const {
    return std::visit(
        [&](const auto& impl) { return TranslateImpl(impl, inputs); },
        op.variant());
  }

  ir::Expr Translate(const OpStmt& op_stmt) const {
    const auto& [op, inputs, outputs] = op_stmt.tuple();
    const auto& input_exprs = Translate(inputs);
    CHECK_EQ(outputs.value()->size(), 1);

    const auto& output = outputs.value()->at(0);
    ir::Expr output_expr =
        ir::Store::Make(tensor2store_.at(output).As<ir::Store>()->tensor,
                        Translate(op, input_exprs),
                        Translate(TensorIteratorExpr4Tensor(output)));

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
  Node2IrExpr node2ir_expr_;
  Tensor2Store tensor2store_;
  Tensor2Load tensor2load_;
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
