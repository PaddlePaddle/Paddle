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

#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/print_map_expr.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn::adt {

namespace {

using Node2LoweredFuncs =
    std::unordered_map<hlir::framework::Node*, std::vector<ir::LoweredFunc>>;
using Node2Store =
    std::unordered_map<hlir::framework::Node*, std::vector<ir::Expr>>;
using Node2Loads =
    std::unordered_map<hlir::framework::Node*, std::vector<ir::Expr>>;

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
    InitNode2LoadAndStore();
  }

  ir::Expr Translate() const {
    VLOG(1) << "Translate MapExpr: ";
    PrintMapExpr(map_expr_, "");
    return ir::Block::Make({Translate(map_expr_)});
  }

 private:
  void InitNode2LoadAndStore() {
    CHECK(node2lowered_funcs_ != nullptr);
    for (const auto& [node, lowered_func] : *node2lowered_funcs_) {
      VLOG(1) << "Origin Lowered_Func :\n" << lowered_func.at(0);
      CHECK_EQ(lowered_func.size(), 1);
      std::vector<ir::Expr> stores{};
      std::vector<ir::Expr> loads{};
      VisitEachExprBody(lowered_func.at(0), [&](const ir::Expr& expr) {
        if (expr.node_type() == ir::IrNodeTy::Store) {
          stores.emplace_back(expr);
        } else if (expr.node_type() == ir::IrNodeTy::Load) {
          loads.emplace_back(expr);
        } else {
          // Do nothing
        }
      });
      CHECK(node2store_.emplace(node, stores).second);
      CHECK(node2loads_.emplace(node, loads).second);
      VLOG(1) << "load.size() = " << loads.size();
      VLOG(1) << "stores.size() = " << stores.size();
    }
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
    const MapStmt<Stmt>& map_stmt = std::get<0>(anchored_map_stmt.tuple());
    return Translate(map_stmt);
  }

  ir::Expr Translate(const MapStmt<Stmt>& map_stmt) const {
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

  std::vector<ir::Expr> Translate(const tIn<List<Arg>>& inputs,
                                  const std::vector<ir::Expr>& loads) const {
    CHECK_EQ(inputs.value()->size(), loads.size());
    std::vector<ir::Expr> ret{};
    for (std::size_t i = 0; i < loads.size(); ++i) {
      const auto& input = inputs.value()->at(i);
      const auto& load = loads.at(i);
      ret.emplace_back(
          ir::Load::Make(load.As<ir::Load>()->tensor,
                         Translate(TensorIteratorExpr4Tensor(input))));
    }
    return ret;
  }

  std::vector<ir::Expr> Translate(const tOut<List<Arg>>& outputs,
                                  const std::vector<ir::Expr>& stores) const {
    CHECK_EQ(outputs.value()->size(), stores.size());
    std::vector<ir::Expr> ret{};
    for (std::size_t i = 0; i < stores.size(); ++i) {
      const auto& output = outputs.value()->at(i);
      const auto& store = stores.at(i);
      ret.emplace_back(
          ir::Store::Make(store.As<ir::Store>()->tensor,
                          store.As<ir::Store>()->value,
                          Translate(TensorIteratorExpr4Tensor(output))));
    }
    return ret;
  }

  ir::Expr TranslateImpl(const hlir::framework::Node* op_node,
                         const std::vector<ir::Expr>& inputs) const {
    if (op_node->op()->name == "elementwise_add") {
      return ir::Add::Make(inputs.at(0), inputs.at(1));
    } else {
      LOG(FATAL) << "Not supported yet";
    }
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
    CHECK(op.Has<const hlir::framework::Node*>());
    const hlir::framework::Node* op_node =
        op.Get<const hlir::framework::Node*>();

    const auto& input_exprs = Translate(
        inputs, node2loads_.at(const_cast<hlir::framework::Node*>(op_node)));
    const auto& output_exprs = Translate(
        outputs, node2store_.at(const_cast<hlir::framework::Node*>(op_node)));
    CHECK_EQ(output_exprs.size(), 1);
    ir::Expr output_expr = output_exprs.at(0);
    output_expr.As<ir::Store>()->value = Translate(op, input_exprs);
    // const auto& iter_values = output_expr.As<ir::Store>()->indices;
    // std::vector<ir::Var> iter_vars = {ir::Var("i_35"), ir::Var("i_36")};
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
  Node2Loads node2loads_;
  Node2Store node2store_;
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
