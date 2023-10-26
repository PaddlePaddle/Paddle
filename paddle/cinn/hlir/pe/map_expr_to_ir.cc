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

#include "paddle/cinn/adt/equation_value_match_trait.h"
#include "paddle/cinn/adt/inline_translator.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn::adt {

namespace {

using IteratorInt = std::int32_t;
using Node2LoweredFuncs =
    std::unordered_map<hlir::framework::Node*, std::vector<ir::LoweredFunc>>;

using TensorIteratorExpr4TensorT =
    std::function<adt::List<adt::TensorIteratorExpr>(const adt::Tensor&)>;
using IterExprs4TensorT =
    std::function<std::vector<ir::Expr>(const adt::Tensor&)>;
using LoopDescriptor4LoopIteratorT =
    std::function<adt::LoopDescriptor(const adt::Iterator&)>;

class MapExprToIrTranslator {
 public:
  explicit MapExprToIrTranslator(const MapExpr& map_expr,
                                 const Node2LoweredFuncs& node2lowered_funcs,
                                 const common::Target& target)
      : map_expr_(map_expr),
        node2lowered_funcs_(&node2lowered_funcs),
        target_(target) {
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

  std::optional<ir::Expr> TranslateOpExprImpl(
      const Load<Tensor>& load,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::nullopt;
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const OpCall<OpExpr>& op_call,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    LOG(FATAL) << "Dead code, no TensorIndexExpr for OpCall";
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const Load<Tensor>& op_call,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [tensor] = op_call.tuple();
    return IterExprs4Tensor(tensor);
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::vector<ir::Expr> TranslateTensorIndex(
      const OpExpr& op_expr, const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::visit(
        [&](const auto& impl) {
          return TranslateTensorIndexImpl(impl, IterExprs4Tensor);
        },
        op_expr.variant());
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const hlir::framework::Node* op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;

    if (store_rvalue.As<ir::Load>()) {
      CHECK_EQ(store_rvalue->operands.size(), 0);
      CHECK_EQ(op_expr_children->size(), 1);
      store_rvalue.As<ir::Load>()->indices =
          TranslateTensorIndex(op_expr_children->at(0), IterExprs4Tensor);
    } else {
      if (!op_expr_children->empty()) {
        CHECK_EQ(store_rvalue->operands.size(), op_expr_children->size());
        for (int i = 0; i < op_expr_children->size(); ++i) {
          const auto& opt_operant = TranslateOpExpr(
              op_expr_children->at(i), std::nullopt, IterExprs4Tensor);
          if (opt_operant.has_value()) {
            store_rvalue->operands.at(i) = opt_operant.value();
          } else {
            store_rvalue->operands.at(i).As<ir::Load>()->indices =
                TranslateTensorIndex(op_expr_children->at(i), IterExprs4Tensor);
          }
        }
      } else {
        // Do nothing
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const tReduceInit<const hlir::framework::Node*>& op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
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
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    VLOG(1) << "tReduceAcc store_rvalue:\n" << store_rvalue;

    CHECK_EQ(store_rvalue->operands.size(), 2);
    CHECK_EQ(op_expr_children->size(), 1);
    CHECK(opt_output_tensor.has_value());
    store_rvalue->operands.at(0).As<ir::Load>()->indices =
        IterExprs4Tensor(opt_output_tensor.value());
    {
      const auto& opt_operant = TranslateOpExpr(
          op_expr_children->at(0), std::nullopt, IterExprs4Tensor);
      if (opt_operant.has_value()) {
        store_rvalue->operands.at(1) = opt_operant.value();
      } else {
        store_rvalue->operands.at(1).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(0), IterExprs4Tensor);
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpExprImpl(
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [op, op_expr_children] = op_expr.tuple();
    return std::visit(
        [&](const auto& impl) {
          return TranslateOpCallImpl(
              impl, op_expr, opt_output_tensor, IterExprs4Tensor);
        },
        op.variant());
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::optional<ir::Expr> TranslateOpExpr(
      const OpExpr& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::visit(
        [&](const auto& impl) {
          return TranslateOpExprImpl(impl, opt_output_tensor, IterExprs4Tensor);
        },
        op_expr.variant());
  }

  ir::Expr TranslateImpl(const OpExprStmt& op_expr_stmt) const {
    return Translate(op_expr_stmt);
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const Tensor& tensor,
                              const DoEachT& DoEach) const {
    const List<Value>& iterator_values = TensorIteratorExpr4Tensor(tensor);
    for (const auto& iterator_value : *iterator_values) {
      DoEach(iterator_value);
    }
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValueImpl(const OpCall<OpExpr>& op_call,
                                  const DoEachT& DoEach) const {
    const auto& [_, children] = op_call.tuple();
    for (const auto& child : *children) {
      VisitEachIteratorValue(child, DoEach);
    }
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValueImpl(const Load<Tensor>& load,
                                  const DoEachT& DoEach) const {
    const auto& [tensor] = load.tuple();
    VisitEachIteratorValue(tensor, DoEach);
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const OpExpr& op_expr,
                              const DoEachT& DoEach) const {
    return std::visit(
        [&](const auto& impl) {
          return VisitEachIteratorValueImpl(impl, DoEach);
        },
        op_expr.variant());
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const OpExprStmt& op_expr_stmt,
                              const DoEachT& DoEach) const {
    const auto& [tensor, op_expr] = op_expr_stmt.tuple();
    VisitEachIteratorValue(tensor, DoEach);
    VisitEachIteratorValue(op_expr, DoEach);
  }

  IterExprs4TensorT MakeGetterIterExprs4Tensor(
      const OpExprStmt& op_expr_stmt,
      std::vector<std::pair<ir::Var, ir::Expr>>* binding_var2value) const {
    std::unordered_map<Value, std::pair<ir::Var, ir::Expr>> value2var_expr{};
    VisitEachIteratorValue(op_expr_stmt, [&](const Value& value) {
      if (value2var_expr.count(value) == 0) {
        ir::Var var{std::string("m_expr_i_") +
                    std::to_string(UniqueId::New().unique_id())};
        ir::Expr expr = TranslateTensorIterator(value);
        CHECK(value2var_expr.emplace(value, std::make_pair(var, expr)).second);
      } else {
        // Do nothing
      }
    });
    for (const auto& [_, pair] : value2var_expr) {
      binding_var2value->push_back(pair);
    }
    return [value2var_expr, this](const Tensor& tensor) {
      const List<Value>& iterator_values = TensorIteratorExpr4Tensor(tensor);
      std::vector<ir::Expr> ret{};
      ret.reserve(iterator_values->size());
      for (const auto& iterator_value : *iterator_values) {
        const auto& it = value2var_expr.find(iterator_value);
        CHECK(it != value2var_expr.end());
        ret.emplace_back(it->second.first);
      }
      return ret;
    };
  }

  std::vector<ir::Var> GetVectorOfPairFirst(
      const std::vector<std::pair<ir::Var, ir::Expr>>& pairs) const {
    std::vector<ir::Var> ret{};
    ret.reserve(pairs.size());
    for (const auto& pair : pairs) {
      ret.emplace_back(pair.first);
    }
    return ret;
  }

  std::vector<ir::Expr> GetVectorOfPairSecond(
      const std::vector<std::pair<ir::Var, ir::Expr>>& pairs) const {
    std::vector<ir::Expr> ret{};
    ret.reserve(pairs.size());
    for (const auto& pair : pairs) {
      ret.emplace_back(pair.second);
    }
    return ret;
  }

  // using OpExprStmt = Store<Tensor, OpExpr>;
  ir::Expr Translate(const OpExprStmt& op_expr_stmt) const {
    const auto& [output_tensor, op_expr] = op_expr_stmt.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    CHECK(store_expr.has_value());
    std::optional<Tensor> opt_output_tensor = output_tensor;

    std::vector<std::pair<ir::Var, ir::Expr>> binding_var2value{};
    const auto& IterExprs4Tensor =
        MakeGetterIterExprs4Tensor(op_expr_stmt, &binding_var2value);

    const auto& opt_rvalue =
        TranslateOpExpr(op_expr, opt_output_tensor, IterExprs4Tensor);
    CHECK(opt_rvalue.has_value());

    const auto& output_expr =
        ir::Store::Make(store_expr.value().As<ir::Store>()->tensor,
                        opt_rvalue.value(),
                        IterExprs4Tensor(output_tensor));

    ir::Expr ret = ir::ScheduleBlock::Make(
        GetVectorOfPairFirst(binding_var2value),
        {},
        {},
        output_expr.As<ir::Store>()->tensor.as_tensor()->name,
        output_expr);
    ret = ir::ScheduleBlockRealize::Make(
        GetVectorOfPairSecond(binding_var2value), ret);
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
      ir::Expr min{IteratorInt(0)};
      ir::Expr extent = GetLoopSize(ld);
      const auto& [for_type, vectorize_info, bind_info] = GetForTypeAndInfo(ld);
      ir::DeviceAPI device_api = GetDeviceApi();
      ret = ir::For::Make(var,
                          min,
                          extent,
                          for_type,
                          device_api,
                          ret,
                          vectorize_info,
                          bind_info);
    }
    return ret;
  }

  ir::Expr Translate(const InlineStmt& inline_stmt) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      inline_stmt.variant());
  }

  ir::Expr Translate(const MapStmt<Stmt>& map_stmt) const {
    Stmt stmt = map_stmt;
    InternalStmt internal_stmt = ConvertToInternalStmt(stmt);
    InlineStmt inline_stmt = ConvertToInlineStmt(internal_stmt);
    return Translate(inline_stmt);
  }

  ir::DeviceAPI GetDeviceApi() const { return ir::DeviceAPI::Host; }

  ir::Expr GetLoopSize(const LoopDescriptor& ld) const {
    const auto& [_, loop_size] = ld.tuple();
    CHECK(loop_size.Has<std::int64_t>());
    return ir::Expr{IteratorInt(loop_size.Get<std::int64_t>())};
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0x& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 0, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0y& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 1, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0z& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 2, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1x& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 0, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1y& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 1, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1z& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 2, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Temporal& loop_type,
                        const LoopDescriptor& ld) const {
    return std::make_tuple(
        ir::ForType::Serial, ir::VectorizeInfo(), ir::BindInfo());
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Vectorize& loop_type,
                        const LoopDescriptor& ld) const {
    LOG(FATAL) << "Vectorize not supported yet";
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Unroll& loop_type,
                        const LoopDescriptor& ld) const {
    LOG(FATAL) << "Unroll not supported yet";
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo> GetForTypeAndInfo(
      const LoopDescriptor& ld) const {
    const auto& [loop_type, _] = ld.tuple();
    return std::visit(
        [&](const auto& impl) { return GetForTypeAndInfoImpl(impl, ld); },
        loop_type.variant());
  }

  ir::Expr Mul(const ir::Expr& a, std::int64_t b) const {
    if (b == 1) {
      return a;
    } else {
      ir::Expr b_expr{IteratorInt(b)};
      return ir::Mul::Make(a, b_expr);
    }
  }

  ir::Expr Accumulate(const std::vector<ir::Expr>& strided_exprs) const {
    if (strided_exprs.size() == 0) {
      LOG(FATAL) << "Dead code";
    } else if (strided_exprs.size() == 1) {
      return strided_exprs.at(0);
    } else {
      ir::Expr ret = strided_exprs.at(0);
      for (int i = 1; i < strided_exprs.size(); ++i) {
        ret = ir::Add::Make(ret, strided_exprs.at(i));
      }
      return ret;
    }
    LOG(FATAL) << "Dead code";
  }

  std::int64_t GetStride(const List<Constant>& dims, int start) const {
    CHECK_GE(start, -1);
    std::int64_t ret = 1;
    for (int idx = start + 1; idx < dims->size(); ++idx) {
      CHECK(dims->at(idx).Has<std::int64_t>());
      ret *= dims->at(idx).Get<std::int64_t>();
    }
    return ret;
  }

  using IndexDotValueOfList = IndexDotValue<List<Value>, List<std::int64_t>>;
  ir::Expr TranslateIndexDotValueOfList(const Value& value) const {
    const auto& [list_value, dot_dims_value] =
        value.Get<IndexDotValue<Value, Constant>>().tuple();
    const auto& values = list_value.Get<List<Value>>();
    const auto& dim_values = dot_dims_value.Get<List<Constant>>();
    CHECK_EQ(values->size(), dim_values->size());

    std::vector<ir::Expr> strided_exprs{};
    for (std::size_t i = 0; i < values->size(); ++i) {
      const auto& value_expr = TranslateTensorIterator(values->at(i));
      const auto& stride_value = GetStride(dim_values, i);
      strided_exprs.emplace_back(Mul(value_expr, stride_value));
    }
    return Accumulate(strided_exprs);
  }

  using ListGetItemOfUnDot =
      ListGetItem<IndexUnDotValue<Value, List<std::int64_t>>, std::int64_t>;
  ir::Expr TranslateListGetItemOfUnDot(const Value& value) const {
    const auto& [undot_value, idx_value] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& [tensor_index_value, dims_value] =
        undot_value.Get<IndexUnDotValue<Value, Constant>>().tuple();
    ir::Expr tensor_index_expr = TranslateTensorIterator(tensor_index_value);
    std::int64_t idx = idx_value.Get<std::int64_t>();
    const auto& dims = dims_value.Get<List<Constant>>();

    ir::Expr mod_operand{IteratorInt(GetStride(dims, idx - 1))};
    ir::Expr div_operant{IteratorInt(GetStride(dims, idx))};
    return ir::Div::Make(ir::Mod::Make(tensor_index_expr, mod_operand),
                         div_operant);
  }

  ir::Expr TranslateIterator(const Value& value) const {
    const auto& iterator = value.Get<Iterator>();
    return ir::Var("v_" + std::to_string(iterator.value().unique_id()));
  }

  ir::Expr TranslateTensorIterator(const Value& value) const {
    if (Match<IndexDotValueOfList>(value)) {
      return TranslateIndexDotValueOfList(value);
    } else if (Match<ListGetItemOfUnDot>(value)) {
      return TranslateListGetItemOfUnDot(value);
    } else if (Match<Iterator>(value)) {
      return TranslateIterator(value);
    } else {
      LOG(FATAL) << "Not supported yet! " << ToTxtString(value);
    }
  }

  std::vector<ir::Expr> Translate(
      const List<TensorIteratorExpr>& iterator_exprs) const {
    std::vector<ir::Expr> ret{};
    for (const auto& iterator_expr : *iterator_exprs) {
      ret.emplace_back(TranslateTensorIterator(iterator_expr));
    }
    return ret;
  }

  MapExpr map_expr_;
  const Node2LoweredFuncs* node2lowered_funcs_;
  const common::Target target_;
  TensorIteratorExpr4TensorT TensorIteratorExpr4Tensor;
  LoopDescriptor4LoopIteratorT LoopDescriptor4LoopIterator;
};

}  // namespace

ir::Expr MapExprToIr(const MapExprCtx& map_expr_ctx,
                     const common::Target& target) {
  const auto& expr =
      MapExprToIrTranslator(
          map_expr_ctx.map_expr(), map_expr_ctx.node2lowered_funcs(), target)
          .Translate();
  VLOG(1) << "Finish MapExprToIr\n" << expr;
  return expr;
}

}  // namespace cinn::adt
