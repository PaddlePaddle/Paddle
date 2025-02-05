// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/insert_debug_log_callee.h"

#include <sstream>
#include <string>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
using cinn::utils::StringFormat;

namespace {

struct StoreDebugInfoBuilder : public ir::IRVisitor {
  std::tuple<std::string, std::vector<Expr>> operator()(const Expr *e) {
    IRVisitor::Visit(e);
    return std::make_tuple(format_.str(), args_);
  }

 private:
#define _BINARY_OP(Op__, repr__)           \
  void Visit(const ir::Op__ *x) override { \
    format_ << "(";                        \
    IRVisitor::Visit(&x->a());             \
    format_ << " " << #repr__ << " ";      \
    IRVisitor::Visit(&x->b());             \
    format_ << ")";                        \
  }
  _BINARY_OP(Add, +);
  _BINARY_OP(Mul, *);
  _BINARY_OP(Div, /);
  _BINARY_OP(Sub, -);
  _BINARY_OP(Mod, %);
  _BINARY_OP(LT, <);
  _BINARY_OP(LE, <=);
  _BINARY_OP(GT, >);
  _BINARY_OP(GE, >=);
#undef _BINARY_OP

  void Visit(const ir::Load *x) override {
    format_ << type_specifier(x->type());
    args_.push_back(Expr(&Reference(x)));
  }

 public:
  void Visit(const Expr *x) override { IRVisitor::Visit(x); }
  void Visit(const ir::IntImm *x) override {
    format_ << type_specifier(x->type());
    args_.push_back(&Reference(x));
  }
  void Visit(const ir::UIntImm *x) override {
    format_ << type_specifier(x->type());
    args_.push_back(&Reference(x));
  }
  void Visit(const ir::FloatImm *x) override {
    format_ << type_specifier(x->type());
    args_.push_back(&Reference(x));
  }
  void Visit(const ir::StringImm *x) override {}
  void Visit(const ir::EQ *x) override {}
  void Visit(const ir::_Var_ *x) override {}
  void Visit(const ir::NE *x) override {}
  void Visit(const ir::And *x) override {}
  void Visit(const ir::Or *x) override {}
  void Visit(const ir::Min *x) override {}
  void Visit(const ir::Max *x) override {}
  void Visit(const ir::Minus *x) override {}
  void Visit(const ir::Not *x) override {}
  void Visit(const ir::Cast *x) override {}
  void Visit(const ir::For *x) override {}
  void Visit(const ir::PolyFor *x) override {}
  void Visit(const ir::Select *x) override {}
  void Visit(const ir::IfThenElse *x) override {}
  void Visit(const ir::Block *x) override {}
  void Visit(const ir::Call *x) override {}
  void Visit(const ir::Store *x) override {
    format_ << x->tensor.as_tensor()->name << "[] = ";
    Visit(&x->value);
    LOG(INFO) << "store value " << x->value;
  }
  void Visit(const ir::Alloc *x) override {}
  void Visit(const ir::Free *x) override {}
  void Visit(const ir::_Buffer_ *x) override {}
  void Visit(const ir::_Tensor_ *x) override {}
  void Visit(const ir::Let *x) override {}
  void Visit(const ir::Reduce *x) override {}
  void Visit(const ir::Ramp *x) override {}
  void Visit(const ir::Broadcast *x) override {}
  void Visit(const ir::FracOp *x) override {}
  void Visit(const ir::Product *x) override {}
  void Visit(const ir::Sum *x) override {}

 private:
  std::string type_specifier(const Type &type) {
    if (type.is_float()) return "%f";
    if (type.is_int()) return "%d";
    CINN_NOT_IMPLEMENTED
    return "";
  }

 private:
  std::stringstream format_;
  std::vector<Expr> args_;
  bool in_load_{false};
};

struct InsertDebugLogCalleeMutator : public ir::IRMutator<> {
  void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

  void Visit(const ir::_LoweredFunc_ *op, Expr *expr) {
    auto *node = expr->As<ir::_LoweredFunc_>();
    auto *body_block = node->body.As<ir::Block>();
    PADDLE_ENFORCE_NOT_NULL(
        body_block,
        ::common::errors::InvalidArgument(
            "Expected 'body_block' to be non-null, but got null."));

    auto msg = StringFormat("running : %s", GetDebugString(*expr).c_str());
    auto debug_node = CreateDebugStatement(msg);

    ir::IRMutator<>::Visit(&node->body, &node->body);

    auto deal_with_exprs =
        [&](std::vector<Expr> *exprs) {  // deal with op->argument_prepare_exprs
          std::vector<Expr> new_stmts;
          for (auto &expr : *exprs) {
            auto msg =
                StringFormat("running : %s", GetDebugString(expr).c_str());
            new_stmts.push_back(CreateDebugStatement(msg));
            new_stmts.push_back(expr);
          }
          *exprs = new_stmts;
        };

    deal_with_exprs(&node->alloc_output_buffer_exprs);
    deal_with_exprs(&node->dealloc_output_buffer_exprs);
    deal_with_exprs(&node->buffer_data_cast_exprs);
    deal_with_exprs(&node->argument_prepare_exprs);

    body_block->stmts.insert(body_block->stmts.begin(), debug_node);
  }

  void Visit(const ir::Block *op, Expr *expr) {
    auto *node = expr->As<ir::Block>();
    std::vector<Expr> new_stmts;
    for (auto &e : op->stmts) {
      if (!IsDebugInfoNode(e)) {
        std::string msg;
        if (!e.As<ir::Store>()) {
          msg = StringFormat("running: %s", GetDebugString(e).c_str());
          auto debug_info_node = CreateDebugStatement(msg);
          new_stmts.push_back(debug_info_node);
        } else {
          auto _msg_args_ = StoreDebugInfo(e);
          auto &msg = std::get<0>(_msg_args_);
          auto &args = std::get<1>(_msg_args_);
          auto debug_info_node =
              CreateDebugStatement("running: " + msg, std::move(args));
          new_stmts.push_back(debug_info_node);
        }
      }

      ir::IRMutator<>::Visit(&e, &Reference(&e));

      new_stmts.push_back(e);

      if (!IsDebugInfoNode(e) && e.As<ir::Store>()) {
        auto _msg_args_ = StoreDebugInfo(e);
        auto &msg = std::get<0>(_msg_args_);
        auto &args = std::get<1>(_msg_args_);
        auto debug_info_node = CreateDebugStatement(msg, std::move(args));
        new_stmts.push_back(debug_info_node);

        {  // detailed debug
          auto _format_args_ = StoreDebugInfoBuilder()(&e);
          auto &format = std::get<0>(_format_args_);
          auto &args = std::get<1>(_format_args_);
          new_stmts.push_back(CreateDebugStatement(format, std::move(args)));
        }
      }
    }

    node->stmts = new_stmts;
  }

  std::string GetDebugString(const Expr &e) {
    std::stringstream ss;
    switch (e.node_type()) {
      case ir::IrNodeTy::Block:
        ss << "<block>";
        break;
      case ir::IrNodeTy::For: {
        auto *node = e.As<ir::For>();
        ss << "<For " << node->loop_var << " in [" << node->min << ", "
           << node->extent << ")>";
        break;
      }
      case ir::IrNodeTy::PolyFor: {
        auto *node = e.As<ir::PolyFor>();
        ss << "<PolyFor " << node->iterator << " in [" << node->init << ", "
           << node->ExtractExtent() << ")"
           << " with condition: " << node->condition << ">";
        break;
      }
      case ir::IrNodeTy::LoweredFunc: {
        auto *node = e.As<ir::_LoweredFunc_>();
        ss << "<LoweredFunc " << node->name << ">";
        break;
      }
      case ir::IrNodeTy::Call: {
        auto *node = e.As<ir::Call>();
        if (node->name == runtime::intrinsic::debug_log_repr) {
          return "";
        }
        ss << e;
        break;
      }
      default:
        ss << "NodeTy " << e->node_type() << ": " << e;
        break;
    }

    return ss.str();
  }

  std::tuple<std::string, std::vector<Expr>> StoreDebugInfo(const Expr &e) {
    const auto *node = e.As<ir::Store>();

    std::stringstream format_ss;

    {  // destination
      format_ss << node->tensor.as_tensor()->name << "[";
      for (auto &index : node->indices) format_ss << "%d ";
      format_ss << "] = %f";
    }

    format_ss << ", ";

    std::vector<Expr> val_reprs;
    for (auto &index : node->indices) val_reprs.push_back(index);
    val_reprs.push_back(ir::Load::Make(node->tensor, node->indices));

    return std::make_tuple(format_ss.str(), val_reprs);
  }

  inline bool IsDebugInfoNode(const Expr &e) {
    return e.As<ir::Call>() &&
           e.As<ir::Call>()->name == runtime::intrinsic::debug_log_repr;
  }

  Expr CreateDebugStatement(const std::string &msg,
                            std::vector<Expr> &&args = {}) {
    args.insert(args.begin(), Expr(msg));
    return ir::Call::Make(Void(),
                          runtime::intrinsic::debug_log_repr,
                          args,
                          {},
                          ir::CallType ::Intrinsic,
                          ir::FunctionRef(),
                          0);
  }
};

}  // namespace

void InsertDebugLogCallee(Expr *e) { InsertDebugLogCalleeMutator()(e); }

}  // namespace optim
}  // namespace cinn
