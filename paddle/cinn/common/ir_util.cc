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

#include "paddle/cinn/common/ir_util.h"

#include <algorithm>
#include <stack>
#include <unordered_set>

#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

namespace {

// ramp + scalar or broadcast
Expr RampRelatedMul(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));
  PADDLE_ENFORCE_EQ(ramp->base.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->base should be int32."));
  PADDLE_ENFORCE_EQ(ramp->stride.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->stride should be int32."));
  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base * other, ramp->stride * other, ramp->lanes);
}

Expr RampRelatedMul(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value * other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedMul(ir::Ramp *ramp, ir::Ramp *other) {
  CINN_NOT_IMPLEMENTED
  return Expr();
}
// ramp + scalar
Expr RampRelatedAdd(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));

  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base + other, ramp->stride, ramp->lanes);
}
Expr RampRelatedAdd(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value + other, broadcast->lanes);
}
// ramp + ramp
Expr RampRelatedAdd(ir::Ramp *ramp, ir::Ramp *other) {
  PADDLE_ENFORCE_NOT_NULL(
      ramp,
      ::common::errors::InvalidArgument("Ramp pointer should not be null."));
  PADDLE_ENFORCE_NOT_NULL(other,
                          ::common::errors::InvalidArgument(
                              "Other ramp pointer should not be null."));
  if (ramp->lanes == other->lanes) {
    Expr base_add = optim::ArithSimplify(ramp->base + other->base);
    Expr stride_add = optim::ArithSimplify(ramp->stride + other->stride);
    VLOG(2) << base_add;
    VLOG(2) << stride_add;
    return ir::Ramp::Make(base_add, stride_add, ramp->lanes);
  }
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr RampRelatedAdd(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (b->type().lanes() == 1 || b_broadcast)) {
    return RampRelatedAdd(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().lanes() == 1 || a_broadcast)) {
    return RampRelatedAdd(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a + b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedAdd(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedAdd(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedAdd(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value + b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

Expr RampRelatedMul(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (!b->type().is_vector() || b_broadcast)) {
    return RampRelatedMul(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().is_vector() || a_broadcast)) {
    return RampRelatedMul(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a * b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedMul(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedMul(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedMul(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value * b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    VLOG(3) << "a,b: " << a << " " << b;
    CINN_NOT_IMPLEMENTED
  }
}

}  // namespace

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices) {
  VLOG(3) << "Begin IndiceToAbsOffset";
  VLOG(3) << "shape is : " << utils::Join(shape, ",");
  VLOG(3) << "indices is : " << utils::Join(indices, ",");
  PADDLE_ENFORCE_LE(shape.size(),
                    indices.size(),
                    ::common::errors::InvalidArgument(
                        "The size of shape should be less than or "
                        "equal to the size of indices."));
  Expr res(0);

  for (int32_t i = 0; i < shape.size(); i++) {
    PADDLE_ENFORCE_EQ(
        shape[i].type() == Int(64) || shape[i].type() == Int(32),
        true,
        ::common::errors::InvalidArgument(
            "The shape data type currently supports only int32 or int64, but "
            "the current data type of shape[{}] is {}",
            i,
            shape[i].type()));

    Expr indice_cast = indices[i];
    optim::SimplifyCast(&indice_cast);
    res = RampRelatedAdd(RampRelatedMul(res, shape[i]), indice_cast);
    if (res.is_index()) {
      res = res.as_index().Normalize(ir::IndexExpr::OptLevel::kLevel2);
    } else {
      VLOG(8) << "**** expr is not index ****: " << res;
    }
  }
  VLOG(3) << "End IndiceToAbsOffset";

  return res;
}

Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return IndiceToAbsOffset(shape, indices);
}

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis) {
  std::vector<Expr> indices;
  for (int i = 0; i < preceding_n_axis; i++) indices.push_back(shape[i]);
  return IndiceToAbsOffset(shape, indices);
}

namespace {

class SubstituteMutator : ir::IRMutator<ir::Expr *> {
 public:
  explicit SubstituteMutator(const std::map<const ir::_Var_ *, Expr> &var_map) {
    for (auto &item : var_map) {
      var_map_[item.first->name] = item.second;
    }
  }

  void operator()(ir::Expr *expr) { Visit(expr); }

 private:
  void Visit(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *op, ir::Expr *expr) override {
    auto it = var_map_.find(op->name);
    if (it == var_map_.end()) return;
    *expr = it->second;
  }

  Expr *expr_{};
  std::map<std::string, Expr> var_map_;
};

}  // namespace

void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map) {
  SubstituteMutator mutator(var_map);
  mutator(expr);
}

bool is_zero(Expr v) {
  v = optim::ArithSimplify(v);
  auto *int_n = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value == 0.f;
  return false;
}

Expr CastIfNeeded(Expr body, Type type) {
  if (body.type() == type) return body;
  return ir::Cast::Make(type, body);
}

bool MathEqual(const Expr &a, const Expr &b) {
  auto c = a - b;
  c = optim::ArithSimplify(c);
  return is_zero(c);
}

Expr select(Expr cond, Expr true_value, Expr false_value) {
  return ir::Select::Make(cond, true_value, false_value);
}

Expr and_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::And::Make(res, conds[i]);
  }
  return res;
}

Expr or_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::Or::Make(res, conds[i]);
  }
  return res;
}

void CheckTensorUniqueInExpr(Expr expr) {
  auto tensor_uniq = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_tensor(); });
  paddle::flat_hash_map<std::string, const ir::_Tensor_ *> tensor_names;
  for (auto &t : tensor_uniq) {
    auto *tp = t.as_tensor();
    if (!tensor_names.count(tp->name)) {
      tensor_names[tp->name] = tp;
    } else {
      PADDLE_ENFORCE_EQ(
          tensor_names[tp->name],
          tp,
          ::common::errors::InvalidArgument(
              "Found tensor not unique, The original express is %d .", expr));
    }
  }
}

Expr cast(Expr e, Type type) {
  if (e.is_constant()) {
    if (type.is_bool()) {
      return Expr(static_cast<bool>(e.get_constant()));
    } else if (type.is_int(8)) {
      return Expr(static_cast<int8_t>(e.get_constant()));
    } else if (type.is_int(16)) {
      return Expr(static_cast<int16_t>(e.get_constant()));
    } else if (type.is_int(32)) {
      return Expr(static_cast<int32_t>(e.get_constant()));
    } else if (type.is_int(64)) {
      return Expr(static_cast<int64_t>(e.get_constant()));
    } else if (type.is_uint(8)) {
      return Expr(static_cast<uint8_t>(e.get_constant()));
    } else if (type.is_uint(16)) {
      return Expr(static_cast<uint16_t>(e.get_constant()));
    } else if (type.is_uint(32)) {
      return Expr(static_cast<uint32_t>(e.get_constant()));
    } else if (type.is_uint(64)) {
      return Expr(static_cast<uint64_t>(e.get_constant()));
    } else if (type.is_float(32)) {
      return Expr(static_cast<float>(e.get_constant()));
    } else if (type.is_float(64)) {
      return Expr(static_cast<double>(e.get_constant()));
    } else if (type.is_bfloat16()) {
      return Expr(static_cast<cinn::common::bfloat16>(e.get_constant()));
    } else if (type.is_float16()) {
      return Expr(static_cast<cinn::common::float16>(e.get_constant()));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }

  return ir::Cast::Make(type, e);
}

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr) {
  struct Visitor : public ir::IRMutator<> {
    std::vector<std::string> iters;
    const std::string &target_tensor_name;

    explicit Visitor(const std::string &target_tensor_name)
        : target_tensor_name(target_tensor_name) {}

    std::vector<std::string> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return iters;
    }

    void Visit(const ir::Store *op, Expr *expr) {
      if (op->tensor.as_tensor()->name == target_tensor_name) {
        PADDLE_ENFORCE_EQ(iters.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The iterators vector should be empty."));
        for (auto &e : for_stack) {
          auto *for_n = e->As<ir::For>();
          auto *polyfor_n = e->As<ir::PolyFor>();
          if (for_n) {
            iters.push_back(for_n->loop_var->name);
          } else {
            iters.push_back(polyfor_n->iterator->name);
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }

    std::vector<Expr *> for_stack;
  };

  return Visitor(target_tensor_name)(expr);
}

std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name) {
  VLOG(4) << "search store " << tensor_name << " in expr:\n";
  VLOG(4) << *expr;
  struct Mutator : public ir::IRMutator<> {
    std::vector<Expr *> forloop_stack;
    bool found{false};

    std::string tensor_name;

    explicit Mutator(const std::string &tensor_name)
        : tensor_name(tensor_name) {}

    std::vector<Expr *> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return forloop_stack;
    }

    void Visit(const ir::For *op, Expr *expr) {
      auto *node = expr->As<ir::For>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::PolyFor *op, Expr *expr) {
      auto *node = expr->As<ir::PolyFor>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::Store *op, Expr *expr) {
      found = op->tensor.as_tensor()->name == tensor_name;
    }
  };

  return Mutator(tensor_name)(expr);
}

Expr max(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Max::Make(a, b);
}

Expr min(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Min::Make(a, b);
}

void OpDataTypePromote(Expr *expr) {
  struct TypePromote : public ir::IRMutator<> {
    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }
    // type promote for operand of binary op
#define __(op__)                                            \
  void Visit(const ir::op__ *op, ir::Expr *expr) override { \
    ir::TryElevateInt32ToInt64_((*expr)->operands);         \
    IRMutator::Visit(op, expr);                             \
  };
    __(Sum)
    __(Product)
    NODETY_BINARY_OP_FOR_EACH(__)
#undef __

    void Visit(const ir::Select *op, ir::Expr *expr) override {
      auto node = expr->As<ir::Select>();

      auto promote_args = std::move(
          ir::TryElevateInt32ToInt64({node->true_value, node->false_value}));
      node->true_value = promote_args.at(0);
      node->false_value = promote_args.at(1);

      IRMutator::Visit(op, expr);
    }

    void Visit(const ir::Load *op, ir::Expr *expr) {
      auto node = expr->As<ir::Load>();
      ir::TryElevateInt32ToInt64_(node->indices);
      IRMutator::Visit(op, expr);
    }

    void Visit(const ir::Store *op, ir::Expr *expr) {
      auto node = expr->As<ir::Store>();
      ir::TryElevateInt32ToInt64_(node->indices);
      IRMutator::Visit(op, expr);
    }

    void Visit(const ir::Let *op, ir::Expr *expr) {
      auto node = expr->As<ir::Let>();
      // For Symbol of LetOp, we need to insert a cast to convert its type, but
      // inside LetOp, we should directly convert the Symbol type instead of
      // inserting a cast.so we set the flag to false before the conversion and
      // set it to true after the conversion, e.g.
      // inside LetOp: type of v, v1 are int32.
      //   int32 v = v1 * 2   ==TypePromote==>  int64 v = v1 * 2ll
      // outside LetOp: type of v, v2 are int32 and v is defined by LetOp.
      //   v2 = v * 2         ==TypePromote==>  v2 = (int64)v * 2ll
      if (node->symbol.is_var()) {
        node->symbol.as_var()->is_let_symbol = false;
      }
      auto promote_args =
          std::move(ir::TryElevateInt32ToInt64({node->symbol, node->body}));
      node->symbol = promote_args.at(0);
      node->body = promote_args.at(1);
      if (node->symbol.is_var()) {
        node->symbol.as_var()->is_let_symbol = true;
      }
      IRMutator::Visit(op, expr);
    }
  };

  TypePromote visitor;
  visitor(expr);
}

void OpDataTypePromote(ir::Module *module) {
  auto node = module->As<ir::_Module_>();
  for (auto &func : node->functions) {
    OpDataTypePromote(&func->body);
  }
  for (auto &buffer : node->buffers) {
    OpDataTypePromote(&buffer);
  }
  for (auto &submodule : node->submodules) {
    OpDataTypePromote(&submodule);
  }
}

void OpDataTypePromote(ir::LoweredFunc *func) {
  auto node = func->As<ir::_LoweredFunc_>();
  OpDataTypePromote(&node->body);
}
}  // namespace common
}  // namespace cinn
