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
#include <unordered_set>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace common {

namespace {

// ramp + scalar or broadcast
Expr RampRelatedMul(ir::Ramp *ramp, Expr other) {
  CHECK_EQ(other.type().ElementOf(), Int(32));
  CHECK_EQ(ramp->base.type(), Int(32));
  CHECK_EQ(ramp->stride.type(), Int(32));
  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    CHECK_EQ(ramp->lanes, other_broadcast->lanes);
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base * other, ramp->stride * other, ramp->lanes);
}

Expr RampRelatedMul(ir::Broadcast *broadcast, Expr other) {
  CHECK_EQ(other.type().lanes(), 1);
  return ir::Broadcast::Make(broadcast->value * other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedMul(ir::Ramp *ramp, ir::Ramp *other) {
  CINN_NOT_IMPLEMENTED
  return Expr();
}
// ramp + scalar
Expr RampRelatedAdd(ir::Ramp *ramp, Expr other) {
  CHECK_EQ(other.type().ElementOf(), Int(32));

  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    CHECK_EQ(ramp->lanes, other_broadcast->lanes);
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base + other, ramp->stride, ramp->lanes);
}
Expr RampRelatedAdd(ir::Broadcast *broadcast, Expr other) {
  CHECK_EQ(other.type().lanes(), 1);
  return ir::Broadcast::Make(broadcast->value + other, broadcast->lanes);
}
// ramp + ramp
Expr RampRelatedAdd(ir::Ramp *ramp, ir::Ramp *other) {
  CHECK(ramp);
  CHECK(other);
  if (ramp->lanes == other->lanes) {
    Expr base_add = common::AutoSimplify(ramp->base + other->base);
    Expr stride_add = common::AutoSimplify(ramp->stride + other->stride);
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
    CHECK_EQ(a_broadcast->lanes, b_broadcast->lanes);
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
    CHECK_EQ(a_broadcast->lanes, b_broadcast->lanes);
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
  CHECK_LE(shape.size(), indices.size());
  Expr res;
  for (int i = 0; i < shape.size(); i++) {
    CHECK_EQ(shape[i].type(), Int(32));
    Expr indice_prod = indices[i];
    optim::SimplifyCast(&indice_prod);
    for (int j = i + 1; j < shape.size(); j++) {
      indice_prod = RampRelatedMul(indice_prod, shape[j]);
    }
    if (res.defined()) {
      res = RampRelatedAdd(res, indice_prod);
    } else {
      res = indice_prod;
    }
  }
  return common::AutoSimplify(res);
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
  v = AutoSimplify(v);
  auto *int_n = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value = 0.f;
  return false;
}

Expr CastIfNeeded(Expr body, Type type) {
  if (body.type() == type) return body;
  return ir::Cast::Make(type, body);
}

bool MathEqual(const Expr &a, const Expr &b) {
  auto c = a - b;
  c = AutoSimplify(c);
  return is_zero(c);
}

Expr select(Expr cond, Expr true_value, Expr false_value) {
  return ir::Select::Make(cond, true_value, false_value);
}

Expr and_all(const std::vector<Expr> &conds) {
  CHECK(!conds.empty());
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::And::Make(res, conds[i]);
  }
  return res;
}

Expr or_all(const std::vector<Expr> &conds) {
  CHECK(!conds.empty());
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::Or::Make(res, conds[i]);
  }
  return res;
}

void CheckTensorUniqueInExpr(Expr expr) {
  auto tensor_uniq = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_tensor(); });
  absl::flat_hash_map<std::string, const ir::_Tensor_ *> tensor_names;
  for (auto &t : tensor_uniq) {
    auto *tp = t.as_tensor();
    if (!tensor_names.count(tp->name)) {
      tensor_names[tp->name] = tp;
    } else {
      CHECK_EQ(tensor_names[tp->name], tp)
          << "Found tensor not unique [" << tp->name
          << "]\nThe original expression is \n"
          << expr;
    }
  }
}

void CheckBufferUniqueInExpr(Expr expr) {
  // the buffers exists in tensor and lowered functions.
  CheckTensorUniqueInExpr(expr);

  auto tensors = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_tensor(); });
  auto funcs = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_lowered_func(); });

  absl::flat_hash_map<std::string, const ir::_Buffer_ *> buffer_name;
  auto check_buffer_uniq = [&](const ir::_Buffer_ *b) {
    if (buffer_name.count(b->name)) {
      CHECK_EQ(buffer_name[b->name], b);
    } else {
      buffer_name[b->name] = b->const_self();
    }
  };
  for (auto &e : tensors) {
    auto *t = e.as_tensor();
    if (t->buffer.defined()) {
      check_buffer_uniq(t->buffer->const_self());
    }
  }

  for (auto &e : funcs) {
    auto *f = e.as_lowered_func();
    for (auto &b : f->temp_bufs) {
      if (b.defined()) {
        check_buffer_uniq(b->const_self());
      }
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
        CHECK(iters.empty());
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
  CHECK_EQ(a.type(), b.type());
  return ir::Max::Make(a, b);
}

Expr min(Expr a, Expr b) {
  CHECK_EQ(a.type(), b.type());
  return ir::Min::Make(a, b);
}

}  // namespace common
}  // namespace cinn
