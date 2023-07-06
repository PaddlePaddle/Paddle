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

/**
 * This file implements the IRMutator as the base interface to mutate the IR.
 */
#pragma once

#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

//! T might be Expr* or const Expr*
template <typename T = Expr *>
class IRMutator : public IRVisitorRequireReImplVisitor<void, T> {
 public:
  void Visit(const Expr *expr, T op) override;

#define __(op__) void Visit(const op__ *op, T expr) override;
  NODETY_FORALL(__)
#undef __
};

template <typename T>
void IRMutator<T>::Visit(const Expr *expr, T op) {
  IRVisitorRequireReImplVisitor<void, T>::Visit(expr, op);
}

#define UNARY_OP_IMPL(op__)                                                \
  template <typename T>                                                    \
  void IRMutator<T>::Visit(const op__ *expr, T op) {                       \
    auto *node = op->template As<op__>();                                  \
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->v(), &node->v()); \
  }

#define BINARY_OP_IMPL(op__)                                               \
  template <typename T>                                                    \
  void IRMutator<T>::Visit(const op__ *expr, T op) {                       \
    auto *node = op->template As<op__>();                                  \
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->a(), &node->a()); \
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->b(), &node->b()); \
  }

NODETY_UNARY_OP_FOR_EACH(UNARY_OP_IMPL)
NODETY_BINARY_OP_FOR_EACH(BINARY_OP_IMPL)

#undef UNARY_OP_IMPL
#undef BINARY_OP_IMPL

template <typename T>
void IRMutator<T>::Visit(const IntImm *expr, T op) {}
template <typename T>
void IRMutator<T>::Visit(const UIntImm *expr, T op) {}
template <typename T>
void IRMutator<T>::Visit(const FloatImm *expr, T op) {}
template <typename T>
void IRMutator<T>::Visit(const StringImm *expr, T op) {}
template <typename T>
void IRMutator<T>::Visit(const Cast *expr, T op) {
  auto *node = op->template As<Cast>();
  Visit(&node->v(), &node->v());
}
template <typename T>
void IRMutator<T>::Visit(const For *expr, T op) {
  auto *node = op->template As<For>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->min, &node->min);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->extent, &node->extent);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &node->body);
}
template <typename T>
void IRMutator<T>::Visit(const PolyFor *expr, T op) {
  auto *node = op->template As<PolyFor>();
  // IRVisitorRequireReImplVisitor<void,T>::Visit(&node->iterator,
  // &node->iterator);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->init, &node->init);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->condition,
                                                &node->condition);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->inc, &node->inc);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &node->body);
}
template <typename T>
void IRMutator<T>::Visit(const Select *expr, T op) {
  auto *node = op->template As<Select>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->condition,
                                                &node->condition);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->true_value,
                                                &node->true_value);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->false_value,
                                                &node->false_value);
}
template <typename T>
void IRMutator<T>::Visit(const IfThenElse *expr, T op) {
  auto *node = op->template As<IfThenElse>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->condition,
                                                &node->condition);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->true_case,
                                                &node->true_case);
  if (node->false_case.defined())
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->false_case,
                                                  &node->false_case);
}
template <typename T>
void IRMutator<T>::Visit(const Block *expr, T op) {
  auto *node = op->template As<Block>();
  for (auto &expr : node->stmts) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&expr, &expr);
  }
}
template <typename T>
void IRMutator<T>::Visit(const Call *expr, T op) {
  auto *node = op->template As<Call>();
  for (auto &expr : node->read_args) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&expr, &expr);
  }
  for (auto &expr : node->write_args) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&expr, &expr);
  }
}
template <typename T>
void IRMutator<T>::Visit(const _Module_ *expr, T op) {
  auto *node = op->template As<_Module_>();
  for (auto &func : node->functions) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&func, &func);
  }
  for (auto &func : node->buffers) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&func, &func);
  }
  for (auto &expr : node->submodules) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&expr, &expr);
  }
}
template <typename T>
void IRMutator<T>::Visit(const _Var_ *expr, T op) {
  auto *node = op->template As<ir::_Var_>();
  if (node->lower_bound.defined()) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->lower_bound,
                                                  &node->lower_bound);
  }
  if (node->upper_bound.defined()) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->upper_bound,
                                                  &node->upper_bound);
  }
}
template <typename T>
void IRMutator<T>::Visit(const Load *expr, T op) {
  auto *node = op->template As<Load>();
  for (auto &idx : node->indices)
    IRVisitorRequireReImplVisitor<void, T>::Visit(&idx, &idx);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->tensor, &node->tensor);
}
template <typename T>
void IRMutator<T>::Visit(const Store *expr, T op) {
  auto *node = op->template As<Store>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->value, &node->value);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->tensor, &node->tensor);
  for (auto &idx : node->indices)
    IRVisitorRequireReImplVisitor<void, T>::Visit(&idx, &idx);
}
template <typename T>
void IRMutator<T>::Visit(const Alloc *expr, T op) {
  auto *node = op->template As<Alloc>();
  for (auto &e : node->extents) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&e, &e);
  }

  if (node->condition.defined())
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->condition,
                                                  &node->condition);
  if (node->body.defined()) {
    Expr body(node->body);
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &body);
  }
}
template <typename T>
void IRMutator<T>::Visit(const Free *expr, T op) {
  auto *node = op->template As<Free>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->destination,
                                                &node->destination);
}
template <typename T>
void IRMutator<T>::Visit(const _Buffer_ *expr, T op) {
  auto *node = op->template As<_Buffer_>();

  for (auto &e : node->shape) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&e, &e);
  }
  for (auto &e : node->strides) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&e, &e);
  }
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->elem_offset,
                                                &node->elem_offset);
}
template <typename T>
void IRMutator<T>::Visit(const _Tensor_ *expr, T op) {
  auto *node = op->template As<_Tensor_>();

  for (auto &e : node->shape) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&e, &e);
  }
}
template <typename T>
void IRMutator<T>::Visit(const _LoweredFunc_ *expr, T op) {
  auto *node = op->template As<_LoweredFunc_>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &node->body);
}
template <typename T>
void IRMutator<T>::Visit(const Let *expr, T op) {
  auto *node = op->template As<Let>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->symbol, &node->symbol);
  if (node->body.defined())
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &node->body);
}
template <typename T>
void IRMutator<T>::Visit(const Reduce *expr, T op) {
  auto *node = op->template As<Reduce>();
  if (node->init.defined())
    IRVisitorRequireReImplVisitor<void, T>::Visit(&node->init, &node->init);
  CHECK(node->body.defined());
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->body, &node->body);
}

template <typename T>
void IRMutator<T>::Visit(const Ramp *expr, T op) {
  auto *node = op->template As<Ramp>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->base, &node->base);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->stride, &node->stride);
}

template <typename T>
void IRMutator<T>::Visit(const Broadcast *expr, T op) {
  auto *node = op->template As<Broadcast>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->value, &node->value);
}

template <typename T>
void IRMutator<T>::Visit(const FracOp *expr, T op) {
  auto *node = op->template As<FracOp>();
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->a(), &node->a());
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->b(), &node->b());
}

template <typename T>
void IRMutator<T>::Visit(const Product *expr, T op) {
  auto *node = op->template As<Product>();
  for (auto &x : node->operands()) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&x, &x);
  }
}

template <typename T>
void IRMutator<T>::Visit(const Sum *expr, T op) {
  auto *node = op->template As<Sum>();
  for (auto &x : node->operands()) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&x, &x);
  }
}
template <typename T>
void IRMutator<T>::Visit(const PrimitiveNode *expr, T op) {
  auto *node = op->template As<PrimitiveNode>();
  for (auto &args : node->arguments) {
    for (auto &arg : args) {
      IRVisitorRequireReImplVisitor<void, T>::Visit(&arg, &arg);
    }
  }
}

template <typename T>
void IRMutator<T>::Visit(const IntrinsicOp *expr, T op) {
  auto *node = op->template As<IntrinsicOp>();
  switch (node->getKind()) {
    case ir::IntrinsicKind::kBufferGetDataHandle: {
      auto *n = llvm::dyn_cast<intrinsics::BufferGetDataHandle>(node);
      Visit(&n->buffer, &n->buffer);
    } break;
    case ir::IntrinsicKind::kBufferGetDataConstHandle: {
      auto *n = llvm::dyn_cast<intrinsics::BufferGetDataConstHandle>(node);
      Visit(&n->buffer, &n->buffer);
    } break;
    case ir::IntrinsicKind::kPodValueToX: {
      auto *n = llvm::dyn_cast<intrinsics::PodValueToX>(node);
      Visit(&n->pod_value_ptr, &n->pod_value_ptr);
    } break;
    case ir::IntrinsicKind::kBuiltinIntrin: {
      auto *n = llvm::dyn_cast<intrinsics::BuiltinIntrin>(node);
      for (auto &expr : n->args) {
        Visit(&expr, &expr);
      }
    } break;
  }
}

template <typename T>
void IRMutator<T>::Visit(const _BufferRange_ *expr, T op) {
  auto *node = op->template As<_BufferRange_>();
  CHECK(node);
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->buffer, &node->buffer);
  for (auto &var : node->ranges) {
    if (var->lower_bound.defined()) {
      IRVisitorRequireReImplVisitor<void, T>::Visit(&var->lower_bound,
                                                    &var->lower_bound);
    }
    if (var->upper_bound.defined()) {
      IRVisitorRequireReImplVisitor<void, T>::Visit(&var->upper_bound,
                                                    &var->upper_bound);
    }
  }
}

template <typename T>
void IRMutator<T>::Visit(const ScheduleBlock *expr, T op) {
  auto *node = op->template As<ScheduleBlock>();
  CHECK(node);
  for (auto &var : node->iter_vars) {
    if (var->lower_bound.defined()) {
      IRVisitorRequireReImplVisitor<void, T>::Visit(&var->lower_bound,
                                                    &var->lower_bound);
    }
    if (var->upper_bound.defined()) {
      IRVisitorRequireReImplVisitor<void, T>::Visit(&var->upper_bound,
                                                    &var->upper_bound);
    }
  }
  for (auto &buffer_region : node->read_buffers) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&buffer_region,
                                                  &buffer_region);
  }
  for (auto &buffer_region : node->write_buffers) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&buffer_region,
                                                  &buffer_region);
  }
  IRVisitorRequireReImplVisitor<void, T>::Visit(&(node->body), &(node->body));
}

template <typename T>
void IRMutator<T>::Visit(const ScheduleBlockRealize *expr, T op) {
  auto *node = op->template As<ScheduleBlockRealize>();
  CHECK(node);
  for (auto &value : node->iter_values) {
    IRVisitorRequireReImplVisitor<void, T>::Visit(&value, &value);
  }
  IRVisitorRequireReImplVisitor<void, T>::Visit(&node->schedule_block,
                                                &node->schedule_block);
}

}  // namespace ir
}  // namespace cinn
