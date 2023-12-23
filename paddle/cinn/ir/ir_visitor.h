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

#pragma once
#include <functional>
#include <set>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"

namespace cinn {
namespace ir {

struct _Tensor_;

/**
 * Base class of all the methods visit the IR tree.
 * @param RetTy return type.
 * @param Args type of the extra arguments passed to the all the methods.
 */
template <typename RetTy = void, typename... Args>
class IRVisitorRequireReImpl {
 public:
  //! Visit a expression.
  // @{
  virtual RetTy Visit(const ir::Expr* expr, Args... args) {
    CHECK(expr->defined());
    switch (expr->node_type()) {
#define __(op__)           \
  case ir::IrNodeTy::op__: \
    return Visit(expr->As<ir::op__>(), args...);

      NODETY_FORALL(__)

      default:
        LOG(FATAL) << "not supported NodeTy";
#undef __
    }
    return RetTy();
  }
  // @}
 protected:
#define __(op__) virtual RetTy Visit(const ir::op__* op, Args... args) = 0;
  NODETY_FORALL(__)
#undef __
};

/**
 * Base of all the Ir readonly visitor.
 */
struct IRVisitor : public IRVisitorRequireReImpl<void> {
  IRVisitor() = default;

  void Visit(const Expr* x) { IRVisitorRequireReImpl::Visit(x); }
#define __m(t__) \
  virtual void Visit(const t__* x) { return VisitDefault(x); }
  NODETY_FORALL(__m)
#undef __m

  virtual void VisitDefault(const Object* obj) {
    LOG(FATAL) << "not supported NodeTy";
  }
};
// std::set<Expr> CollectIRNodes(Expr expr, std::function<bool(const Expr*)>
// teller);

bool operator==(Expr a, Expr b);
bool operator!=(Expr a, Expr b);

}  // namespace ir
}  // namespace cinn
