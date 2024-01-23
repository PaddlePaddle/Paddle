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

#include "paddle/cinn/optim/trans_buffer_with_dynamic_shape.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/string.h"

namespace cinn::optim {

namespace {

struct RemoveSymbolMutator : public ir::IRMutator<> {
  explicit RemoveSymbolMutator(bool use_upper_bound) {
    this->use_upper_bound = use_upper_bound;
  }

  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const ir::_Var_* var, Expr* expr) override {
    auto node = expr->As<ir::_Var_>();
    CHECK(node->lower_bound.defined() && node->lower_bound.is_constant() &&
          node->upper_bound.defined() && node->upper_bound.is_constant())
        << "Temporary buffer with dynamic shape cannot be transformed into "
           "buffer with static shape!\n";
    if (use_upper_bound) {
      *expr = Expr(node->upper_bound.as_int32());
    } else {
      *expr = Expr(node->lower_bound.as_int32());
    }
  }

  bool use_upper_bound;
};

RemoveSymbolMutator max_mutator(true);
RemoveSymbolMutator min_mutator(false);

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::_Tensor_* tensor, Expr* expr) override {
    bool need_trans = false;
    if (cinn::utils::Endswith(tensor->name, "_temp_buffer")) need_trans = true;
    for (auto& e : expr->as_tensor()->shape) {
      if (!e.is_constant()) {
        auto new_shape1 = ir::ir_utils::IRCopy(e);
        auto new_shape2 = ir::ir_utils::IRCopy(e);
        max_mutator(&new_shape1);
        min_mutator(&new_shape2);
        new_shape1 = common::AutoSimplify(new_shape1);
        new_shape2 = common::AutoSimplify(new_shape2);
        CHECK(new_shape1.is_constant());
        CHECK(new_shape2.is_constant());
        int res_shape = std::max(new_shape1.as_int32(), new_shape2.as_int32());
        e = ir::Expr(res_shape);
      }
      Visit(&e, &e);
    }
  }
};

}  // namespace

void TransBufferWithDynamicShape(ir::Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
