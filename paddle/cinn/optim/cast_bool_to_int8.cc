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

#include "paddle/cinn/optim/cast_bool_to_int8.h"

#include <glog/logging.h>

#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "Expected 'node' to be non-null, but got null."));

    auto value = node->value;
    if (op->type().is_bool() && op->value->type().is_bool()) {
      value = ir::Cast::Make(Int(8), value);
      *expr = ir::Store::Make(node->tensor, value, node->indices);
    }
  }
};

}  // namespace

void CastBoolExprToInt8Impl(common::UnknownArch, Expr* e) {
  LOG(FATAL) << "unknown architecture.";
}

void CastBoolExprToInt8Impl(common::X86Arch, Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

void CastBoolExprToInt8Impl(common::ARMArch, Expr* e) {
  // Do nothing.
}

void CastBoolExprToInt8Impl(common::NVGPUArch, Expr* e) {
  // Do nothing.
}

void CastBoolExprToInt8Impl(common::HygonDCUArchHIP, Expr* e) {
  // Do nothing.
}

void CastBoolExprToInt8Impl(common::HygonDCUArchSYCL, Expr* e) {
  // Do nothing.
}

void CastBoolExprToInt8(common::Arch arch, Expr* e) {
  return std::visit(
      [&](const auto& impl) { return CastBoolExprToInt8Impl(impl, e); },
      arch.variant());
}

void CastBoolToInt8(Expr* e, Target target) {
  CastBoolExprToInt8(target.arch, e);
}
}  // namespace cinn::optim
