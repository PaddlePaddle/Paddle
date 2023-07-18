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

#include "paddle/cinn/optim/cast_simplify.h"

#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn::optim {

using cinn::common::bfloat16;
using cinn::common::float16;

namespace {

template <typename CastType, typename T>
CastType NormCastValue(T value) {
  if (type_of<CastType>().is_uint() || type_of<T>().is_uint()) {
    // not support uint
    return static_cast<CastType>(value);
  }

  if (std::isinf(value)) {
    return std::numeric_limits<CastType>::infinity();
  } else if (std::isnan(value)) {
    return std::numeric_limits<CastType>::signaling_NaN();
  } else if (value >= static_cast<T>(std::numeric_limits<CastType>::max())) {
    return std::numeric_limits<CastType>::max();
  } else if (value <= static_cast<T>(std::numeric_limits<CastType>::lowest())) {
    return std::numeric_limits<CastType>::lowest();
  }
  return static_cast<CastType>(value);
}

struct Mutator : ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::Cast* op, Expr* expr) {
    auto* node = expr->As<ir::Cast>();

    Visit(&node->v(), &node->v());

    if (op->type() == op->v().type()) {
      *expr = op->v();
      return;
    }

#define __CAST_TO_TYPE(type__)                                          \
  if (auto* i = op->v().As<ir::IntImm>()) {                             \
    *expr = Expr(static_cast<type__>(i->value));                        \
  } else if (auto* f = op->v().As<ir::FloatImm>()) {                    \
    *expr = Expr(static_cast<type__>(NormCastValue<type__>(f->value))); \
  } else if (auto* u = op->v().As<ir::UIntImm>()) {                     \
    *expr = Expr(static_cast<type__>(u->value));                        \
  } else {                                                              \
    CINN_NOT_IMPLEMENTED                                                \
  }

    if (op->v().is_constant()) {
      if (op->type() == type_of<int8_t>()) {
        __CAST_TO_TYPE(int8_t)
      } else if (op->type() == type_of<int16_t>()) {
        __CAST_TO_TYPE(int16_t)
      } else if (op->type() == type_of<int32_t>()) {
        __CAST_TO_TYPE(int32_t)
      } else if (op->type() == type_of<int64_t>()) {
        __CAST_TO_TYPE(int64_t)
      } else if (op->type() == type_of<uint8_t>()) {
        __CAST_TO_TYPE(uint8_t)
      } else if (op->type() == type_of<uint16_t>()) {
        __CAST_TO_TYPE(uint16_t)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<float>()) {
        __CAST_TO_TYPE(float)
      } else if (op->type() == type_of<double>()) {
        __CAST_TO_TYPE(double)
      } else if (op->type() == type_of<bool>()) {
        __CAST_TO_TYPE(bool)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<bfloat16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(bfloat16)
      } else if (op->type() == type_of<float16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(float16)
      } else {
        CINN_NOT_IMPLEMENTED
      }
    }
#undef __CAST_TO_TYPE
  }
};

}  // namespace

void CastSimplify(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
