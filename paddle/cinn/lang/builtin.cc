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

#include "paddle/cinn/lang/builtin.h"

#include <cmath>
#include <limits>
#include <utility>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/buffer.h"

namespace cinn {
namespace lang {

using cinn::common::bfloat16;
using cinn::common::float16;

Expr logic_and(const std::vector<Expr>& conds) {
  PADDLE_ENFORCE_EQ(
      !conds.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The input conditions vector for logic_and should not be empty."));
  auto start = ir::And::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = ir::And::Make(start, conds[i]);
  }
  return start;
}

Expr logic_or(const std::vector<Expr>& conds) {
  PADDLE_ENFORCE_EQ(
      !conds.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The input conditions vector for logic_or should not be empty."));
  auto start = ir::Or::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = ir::Or::Make(start, conds[i]);
  }
  return start;
}

//! extern call op
#define EXTERN_CALL_IMP(name__, target__)                     \
  Expr name__(Expr e) {                                       \
    return ir::Call::Make(                                    \
        e->type(), #target__, {e}, {}, ir::CallType::Extern); \
  }

#define EXTERN_CALL_IMP_NO_VEC(name__, target__)      \
  Expr name__(Expr e) {                               \
    return ir::Call::Make(e->type(),                  \
                          #target__,                  \
                          {e},                        \
                          {},                         \
                          ir::CallType::Extern,       \
                          ir::FunctionRef(),          \
                          0,                          \
                          {{"vectorizable", false}}); \
  }

EXTERN_CALL_IMP(Exp, exp);
EXTERN_CALL_IMP_NO_VEC(Erf, erf);
EXTERN_CALL_IMP(Sqrt, sqrt);
EXTERN_CALL_IMP(Rsqrt, rsqrt);
EXTERN_CALL_IMP(Log, log);
EXTERN_CALL_IMP(Log2, log2);
EXTERN_CALL_IMP(Log10, log10);
EXTERN_CALL_IMP(Floor, floor);
EXTERN_CALL_IMP(Ceil, ceil);
EXTERN_CALL_IMP(Round, round);
EXTERN_CALL_IMP(Trunc, trunc);
EXTERN_CALL_IMP(Cos, cos);
EXTERN_CALL_IMP(Sin, sin);
EXTERN_CALL_IMP(Cosh, cosh);
EXTERN_CALL_IMP(Tan, tan);
EXTERN_CALL_IMP(Tanh, tanh);
EXTERN_CALL_IMP(Sinh, sinh);
EXTERN_CALL_IMP_NO_VEC(Acos, acos);
EXTERN_CALL_IMP_NO_VEC(Acosh, acosh);
EXTERN_CALL_IMP_NO_VEC(Asin, asin);
EXTERN_CALL_IMP_NO_VEC(Asinh, asinh);
EXTERN_CALL_IMP_NO_VEC(Atan, atan);
EXTERN_CALL_IMP_NO_VEC(Atanh, atanh);
EXTERN_CALL_IMP(Cbrt, cbrt);
EXTERN_CALL_IMP(Clz, clz);
EXTERN_CALL_IMP(Popc, popc);

#undef EXTERN_CALL_IMP
#undef EXTERN_CALL_IMP_NO_VEC

#define EXTERN_BINARY_CALL_IMP(name__, target__)                              \
  Expr name__(Expr a, Expr b) {                                               \
    PADDLE_ENFORCE_EQ(                                                        \
        a.type(),                                                             \
        b.type(),                                                             \
        ::common::errors::InvalidArgument(#name__ "'s inputs type not equal," \
                                                  "where a:%s but b:%s.",     \
                                          a.type(),                           \
                                          b.type()));                         \
    return ir::Call::Make(                                                    \
        a->type(), #target__, {a, b}, {}, ir::CallType::Extern);              \
  }

EXTERN_BINARY_CALL_IMP(Remainder, mod)
EXTERN_BINARY_CALL_IMP(LogicalRightShift, logical_right_shift)
EXTERN_BINARY_CALL_IMP(Pow, pow)
EXTERN_BINARY_CALL_IMP(Mod, mod)

#undef EXTERN_BINARY_CALL_IMP

Expr Zero(const Type& type) { return ir::Zero(type); }

Expr One(const Type& type) { return ir::One(type); }

Expr FloorDivide(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "FloorDivide's inputs type not equal, where a:%s "
                        " but b:%s.",
                        a.type(),
                        b.type()));
  if (a.type().is_float()) {
    return Floor(a / b);
  } else if (a.type().is_uint()) {
    return a / b;
  } else {
    auto div = a / b;
    auto mod = a % b;
    auto ret = ir::Select::Make(
        ir::EQ::Make(mod, cinn::common::make_const(a.type(), 0)),
        div,
        div - cinn::common::make_const(a.type(), 1));
    return ir::Select::Make((a > 0 && b > 0) || (a < 0 && b < 0), div, ret);
  }
}

Expr min_value(const Type& type) {
  PADDLE_ENFORCE_EQ(type.lanes(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The value of min type's lanes is incorrect"
                        "Expected value is 1, but receive %d. ",
                        type.lanes()));
#define FOR_CASE(type__)                                                     \
  if (type == type_of<type__>()) {                                           \
    return Expr(static_cast<type__>(std::numeric_limits<type__>::lowest())); \
  }
  FOR_CASE(int8_t)
  FOR_CASE(int16_t)
  FOR_CASE(int32_t)
  FOR_CASE(int64_t)
  FOR_CASE(uint8_t)
  FOR_CASE(uint16_t)
  FOR_CASE(uint32_t)
  FOR_CASE(uint64_t)
  FOR_CASE(bfloat16)
  FOR_CASE(float16)
  FOR_CASE(float)
  FOR_CASE(double)
#undef FOR_CASE
  return Expr();
}

Expr max_value(const Type& type) {
  PADDLE_ENFORCE_EQ(type.lanes(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The value of max type's lanes is incorrect"
                        "Expected value is 1, but receive %d. ",
                        type.lanes()));

#define FOR_CASE(type__)                                                  \
  if (type == type_of<type__>()) {                                        \
    return Expr(static_cast<type__>(std::numeric_limits<type__>::max())); \
  }
  FOR_CASE(int8_t)
  FOR_CASE(int16_t)
  FOR_CASE(int32_t)
  FOR_CASE(int64_t)
  FOR_CASE(uint8_t)
  FOR_CASE(uint16_t)
  FOR_CASE(uint32_t)
  FOR_CASE(uint64_t)
  FOR_CASE(bfloat16)
  FOR_CASE(float16)
  FOR_CASE(float)
  FOR_CASE(double)
#undef FOR_CASE

  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr Epsilon(const Type& type) {
  PADDLE_ENFORCE_EQ(type.lanes(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The value of epsilon type's lanes is incorrect"
                        "Expected value is 1, but receive %d. ",
                        type.lanes()));

#define FOR_CASE(type__)                                                      \
  if (type == type_of<type__>()) {                                            \
    return Expr(static_cast<type__>(std::numeric_limits<type__>::epsilon())); \
  }
  FOR_CASE(int8_t)
  FOR_CASE(int16_t)
  FOR_CASE(int32_t)
  FOR_CASE(int64_t)
  FOR_CASE(uint8_t)
  FOR_CASE(uint16_t)
  FOR_CASE(uint32_t)
  FOR_CASE(uint64_t)
  FOR_CASE(bfloat16)
  FOR_CASE(float16)
  FOR_CASE(float)
  FOR_CASE(double)
#undef FOR_CASE

  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr Abs(Expr e) {
  Type type = e->type();
  Type bool_type = Bool(type.lanes());
  if (type.is_uint()) {
    return e;
  } else if (type.is_int() || type.is_float()) {
    auto node = e.As<ir::IntImm>();
    if (node) {
      return make_const(type, std::abs(node->value));
    }
    return ir::Select::Make(e > Zero(e->type()), e, -e);
  } else {
    std::stringstream ss;
    ss << "Abs Not support data type " << type;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return e;
}

Expr IsNan(Expr e) {
  Type type = e->type();
  if (type.is_int() || type.is_uint()) {
    return cinn::common::make_bool(false, type.lanes());
  } else if (type.is_float()) {
    auto* node = e.As<ir::FloatImm>();
    if (node) {
      return cinn::common::make_bool(std::isnan(node->value), type.lanes());
    }
    return CallExtern("isnan", {e}, {{"vectorizable", false}});
  } else {
    std::stringstream ss;
    ss << type << "is not supported for isnan op.";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    return e;
  }
}

Expr Infinity(const Type& type) {
  PADDLE_ENFORCE_EQ(type.lanes(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The value of infinity type's lanes is incorrect"
                        "Expected value is 1, but receive %d. ",
                        type.lanes()));
  if (type.is_float()) {
    if (type.bits() == 64) {
      return make_const(type, std::numeric_limits<double>::infinity());
    } else if (type.bits() == 32) {
      return make_const(type, std::numeric_limits<float>::infinity());
    } else if (type.bits() == 16) {
      return make_const(type, std::numeric_limits<float16>::infinity());
    }
  }
  std::stringstream ss;
  ss << "Cannot decide infinity for type " << type;
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  return Expr();
}

Expr IsInf(Expr e) {
  Type type = e->type();
  if (type.is_int() || type.is_uint()) {
    return cinn::common::make_bool(false, type.lanes());
  } else if (type.is_float()) {
    auto* node = e.As<ir::FloatImm>();
    if (node) {
      return cinn::common::make_bool(std::isinf(node->value), type.lanes());
    }
    return CallExtern("isinf", {e}, {{"vectorizable", false}});
  } else {
    std::stringstream ss;
    ss << type << "is not supported for isinf op.";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    return e;
  }
}

Expr IsFinite(Expr e) { return !IsInf(e) && !IsNan(e); }

}  // namespace lang
}  // namespace cinn
