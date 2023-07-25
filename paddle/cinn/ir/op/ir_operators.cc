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

#include "paddle/cinn/ir/op/ir_operators.h"

#include <limits>
#include <string>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn {
namespace ir {
using attr_t = absl::variant<int, float, bool, std::string>;

Expr operator<<(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      CHECK(int_b->value >= 0 && int_b->value < t_a.bits())
          << "Shift amount must be non-negative and less than " << t_a.bits()
          << " for type " << t_a << std::endl;
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value << int_b->value);
    }
  }
  return lang::CallExtern("left_shift", {a, b}, {{"vectorizable", false}});
}

Expr operator>>(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      CHECK(int_b->value >= 0 && int_b->value < t_a.bits())
          << "Shift amount must be non-negative and less than " << t_a.bits()
          << " for type " << t_a << std::endl;
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value >> int_b->value);
    }
  }
  return lang::CallExtern("right_shift", {a, b}, {{"vectorizable", false}});
}

Expr operator|(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value | int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  if (target.arch == common::Target::Arch::X86) {
    return lang::CallExtern("bitwise_or", {a, b}, {{"vectorizable", false}});
  } else if (target.arch == common::Target::Arch::NVGPU) {
    auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
    return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
  } else {
    LOG(FATAL) << "Unsupport arch: " << target.arch_str() << " for bitwise_or.";
  }
}

Expr operator&(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value & int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  if (target.arch == common::Target::Arch::X86) {
    return lang::CallExtern("bitwise_and", {a, b}, {{"vectorizable", false}});
  } else if (target.arch == common::Target::Arch::NVGPU) {
    auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
    return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
  } else {
    LOG(FATAL) << "Unsupport arch: " << target.arch_str()
               << " for bitwise_and.";
  }
}

Expr operator^(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value ^ int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  if (target.arch == common::Target::Arch::X86) {
    return lang::CallExtern("bitwise_xor", {a, b}, {{"vectorizable", false}});
  } else if (target.arch == common::Target::Arch::NVGPU) {
    auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
    return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
  } else {
    LOG(FATAL) << "Unsupport arch: " << target.arch_str()
               << " for bitwise_xor.";
  }
}

Expr operator~(Expr a) {
  CHECK(a.type().is_int() || a.type().is_uint());
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  if (target.arch == common::Target::Arch::X86) {
    return lang::CallExtern("bitwise_not", {a}, {{"vectorizable", false}});
  } else if (target.arch == common::Target::Arch::NVGPU) {
    auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
    return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
  } else {
    LOG(FATAL) << "Unsupport arch: " << target.arch_str()
               << " for bitwise_not.";
  }
}

}  // namespace ir
}  // namespace cinn
