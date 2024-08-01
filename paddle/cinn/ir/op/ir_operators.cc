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
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(
      b.type().is_int() || b.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      PADDLE_ENFORCE_EQ(
          int_b->value >= 0 && int_b->value < t_a.bits(),
          true,
          phi::errors::InvalidArgument(
              "Shift amount must be non-negative and less than %d for type %s.",
              t_a.bits(),
              t_a));
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value << int_b->value);
    }
  }
  return lang::CallExtern("left_shift", {a, b}, {{"vectorizable", false}});
}

Expr operator>>(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(
      b.type().is_int() || b.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      PADDLE_ENFORCE_EQ(
          int_b->value >= 0 && int_b->value < t_a.bits(),
          true,
          phi::errors::InvalidArgument(
              "Shift amount must be non-negative and less than %d for type %s.",
              t_a.bits(),
              t_a));
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value >> int_b->value);
    }
  }
  return lang::CallExtern("right_shift", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::UnknownArch,
                       const Target& target,
                       Expr a,
                       Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_or.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseOrCallImpl(common::X86Arch, const Target& target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_or", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::ARMArch, const Target& target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_or.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseOrCallImpl(common::NVGPUArch,
                       const Target& target,
                       Expr a,
                       Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::HygonDCUArchHIP,
                       const Target& target,
                       Expr a,
                       Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCall(const Target& target, Expr a, Expr b) {
  return std::visit(
      [&](const auto& arch) { return BitwiseOrCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator|(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(
      b.type().is_int() || b.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
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
  return BitwiseOrCall(target, a, b);
}

Expr BitwiseAndCallImpl(common::UnknownArch,
                        const Target& target,
                        Expr a,
                        Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_and.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseAndCallImpl(common::X86Arch, const Target& target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_and", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCallImpl(common::ARMArch, const Target& target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_and.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseAndCallImpl(common::NVGPUArch,
                        const Target& target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCallImpl(common::HygonDCUArchHIP,
                        const Target& target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCall(const Target& target, Expr a, Expr b) {
  return std::visit(
      [&](const auto& arch) { return BitwiseAndCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator&(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(
      b.type().is_int() || b.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
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
  return BitwiseAndCall(target, a, b);
}

Expr BitwiseXorCallImpl(common::UnknownArch,
                        const Target& target,
                        Expr a,
                        Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_xor.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseXorCallImpl(common::X86Arch, const Target& target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_xor", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCallImpl(common::ARMArch, const Target& target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_xor.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseXorCallImpl(common::NVGPUArch,
                        const Target& target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCallImpl(common::HygonDCUArchHIP,
                        const Target& target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCall(const Target& target, Expr a, Expr b) {
  return std::visit(
      [&](const auto& arch) { return BitwiseXorCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator^(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(
      b.type().is_int() || b.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
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
  return BitwiseXorCall(target, a, b);
}

Expr BitwiseNotCallImpl(common::UnknownArch, const Target& target, Expr a) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_not.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseNotCallImpl(common::X86Arch, const Target& target, Expr a) {
  return lang::CallExtern("bitwise_not", {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCallImpl(common::ARMArch, const Target& target, Expr a) {
  std::stringstream ss;
  ss << "Unsupport arch: " << target.arch_str() << " for bitwise_not.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseNotCallImpl(common::NVGPUArch, const Target& target, Expr a) {
  auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
  return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCallImpl(common::HygonDCUArchHIP, const Target& target, Expr a) {
  auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
  return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCall(const Target& target, Expr a) {
  return std::visit(
      [&](const auto& arch) { return BitwiseNotCallImpl(arch, target, a); },
      target.arch.variant());
}

Expr operator~(Expr a) {
  PADDLE_ENFORCE_EQ(
      a.type().is_int() || a.type().is_uint(),
      true,
      phi::errors::InvalidArgument("The input's type should be int or uint."));
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  return BitwiseNotCall(target, a);
}

}  // namespace ir
}  // namespace cinn
