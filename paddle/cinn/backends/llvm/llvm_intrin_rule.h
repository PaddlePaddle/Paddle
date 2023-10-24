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

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <llvm/IR/Intrinsics.h>

#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/registry.h"
#include "paddle/cinn/lang/packed_func.h"

namespace cinn {
namespace codegen {

template <int id, int arg_nums, bool add_float_suffix = true>
inline void MakeFloatIntrinOp(lang::Args args, lang::RetValue *rv) {
  CHECK_GE(args.size(), 1U);
  Expr arg = args[0];
  ir::Call *node = arg->as<ir::Call>();
  CHECK(node);
  CHECK_GE(node->read_args.size(), arg_nums);
  if (add_float_suffix) {
    CHECK(node->type().is_float());
    *rv = ir::intrinsics::BuiltinIntrin::Make(
        node->name + "f", node->read_args, id, arg_nums, node->type());
  } else {
    *rv = ir::intrinsics::BuiltinIntrin::Make(
        node->name, node->read_args, id, arg_nums, node->type());
  }
}

void RegisterCpuIntrinRule() {
#define __(intrin_name__, id)                                         \
  ir::Registry::Register("lower_cpu_intrinsic_" #intrin_name__, true) \
      .SetBody(MakeFloatIntrinOp<id, 1>);
  __(exp, ::llvm::Intrinsic::exp)
  __(exp2, ::llvm::Intrinsic::exp2)
  __(sqrt, ::llvm::Intrinsic::sqrt)
  __(log, ::llvm::Intrinsic::log)
  __(log2, ::llvm::Intrinsic::log2)
  __(log10, ::llvm::Intrinsic::log10)
  __(floor, ::llvm::Intrinsic::floor)
  __(ceil, ::llvm::Intrinsic::ceil)
  __(round, ::llvm::Intrinsic::round)
  __(trunc, ::llvm::Intrinsic::trunc)
  __(cos, ::llvm::Intrinsic::cos)
  __(sin, ::llvm::Intrinsic::sin)
  __(fabs, ::llvm::Intrinsic::fabs)
#undef __

// set id -1 if not llvm intrinsics
#define RegisterBitwise(intrin_name__)                                \
  ir::Registry::Register("lower_cpu_intrinsic_" #intrin_name__, true) \
      .SetBody(MakeFloatIntrinOp<-1, 2, false>);
  RegisterBitwise(bitwise_or) RegisterBitwise(bitwise_xor) RegisterBitwise(
      bitwise_and) RegisterBitwise(left_shift) RegisterBitwise(right_shift)
#undef RegisterBitwise

      ir::Registry::Register("lower_cpu_intrinsic_fma", true)
          .SetBody(MakeFloatIntrinOp<::llvm::Intrinsic::fmuladd, 3, false>);

  ir::Registry::Register("lower_cpu_intrinsic_bitwise_not", true)
      .SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isnan", true)
      .SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isfinite", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        *rv = !(lang::IsInf(arg)) && !(lang::IsNan(arg));
      });

  ir::Registry::Register("lower_cpu_intrinsic_isinf", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        Type type = arg->type();
        if (type.is_int() || type.is_uint()) {
          *rv = common::make_bool(false, type.lanes());
        } else if (type.is_float()) {
          *rv = ir::EQ::Make(lang::Abs(arg), lang::Infinity(type)) &&
                !(lang::IsNan(arg));
        }
      });

  ir::Registry::Register("lower_cpu_intrinsic_rsqrt", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        *rv = make_const(arg->type(), 1) / lang::Sqrt(arg);
      });

  ir::Registry::Register("lower_cpu_intrinsic_exp10", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        Expr ln10 = make_const(arg->type(), 2.302585093);
        *rv = lang::Exp(arg * ln10);
      });

  ir::Registry::Register("lower_cpu_intrinsic_tan", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        *rv = lang::Sin(arg) / lang::Cos(arg);
      });

  ir::Registry::Register("lower_cpu_intrinsic_tanh", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        Expr zero = make_const(arg->type(), 0);
        Expr one = make_const(arg->type(), 1);
        Expr two = make_const(arg->type(), 2);
        Expr neg_two = make_const(arg->type(), -2);

        Expr exp_neg2x = lang::Exp(neg_two * arg);
        Expr exp_pos2x = lang::Exp(two * arg);

        Expr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
        Expr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
        *rv = ir::Select::Make(arg >= zero, tanh_pos, tanh_neg);
      });

  ir::Registry::Register("lower_cpu_intrinsic_cosh", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        *rv = (lang::Exp(arg) + lang::Exp(arg * make_const(arg->type(), -1))) /
              make_const(arg->type(), 2);
      });

  ir::Registry::Register("lower_cpu_intrinsic_sinh", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        CHECK_GE(args.size(), 1U);
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        CHECK(node);
        CHECK(!node->read_args.empty());
        Expr arg = node->read_args[0];
        *rv = (lang::Exp(arg) - lang::Exp(arg * make_const(arg->type(), -1))) /
              make_const(arg->type(), 2);
      });
}
}  // namespace codegen
}  // namespace cinn
