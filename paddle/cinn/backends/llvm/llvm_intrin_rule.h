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
#include "paddle/common/enforce.h"
namespace cinn {
namespace codegen {

template <int id, int arg_nums, bool add_float_suffix = true>
inline void MakeFloatIntrinOp(lang::Args args, lang::RetValue *rv) {
  PADDLE_ENFORCE_GE(
      args.size(),
      1U,
      ::common::errors::InvalidArgument(
          "The number of arguments should be at least 1. Received: %d",
          args.size()));

  Expr arg = args[0];
  ir::Call *node = arg->as<ir::Call>();

  PADDLE_ENFORCE_NOT_NULL(node,
                          ::common::errors::InvalidArgument(
                              "The argument must be a valid call expression."));

  PADDLE_ENFORCE_GE(
      node->read_args.size(),
      arg_nums,
      ::common::errors::InvalidArgument(
          "The number of read arguments should be at least %d. Received: %d",
          arg_nums,
          node->read_args.size()));

  if (add_float_suffix) {
    PADDLE_ENFORCE_EQ(node->type().is_float(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The node type should be float. Received: %s",
                          node->type().to_string().c_str()));
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
  RegisterBitwise(bitwise_or);
  RegisterBitwise(bitwise_xor);
  RegisterBitwise(bitwise_and);
  RegisterBitwise(left_shift);
  RegisterBitwise(right_shift);
#undef RegisterBitwise

  ir::Registry::Register("lower_cpu_intrinsic_fma", true)
      .SetBody(MakeFloatIntrinOp<::llvm::Intrinsic::fmuladd, 3, false>);

  ir::Registry::Register("lower_cpu_intrinsic_pow", true)
      .SetBody(MakeFloatIntrinOp<::llvm::Intrinsic::pow, 2, false>);

  ir::Registry::Register("lower_cpu_intrinsic_bitwise_not", true)
      .SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isnan", true)
      .SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isfinite", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(
            node,
            ::common::errors::InvalidArgument(
                "The argument must be a valid call expression."));
        PADDLE_ENFORCE_EQ(
            !node->read_args.empty(),
            true,
            ::common::errors::InvalidArgument(
                "The read_args of the node should not be empty."));

        Expr arg = node->read_args[0];
        *rv = !(lang::IsInf(arg)) && !(lang::IsNan(arg));
      });

  ir::Registry::Register("lower_cpu_intrinsic_isinf", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "The provided read_args are empty."));

        Expr arg = node->read_args[0];
        Type type = arg->type();
        if (type.is_int() || type.is_uint()) {
          *rv = cinn::common::make_bool(false, type.lanes());
        } else if (type.is_float()) {
          *rv = ir::EQ::Make(lang::Abs(arg), lang::Infinity(type)) &&
                !(lang::IsNan(arg));
        }
      });

  ir::Registry::Register("lower_cpu_intrinsic_rsqrt", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

        Expr arg = node->read_args[0];
        *rv = make_const(arg->type(), 1) / lang::Sqrt(arg);
      });

  ir::Registry::Register("lower_cpu_intrinsic_exp10", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

        Expr arg = node->read_args[0];
        Expr ln10 = make_const(arg->type(), 2.302585093);
        *rv = lang::Exp(arg * ln10);
      });

  ir::Registry::Register("lower_cpu_intrinsic_tan", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

        Expr arg = node->read_args[0];
        *rv = lang::Sin(arg) / lang::Cos(arg);
      });

  ir::Registry::Register("lower_cpu_intrinsic_tanh", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

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
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

        Expr arg = node->read_args[0];
        *rv = (lang::Exp(arg) + lang::Exp(arg * make_const(arg->type(), -1))) /
              make_const(arg->type(), 2);
      });

  ir::Registry::Register("lower_cpu_intrinsic_sinh", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(!node->read_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The read_args of the node should not be empty. "
                              "Received empty read_args."));

        Expr arg = node->read_args[0];
        *rv = (lang::Exp(arg) - lang::Exp(arg * make_const(arg->type(), -1))) /
              make_const(arg->type(), 2);
      });

  ir::Registry::Register("lower_cpu_intrinsic_mod", true)
      .SetBody([](lang::Args args, lang::RetValue *rv) {
        PADDLE_ENFORCE_GE(args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of args should be greater than 1."));
        Expr arg0 = args[0];
        ir::Call *node = arg0->as<ir::Call>();
        PADDLE_ENFORCE_NOT_NULL(node,
                                ::common::errors::InvalidArgument(
                                    "The argument must be a valid call "
                                    "expression. Received null."));
        PADDLE_ENFORCE_EQ(node->read_args.size(),
                          2UL,
                          ::common::errors::InvalidArgument(
                              "The 'mod' op must have exactly 2 read_args."));

        Expr lhs = node->read_args[0];
        Expr rhs = node->read_args[1];
        *rv = lhs % rhs;
      });
}
}  // namespace codegen
}  // namespace cinn
