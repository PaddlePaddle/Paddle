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

#include "paddle/cinn/hlir/pe/broadcast.h"

#include <algorithm>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::common::make_zero;
using ir::Tensor;
using lang::Compute;

template <typename FuncOp>
Tensor Broadcast(const FuncOp& op,
                 const Tensor& a,
                 const Tensor& b,
                 const std::string& output_name = "",
                 const Expr& axis = Expr(-1)) {
  auto fn = [=](const std::vector<Expr>& indice) {
    return op(a(indice), b(indice));
  };
  Tensor output = Compute(a->shape, fn, output_name);
  return output;
}

#define HLIR_IMP_BC_PE(name__, compute__)                      \
  Tensor name__(const Tensor& A,                               \
                const Tensor& B,                               \
                const std::string& output_name,                \
                const Expr& axis) {                            \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ }; \
    return Broadcast(fn, A, B, output_name, axis);             \
  }

HLIR_IMP_BC_PE(Add, return a + b;);
HLIR_IMP_BC_PE(Subtract, return a - b;);
HLIR_IMP_BC_PE(Multiply, return a * b;);
HLIR_IMP_BC_PE(Divide, return a / b;);
HLIR_IMP_BC_PE(FloorDivide, return lang::FloorDivide(a, b););
HLIR_IMP_BC_PE(Remainder, return lang::Remainder(a, b););
HLIR_IMP_BC_PE(Mod, return lang::Mod(a, b););
HLIR_IMP_BC_PE(Maximum, return ir::Max::Make(a, b););
HLIR_IMP_BC_PE(Minimum, return ir::Min::Make(a, b););
HLIR_IMP_BC_PE(LeftShift, return a << b;);
HLIR_IMP_BC_PE(RightShift, return a >> b;);
HLIR_IMP_BC_PE(LogicalRightShift, return lang::LogicalRightShift(a, b););
HLIR_IMP_BC_PE(LogicalAnd,
               return ir::Cast::Make(Bool(), a) && ir::Cast::Make(Bool(), b););
HLIR_IMP_BC_PE(LogicalOr,
               return ir::Cast::Make(Bool(), a) || ir::Cast::Make(Bool(), b););
HLIR_IMP_BC_PE(
    LogicalXOr,
    return (ir::Cast::Make(Bool(), a) || ir::Cast::Make(Bool(), b)) &&
           !(ir::Cast::Make(Bool(), a) && ir::Cast::Make(Bool(), b)););
HLIR_IMP_BC_PE(BitwiseAnd, return a & b;);
HLIR_IMP_BC_PE(BitwiseOr, return a | b;);
HLIR_IMP_BC_PE(BitwiseXor, return a ^ b;);
HLIR_IMP_BC_PE(Greater, return a > b;);
HLIR_IMP_BC_PE(Less, return a < b;);
HLIR_IMP_BC_PE(Equal, return ir::EQ::Make(a, b););
HLIR_IMP_BC_PE(NotEqual, return ir::NE::Make(a, b););
HLIR_IMP_BC_PE(GreaterEqual, return a >= b;);
HLIR_IMP_BC_PE(LessEqual, return a <= b;);
HLIR_IMP_BC_PE(Pow, return lang::Pow(a, b););

Tensor Atan2(const Tensor& A,
             const Tensor& B,
             const std::string& output_name,
             const Expr& axis) {
  constexpr double PI = 3.14159265358979323846;

  auto fn = [&](const Expr& elem_a, const Expr& elem_b) {
    auto atan = lang::Atan(elem_a / elem_b);
    auto pi = cinn::common::make_const(atan->type(), PI);
    auto half_pi = cinn::common::make_const(atan->type(), PI / 2);
    auto zero = ir::Zero(atan->type());
    return ir::Select::Make(
        ir::EQ::Make(elem_b, zero),
        ir::Select::Make(
            ir::EQ::Make(elem_a, zero),
            zero,
            ir::Select::Make(ir::GT::Make(elem_a, zero), half_pi, -half_pi)),
        ir::Select::Make(
            ir::GT::Make(elem_b, zero),
            atan,
            ir::Select::Make(
                ir::GE::Make(elem_a, zero), atan + pi, atan - pi)));
  };
  return Broadcast(fn, A, B, output_name, axis);
}

Tensor BroadcastTo(const Tensor& A,
                   const std::vector<int>& out_shape,
                   const std::vector<int>& broadcast_axes,
                   const std::string& out_name) {
  auto A_shape = A->shape;
  PADDLE_ENFORCE_EQ(
      A_shape.size(),
      broadcast_axes.size(),
      ::common::errors::InvalidArgument(
          "broadcast_axes's size should be same with the input shape's size"));
  PADDLE_ENFORCE_GE(out_shape.size(),
                    broadcast_axes.size(),
                    ::common::errors::InvalidArgument(
                        "broadcast_axes's size should be less than "
                        "or equal to out_shape's size"));
  auto axes = broadcast_axes;
  for (auto& axis : axes) {
    // if axis < 0, plus out_shape.size
    if (axis < 0) {
      axis = out_shape.size() + axis;
    }
    PADDLE_ENFORCE_LT(axis,
                      out_shape.size(),
                      ::common::errors::InvalidArgument(
                          "axis should be less than out_shape's size"));
  }
  std::sort(axes.begin(), axes.end());

  return Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> broadcast_indice;
        for (int idx = 0; idx < axes.size(); ++idx) {
          int a_shape_i = A_shape[idx].as_int64();
          if (a_shape_i == 1) {
            broadcast_indice.push_back(ir::Expr(0));
          } else if (a_shape_i == out_shape[axes[idx]]) {
            broadcast_indice.push_back(indice[axes[idx]]);
          } else {
            std::stringstream ss;
            ss << "fail to broad cast input shape " << a_shape_i
               << " to output shape " << out_shape[axes[idx]];
            PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
          }
        }
        return A(broadcast_indice);
      },
      out_name);
}

Tensor BroadcastTo(const Tensor& A,
                   const std::vector<ir::Expr>& out_shape,
                   const std::string& out_name) {
  auto A_shape = A->shape;
  PADDLE_ENFORCE_GE(
      out_shape.size(),
      A_shape.size(),
      ::common::errors::InvalidArgument(
          "broadcast_to's out_shape's size should be GreaterEqual "
          "with the input shape's size"));

  return Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> broadcast_indice;
        int out_A_offset = out_shape.size() - A_shape.size();
        for (int idx = out_A_offset; idx < out_shape.size(); ++idx) {
          ir::Expr a_shape_i = A_shape[idx - out_A_offset];
          if (MathEqual(a_shape_i, ir::Expr(1))) {
            broadcast_indice.push_back(ir::Expr(0));
          } else if (MathEqual(a_shape_i, out_shape[idx])) {
            broadcast_indice.push_back(indice[idx]);
          } else {
            broadcast_indice.push_back(indice[idx] % a_shape_i);
          }
        }
        return A(broadcast_indice);
      },
      out_name);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
