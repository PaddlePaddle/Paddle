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
PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace hlir {
namespace pe {

using cinn::common::make_zero;
using ir::Tensor;
using lang::Compute;

void GetBroadcastShape(const std::vector<Expr>& shape1,
                       const std::vector<Expr>& shape2,
                       std::vector<Expr>* common_shape,
                       std::vector<bool>* broadcast_flag1,
                       std::vector<bool>* broadcast_flag2,
                       int* axis_offset,
                       const Expr& axis) {
  CHECK(common_shape);
  CHECK(broadcast_flag1);
  CHECK(broadcast_flag2);

  std::vector<Expr> shape1_new = shape1;
  std::vector<Expr> shape2_new = shape2;

  if (axis.defined()) {
    int axis_val = axis.as_int32();
    PADDLE_ENFORCE_GE(axis_val,
                      -1,
                      phi::errors::InvalidArgument(
                          "axis should be equal or greater than -1."));
    if (shape1.size() >= shape2.size()) {
      PADDLE_ENFORCE_LE(axis_val,
                        static_cast<int>(shape1.size() - shape2.size()),
                        phi::errors::InvalidArgument(
                            "The axis_val should be less than or equal to "
                            "shape1.size() - shape2.size()."));
      if (axis_val >= 0) {
        *axis_offset = shape1.size() - shape2.size() - axis_val;
        for (int i = 1; i <= *axis_offset; ++i) {
          // specified axis to align, we insert Expr one in tensor B so as to
          // align right with tensor A.
          shape2_new.emplace_back(Expr(1));
          common_shape->insert(common_shape->begin(),
                               shape1[static_cast<int>(shape1.size() - i)]);
          // flag is used to indicate whether to include the indice or not.
          broadcast_flag1->emplace_back(true);
          broadcast_flag2->emplace_back(false);
        }
      }
    } else {
      PADDLE_ENFORCE_LE(axis_val,
                        static_cast<int>(shape2.size() - shape1.size()),
                        phi::errors::InvalidArgument(
                            "The axis_val should be less than or equal to "
                            "shape2.size() - shape1.size()."));
      if (axis_val >= 0) {
        *axis_offset = shape2.size() - shape1.size() - axis_val;
        for (int i = 1; i <= *axis_offset; ++i) {
          // specified axis to align, we insert Expr one in tensor B so as to
          // align right with tensor A.
          shape1_new.emplace_back(Expr(1));
          common_shape->insert(common_shape->begin(),
                               shape2[static_cast<int>(shape2.size() - i)]);
          // flag is used to indicate whether to include the indice or not.
          broadcast_flag2->emplace_back(true);
          broadcast_flag1->emplace_back(false);
        }
      }
    }
  }

  int size1 = shape1_new.size();
  int size2 = shape2_new.size();

  Expr one(1);
  int i;
  i = *axis_offset <= 0 ? 1 : *axis_offset + 1;
  for (; i <= std::min(size1, size2); ++i) {
    // traverse from right to left to get the output shape and broadcast flag
    auto* var1 = shape1_new[size1 - i].As<ir::_Var_>();
    auto* var2 = shape2_new[size2 - i].As<ir::_Var_>();
    if (MathEqual(shape1_new[size1 - i], shape2_new[size2 - i])) {
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      // broadcast flags are recorded in a reverse order
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape1_new[size1 - i])) {
      CHECK(!MathEqual(one, shape2_new[size2 - i]));
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(false);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape2_new[size2 - i])) {
      CHECK(!MathEqual(one, shape1_new[size1 - i]));
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(false);
    } else if (var1 && var2) {
      Expr max_var =
          ir::Max::Make(shape1_new[size1 - i], shape2_new[size2 - i]);
      common_shape->insert(common_shape->begin(), max_var);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var1) {
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var2) {
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else {
      int dim1 = shape1_new[size1 - i].as_int32();
      int dim2 = shape2_new[size2 - i].as_int32();
      if (dim1 == dim2) {
        common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
        // broadcast flags are recorded in a reverse order
        broadcast_flag1->emplace_back(true);
        broadcast_flag2->emplace_back(true);
      } else if (dim1 == 1) {
        common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
        // broadcast flags are recorded in a reverse order
        broadcast_flag1->emplace_back(false);
        broadcast_flag2->emplace_back(true);
      } else if (dim2 == 1) {
        common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
        // broadcast flags are recorded in a reverse order
        broadcast_flag1->emplace_back(true);
        broadcast_flag2->emplace_back(false);
      } else {
        std::stringstream ss;
        ss << "Incompatible broadcast dims " << shape1_new[size1 - i] << " and "
           << shape2_new[size2 - i] << " in: " << shape1_new << " and "
           << shape2_new << std::endl;
        PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
      }
    }
  }
  if (size1 != size2) {
    int max_size = std::max(size1, size2);
    auto& shape = (size1 > size2) ? shape1_new : shape2_new;
    auto var_l = (size1 > size2) ? broadcast_flag1 : broadcast_flag2;
    auto var_s = (size1 > size2) ? broadcast_flag2 : broadcast_flag1;
    for (; i <= max_size; ++i) {
      common_shape->insert(common_shape->begin(), shape[max_size - i]);
      var_l->emplace_back(true);
      var_s->emplace_back(false);
    }
  }
}

void GetBroadcastOutShape(const std::vector<int>& input_shape1,
                          const std::vector<int>& input_shape2,
                          std::vector<int>* common_shape,
                          int axis) {
  std::vector<Expr> shape1;
  std::vector<Expr> shape2;
  auto fn_expr = [](const std::vector<int>& input_shape,
                    std::vector<Expr>* shape) {
    for (int i = 0; i < input_shape.size(); i++) {
      shape->push_back(Expr(input_shape[i]));
    }
  };
  fn_expr(input_shape1, &shape1);
  fn_expr(input_shape2, &shape2);
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;
  int axis_offset = 0;
  std::vector<Expr> out_shape;
  GetBroadcastShape(shape1,
                    shape2,
                    &out_shape,
                    &broadcast_flags1,
                    &broadcast_flags2,
                    &axis_offset,
                    Expr(axis));
  CHECK(common_shape);
  for (auto& shape : out_shape) {
    common_shape->push_back(shape.as_int32());
  }
}

void GetBroadcastIndice(const std::vector<Expr>& indice,
                        const Tensor& tensor_a,
                        const Tensor& tensor_b,
                        int axis_offset,
                        std::vector<Expr>* broadcast_indice1,
                        std::vector<Expr>* broadcast_indice2,
                        const std::vector<bool>& broadcast_flags1,
                        const std::vector<bool>& broadcast_flags2) {
  CHECK(broadcast_indice1);
  CHECK(broadcast_indice2);
  if (broadcast_indice1->empty() && broadcast_indice2->empty()) {
    int flag_size = broadcast_flags1.size();
    int i;
    PADDLE_ENFORCE_GE(
        indice.size(),
        flag_size,
        phi::errors::InvalidArgument(
            "indice size should be greater than or equal to flag size."));
    for (i = 0; i < flag_size; i++) {
      if (broadcast_flags1[flag_size - 1 - i]) {
        // broadcast indices are added from left to right
        broadcast_indice1->push_back(indice[i]);
      } else if (flag_size - i <= tensor_a->shape.size() + axis_offset &&
                 broadcast_indice1->size() < tensor_a->shape.size()) {
        broadcast_indice1->push_back(Expr(0));
      }
      if (broadcast_flags2[flag_size - 1 - i]) {
        broadcast_indice2->push_back(indice[i]);
      } else if (flag_size - i <= tensor_b->shape.size() + axis_offset &&
                 broadcast_indice2->size() < tensor_b->shape.size()) {
        // insert indice 0 when have not yet reached the dimension of tensor.
        // Meanwhile we have to consider the case of axis alignment.
        broadcast_indice2->push_back(Expr(0));
      }
    }
  }
}

template <typename FuncOp>
Tensor Broadcast(const FuncOp& op,
                 const Tensor& a,
                 const Tensor& b,
                 const std::string& output_name = "",
                 const Expr& axis = Expr(-1)) {
  std::vector<Expr> common_shape;
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;

  // the counts of left-shift of tensor b so as to right alignment
  int axis_offset = 0;

  GetBroadcastShape(a->shape,
                    b->shape,
                    &common_shape,
                    &broadcast_flags1,
                    &broadcast_flags2,
                    &axis_offset,
                    axis);

  auto fn = [=](const std::vector<Expr>& indice) {
    std::vector<Expr> broadcast_indice1;
    std::vector<Expr> broadcast_indice2;
    GetBroadcastIndice(indice,
                       a,
                       b,
                       axis_offset,
                       &broadcast_indice1,
                       &broadcast_indice2,
                       broadcast_flags1,
                       broadcast_flags2);
    return op(a(broadcast_indice1), b(broadcast_indice2));
  };
  Tensor output = Compute(common_shape, fn, output_name);
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
      phi::errors::InvalidArgument(
          "broadcast_axes's size should be same with the input shape's size"));
  PADDLE_ENFORCE_GE(
      out_shape.size(),
      broadcast_axes.size(),
      phi::errors::InvalidArgument("broadcast_axes's size should be less than "
                                   "or equal to out_shape's size"));
  auto axes = broadcast_axes;
  for (auto& axis : axes) {
    // if axis < 0, plus out_shape.size
    if (axis < 0) {
      axis = out_shape.size() + axis;
    }
    PADDLE_ENFORCE_LT(axis,
                      out_shape.size(),
                      phi::errors::InvalidArgument(
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
            PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
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
