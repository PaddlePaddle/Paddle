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

#include "paddle/cinn/hlir/pe/elementwise.h"

#include <algorithm>
#include <string>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace pe {

using ir::Expr;
using ir::Tensor;
using lang::Compute;

#define HLIR_IMP_UNARY_PE(name__)                                  \
  std::vector<ir::Tensor> name__(const Tensor& A,                  \
                                 const std::string& output_name) { \
    return {Compute(                                               \
        A->shape,                                                  \
        [=](const std::vector<Expr>& indice) {                     \
          return lang::name__(A(indice));                          \
        },                                                         \
        output_name)};                                             \
  }

#define HLIR_MKL_IMP_UNARY_PE(name__, ex_name__)                           \
  std::vector<ir::Tensor> name__##MKL(const Tensor& A,                     \
                                      const std::string& output_name) {    \
    CHECK(A->type().is_float())                                            \
        << "type should be float or double but get " << A->type();         \
    std::string fn_name =                                                  \
        "cinn_mkl_" #ex_name__ "_v_fp" + std::to_string(A->type().bits()); \
    auto call = Compute(                                                   \
        {Expr(1)},                                                         \
        [=]() -> Expr { return lang::CallExtern(fn_name, {A}); },          \
        output_name);                                                      \
    auto out = call->TupleGet(0);                                          \
    out->WithBuffer(A->type());                                            \
    return {out, call};                                                    \
  }

HLIR_MKL_IMP_UNARY_PE(Exp, exp);
HLIR_MKL_IMP_UNARY_PE(Erf, erf);
HLIR_MKL_IMP_UNARY_PE(Sqrt, sqrt);
HLIR_MKL_IMP_UNARY_PE(Log, log);
HLIR_MKL_IMP_UNARY_PE(Log2, log2);
HLIR_MKL_IMP_UNARY_PE(Log10, log10);
HLIR_MKL_IMP_UNARY_PE(Floor, floor);
HLIR_MKL_IMP_UNARY_PE(Ceil, ceil);
HLIR_MKL_IMP_UNARY_PE(Round, round);
HLIR_MKL_IMP_UNARY_PE(Tanh, tanh);
HLIR_MKL_IMP_UNARY_PE(Trunc, trunc);
HLIR_MKL_IMP_UNARY_PE(Cos, cos);
HLIR_MKL_IMP_UNARY_PE(Sin, sin);
HLIR_MKL_IMP_UNARY_PE(Cosh, cosh);
HLIR_MKL_IMP_UNARY_PE(Tan, tan);
HLIR_MKL_IMP_UNARY_PE(Sinh, sinh);
HLIR_MKL_IMP_UNARY_PE(Acos, acos);
HLIR_MKL_IMP_UNARY_PE(Acosh, acosh);
HLIR_MKL_IMP_UNARY_PE(Asin, asin);
HLIR_MKL_IMP_UNARY_PE(Asinh, asinh);
HLIR_MKL_IMP_UNARY_PE(Atan, atan);
HLIR_MKL_IMP_UNARY_PE(Atanh, atanh);

HLIR_IMP_UNARY_PE(Exp);
HLIR_IMP_UNARY_PE(Erf);
HLIR_IMP_UNARY_PE(Sqrt);
HLIR_IMP_UNARY_PE(Log);
HLIR_IMP_UNARY_PE(Log2);
HLIR_IMP_UNARY_PE(Log10);
HLIR_IMP_UNARY_PE(Floor);
HLIR_IMP_UNARY_PE(Ceil);
HLIR_IMP_UNARY_PE(Round);
HLIR_IMP_UNARY_PE(Trunc);
HLIR_IMP_UNARY_PE(Cos);
HLIR_IMP_UNARY_PE(Cosh);
HLIR_IMP_UNARY_PE(Tan);
HLIR_IMP_UNARY_PE(Sin);
HLIR_IMP_UNARY_PE(Sinh);
HLIR_IMP_UNARY_PE(Acos);
HLIR_IMP_UNARY_PE(Acosh);
HLIR_IMP_UNARY_PE(Asin);
HLIR_IMP_UNARY_PE(Asinh);
HLIR_IMP_UNARY_PE(Atan);
HLIR_IMP_UNARY_PE(Atanh);
HLIR_IMP_UNARY_PE(IsNan);
HLIR_IMP_UNARY_PE(Tanh);
HLIR_IMP_UNARY_PE(IsFinite);
HLIR_IMP_UNARY_PE(IsInf);

HLIR_IMP_UNARY_PE(Negative);
HLIR_IMP_UNARY_PE(Identity);
HLIR_IMP_UNARY_PE(LogicalNot);
HLIR_IMP_UNARY_PE(BitwiseNot);
HLIR_IMP_UNARY_PE(Sigmoid);
HLIR_IMP_UNARY_PE(Sign);
HLIR_IMP_UNARY_PE(Abs);
HLIR_IMP_UNARY_PE(Rsqrt);
HLIR_IMP_UNARY_PE(Cbrt);
HLIR_IMP_UNARY_PE(Clz);
HLIR_IMP_UNARY_PE(Popc);

ir::Tensor Squeeze(const ir::Tensor& A,
                   const std::vector<int>& axes,
                   const std::string& output_name) {
  std::vector<int> position;
  std::vector<Expr> output_shape;
  if (axes.size()) {
    // if axis < 0, plus tensor rank.
    std::vector<int> naxes;
    for (auto axis : axes) {
      if (axis < 0) {
        axis += A->shape.size();
      }

      naxes.push_back(axis);
    }
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      // if can't find idx in axis
      if (std::find(naxes.begin(), naxes.end(), idx) == naxes.end()) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      } else {
        PADDLE_ENFORCE_EQ(A->shape[idx],
                          Expr(1),
                          phi::errors::InvalidArgument(
                              "The dimension to squeeze must be 1."));
      }
    }
  } else {
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      if (A->shape[idx] != Expr(1)) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      }
    }
  }

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indices) {
        std::vector<Expr> indexs(A->shape.size(), Expr(0));
        for (int idx = 0; idx < indices.size(); ++idx) {
          indexs[position[idx]] = indices[idx];
        }
        return A(indexs);
      },
      output_name);
  return res;
}

ir::Tensor ExpandDims(const ir::Tensor& A,
                      const std::vector<int>& axes,
                      const std::vector<int>& out_shape,
                      const std::string& output_name) {
  const auto& posi_axes = utils::GetPositiveAxes(axes, out_shape.size());

  return Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> idx;
        int axes_pos = 0;
        for (int i = 0; i < indice.size(); ++i) {
          if (axes_pos < posi_axes.size() && posi_axes[axes_pos] == i) {
            ++axes_pos;
          } else {
            idx.push_back(indice[i]);
          }
        }
        PADDLE_ENFORCE_EQ(idx.size(),
                          A->shape.size(),
                          phi::errors::InvalidArgument(
                              "The index size not equal with the input rank."));
        return A(idx);
      },
      UniqName(output_name));
}

ir::Tensor Reshape(const ir::Tensor& A,
                   const std::vector<int>& new_shape,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  const std::vector<Expr>& A_expr_shape = A->shape;
  int input_total_size = 1;
  int output_total_size = 1;
  std::vector<Expr> A_stride_info;
  int stride_base = 1;
  A_stride_info.push_back(Expr(stride_base));

  for (int i = A_expr_shape.size() - 1; i > 0; i--) {
    stride_base *= static_cast<int>(A_expr_shape[i].get_constant());
    A_stride_info.insert(A_stride_info.begin(), Expr(stride_base));
  }

  std::vector<Expr> new_stride_info;
  stride_base = 1;
  new_stride_info.push_back(Expr(stride_base));

  for (int i = new_shape.size() - 1; i > 0; --i) {
    stride_base *= new_shape[i];

    new_stride_info.insert(new_stride_info.begin(), Expr(stride_base));
  }

  for (auto& i : new_shape) {
    output_total_size *= i;
    new_expr_shape.push_back(Expr(i));
  }

  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        Expr offset = indice[0] * new_stride_info[0];
        for (int i = 1; i < indice.size(); i++) {
          offset = offset + indice[i] * new_stride_info[i];
        }
        std::vector<Expr> indice_a;
        for (int i = A_expr_shape.size() - 1; i >= 0; i--) {
          auto inner_offset = offset;
          if (i != (A_expr_shape.size() - 1)) {
            inner_offset = inner_offset / A_stride_info[i];
          }
          auto temp = inner_offset % A_expr_shape[i];
          indice_a.insert(indice_a.begin(), temp);
        }
        return A(indice_a);
      },
      name);
  return res;
}

ir::Tensor Reshape(const ir::Tensor& A,
                   const std::vector<ir::Dim>& new_shape,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  const std::vector<Expr>& A_expr_shape = A->shape;
  Expr input_total_size(1);
  Expr output_total_size(1);

  std::vector<Expr> A_stride_info;
  Expr stride_base(1);
  A_stride_info.push_back(stride_base);
  for (int i = A_expr_shape.size() - 1; i > 0; i--) {
    stride_base = stride_base * A_expr_shape[i];
    A_stride_info.insert(A_stride_info.begin(), Expr(stride_base));
  }

  std::vector<Expr> new_stride_info;
  stride_base = Expr(1);
  new_stride_info.push_back(Expr(stride_base));
  for (int i = new_shape.size() - 1; i > 0; --i) {
    stride_base = stride_base * new_shape[i]->dim_expr;
    new_stride_info.insert(new_stride_info.begin(), Expr(stride_base));
  }

  for (auto& i : new_shape) {
    output_total_size = output_total_size * i->dim_expr;
    new_expr_shape.push_back(i->dim_expr);
  }

  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        Expr offset = indice[0] * new_stride_info[0];
        for (int i = 1; i < indice.size(); i++) {
          offset = offset + indice[i] * new_stride_info[i];
        }
        std::vector<Expr> indice_a;
        for (int i = A_expr_shape.size() - 1; i >= 0; i--) {
          auto inner_offset = offset;
          if (i != (A_expr_shape.size() - 1)) {
            inner_offset = inner_offset / A_stride_info[i];
          }
          auto temp = inner_offset % A_expr_shape[i];
          indice_a.insert(indice_a.begin(), temp);
        }
        return A(indice_a);
      },
      name);
  return res;
}

ir::Tensor Cast(const ir::Tensor& A,
                const Type& dtype,
                const std::string& name) {
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr>& indices) {
        return ir::Cast::Make(dtype, A(indices));
      },
      name);
  return res;
}

ir::Tensor Store(const ir::Tensor& A, const std::string& name) {
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr>& indices) { return A(indices); },
      name);
  return res;
}

ir::Tensor Arange(const float start,
                  const float stop,
                  const float step,
                  const Type& dtype,
                  const std::string& output_name) {
  int num = static_cast<int>(std::ceil((stop - start) / step));
  ir::Tensor res = lang::Compute(
      {Expr(num)},
      [=](const std::vector<ir::Expr>& indices) {
        return ir::Cast::Make(
            dtype,
            Expr(start) +
                Expr(step) * ir::Cast::Make(cinn::common::F32(), indices[0]));
      },
      output_name);
  return res;
}

ir::Tensor Tril(const ir::Tensor& A,
                const int diagonal,
                const std::vector<ir::Dim>& out_shape,
                const std::string& name) {
  ir::Tensor res = Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        PADDLE_ENFORCE_GE(indice.size(),
                          size_t(2),
                          phi::errors::InvalidArgument(
                              "The Tril op input tensor must have a rank "
                              "greater than or equal to 2."));
        std::vector<Expr> new_indice(indice.end() - 2, indice.end());
        Expr col_indice = indice.back();
        return ir::Select::Make(new_indice[0] >= new_indice[1] - diagonal,
                                A(indice),
                                ir::Zero(A->type()));
      },
      name);
  return res;
}

ir::Tensor GenerateShape(const std::vector<ir::Tensor>& inputs,
                         const cinn::dialect::SymbolBindings& symbol_bindings,
                         const std::vector<symbol::DimExpr>& output_dim_exprs,
                         const std::string& name) {
  if (output_dim_exprs.size() != 1) {
    VLOG(4) << "pe::GenerateShape will return a meaningless tensor when "
               "output_dim_exprs.size() != 1";
    return Compute(
        {Expr(1)},
        [=](const std::vector<Expr>& indice) { return Expr(1); },
        name);
  }
  cinn::common::DimExprConverterWithSymbolBindings converter(inputs,
                                                             symbol_bindings);
  auto res = Compute(
      {Expr(1)},
      [=, &converter](const std::vector<Expr>& indice) {
        return converter.ConvertToIrExpr(output_dim_exprs[0]);
      },
      name);
  return res;
}

ir::Tensor IsClose(const ir::Tensor& x,
                   const ir::Tensor& y,
                   int axis,
                   float rtol,
                   float atol,
                   bool equal_nan,
                   const std::string& out_name) {
  // [To do] axis is not used in the op.
  // For each a=x[i], b=y[i]:
  // ```
  // if (isnan(a) || isnan(b)) {
  //   out = equal_nan && isnan(a) == isnan(b);
  // } else {
  //   T left = (a > b ? a - b : b - a);
  //   T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
  //   T diff = (left > right ? left - right : right - left);
  //   out = a == b || left <= right || diff <= 1e-15;
  // }
  // ```
  auto fnop = [&](const Expr& a, const Expr& b) {
    // check whether x or y is nan
    auto check_x_nan = lang::IsNan(a);
    auto check_y_nan = lang::IsNan(b);

    // out = equal_nan && isnan(a) == isnan(b);
    auto check_nan_same =
        Expr(equal_nan) && ir::EQ::Make(check_x_nan, check_y_nan);

    // check whether x and y are close
    // T left = (a > b ? a - b : b - a);
    auto left = ir::Select::Make(a > b, a - b, b - a);
    // T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
    auto right = ir::Cast::Make(x->type(), atol) +
                 ir::Select::Make(b > ir::Zero(b->type()),
                                  ir::Cast::Make(x->type(), rtol) * b,
                                  ir::Cast::Make(x->type(), -rtol) * b);
    // T diff = (left > right ? left - right : right - left);
    auto diff = ir::Select::Make(left > right, left - right, right - left);
    // out = a == b || left <= right || diff <= 1e-15;
    auto check_diff = (ir::EQ::Make(a, b) || (left <= right)) ||
                      (diff <= lang::Epsilon(diff->type()));

    return ir::Select::Make(
        check_x_nan || check_y_nan, check_nan_same, check_diff);
  };
  auto fn = [=](const std::vector<Expr>& indice) {
    PADDLE_ENFORCE_EQ(
        indice.size(),
        y->shape.size(),
        phi::errors::InvalidArgument(
            "The indice size should be equal to y's shape size."));
    return fnop(x(indice), y(indice));
  };
  auto res = Compute(x->shape, fn, out_name);
  return res;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
