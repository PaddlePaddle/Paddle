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

#include <string>
#include <vector>

#include "paddle/cinn/hlir/dialect/operator/ir/symbol_bindings.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn {
namespace hlir {
namespace pe {

/**
 * @brief Unary primitive emitters
 *
 * @param A The input Tensor
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
#define HLIR_DCL_UNARY_PE(name__)                            \
  std::vector<ir::Tensor> name__(                            \
      const ir::Tensor& A,                                   \
      const std::string& output_name = "T_" #name__ "_out"); \
  std::vector<ir::Tensor> name__##MKL(                       \
      const ir::Tensor& A,                                   \
      const std::string& output_name = "T_" #name__ "_mkl_out");

HLIR_DCL_UNARY_PE(Exp);
HLIR_DCL_UNARY_PE(Erf);
HLIR_DCL_UNARY_PE(Sqrt);
HLIR_DCL_UNARY_PE(Log);
HLIR_DCL_UNARY_PE(Log2);
HLIR_DCL_UNARY_PE(Log10);
HLIR_DCL_UNARY_PE(Floor);
HLIR_DCL_UNARY_PE(Ceil);
HLIR_DCL_UNARY_PE(Round);
HLIR_DCL_UNARY_PE(Trunc);
HLIR_DCL_UNARY_PE(Cos);
HLIR_DCL_UNARY_PE(Cosh);
HLIR_DCL_UNARY_PE(Tan);
HLIR_DCL_UNARY_PE(Sin);
HLIR_DCL_UNARY_PE(Sinh);
HLIR_DCL_UNARY_PE(Acos);
HLIR_DCL_UNARY_PE(Acosh);
HLIR_DCL_UNARY_PE(Asin);
HLIR_DCL_UNARY_PE(Asinh);
HLIR_DCL_UNARY_PE(Atan);
HLIR_DCL_UNARY_PE(Atanh);
HLIR_DCL_UNARY_PE(IsNan);
HLIR_DCL_UNARY_PE(Tanh);
HLIR_DCL_UNARY_PE(IsFinite);
HLIR_DCL_UNARY_PE(IsInf);

HLIR_DCL_UNARY_PE(Negative);
HLIR_DCL_UNARY_PE(Identity);
HLIR_DCL_UNARY_PE(LogicalNot);
HLIR_DCL_UNARY_PE(BitwiseNot);
HLIR_DCL_UNARY_PE(Sigmoid);
HLIR_DCL_UNARY_PE(Sign);
HLIR_DCL_UNARY_PE(Abs);
HLIR_DCL_UNARY_PE(Rsqrt);
HLIR_DCL_UNARY_PE(Reinterpret);
HLIR_DCL_UNARY_PE(ElementwiseSum);
HLIR_DCL_UNARY_PE(Full);
HLIR_DCL_UNARY_PE(FullLike);
HLIR_DCL_UNARY_PE(Cbrt);
HLIR_DCL_UNARY_PE(Clz);
HLIR_DCL_UNARY_PE(Popc);

template <typename T>
ir::Tensor AssignValue(
    const std::vector<T>& values,
    const cinn::common::Type& type = cinn::common::type_of<T>(),
    const std::string& output_name = "T_assign_value_out") {
  PADDLE_ENFORCE_EQ(!values.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input of pe::AssignValue should not be empty. "
                        "Please provide valid values."));

  auto out = lang::Compute(
      {ir::Expr(static_cast<int>(values.size()))},
      [=](const std::vector<ir::Expr>& indice) {
        auto init_value = (type == cinn::common::type_of<T>())
                              ? ir::Expr(values[0])
                              : cinn::common::cast(ir::Expr(values[0]), type);
        ir::Expr previous = ir::Select::Make(
            ir::EQ::Make(indice[0], ir::Expr(0)), init_value, lang::Zero(type));

        for (int i = 1; i < values.size(); ++i) {
          auto val = (type == cinn::common::type_of<T>())
                         ? ir::Expr(values[i])
                         : cinn::common::cast(ir::Expr(values[i]), type);
          previous = ir::Select::Make(
              ir::EQ::Make(indice[0], ir::Expr(i)), val, previous);
        }
        return previous;
      },
      output_name);

  return out;
}

ir::Tensor Squeeze(
    const ir::Tensor& A,
    const std::vector<int>& axes = {},
    const std::string& output_name = UniqName("T_Elementwise_Squeeze_out"));

ir::Tensor ExpandDims(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const std::vector<int>& out_shape,
    const std::string& output_name = UniqName("T_Elementwise_ExpandDims_out"));

ir::Tensor Reshape(
    const ir::Tensor& A,
    const std::vector<int>& new_shape,
    const std::string& name = UniqName("T_Elementwise_Reshape_out"));

ir::Tensor Reshape(
    const ir::Tensor& A,
    const std::vector<ir::Dim>& new_shape,
    const std::string& name = UniqName("T_Elementwise_Reshape_out"));

ir::Tensor Cast(const ir::Tensor& A,
                const Type& dtype,
                const std::string& name = UniqName("T_Elementwise_Cast_out"));

ir::Tensor Store(const ir::Tensor& A,
                 const std::string& name = UniqName("T_Elementwise_Store_out"));

ir::Tensor Arange(
    const float start,
    const float stop,
    const float step,
    const Type& dtype,
    const std::string& name = UniqName("T_Elementwise_Arange_out"));

ir::Tensor Tril(const ir::Tensor& A,
                const int diagonal,
                const std::vector<ir::Dim>& out_shape,
                const std::string& name = UniqName("T_Elementwise_Tril_out"));

ir::Tensor GenerateShape(
    const std::vector<ir::Tensor>& inputs,
    const cinn::dialect::SymbolBindings& symbol_bindings,
    const std::vector<symbol::DimExpr>& output_dim_exprs,
    const std::vector<ir::Dim>& out_shape,
    const std::vector<Type>& out_type,
    const std::string& name = UniqName("T_Generate_Shape_out"));

// This operator checks if all x and y satisfy the condition: |x - y| <= atol +
// rtol * |y|
ir::Tensor IsClose(
    const ir::Tensor& x,
    const ir::Tensor& y,
    int axis = -1,
    float rtol = 1e-05f,
    float atol = 1e-08f,
    bool equal_nan = false,
    const std::string& out_name = cinn::common::UniqName("IsClose_output"));

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
