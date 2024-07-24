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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {

/**
 * @brief Compute A && B with auto-broadcasting.
 *
 * @param A The first Tensor or Expr
 * @param B The second Tensor or Expr
 * @param axis Tensor B's beginning position of Tensor A. Default is -1(right
 * align) and then axis = rank(X)-rank(Y).
 * @param out_name The name of the output Tensor
 *
 * @return The result Tensor or Expr.
 * @notes Tensor A's shape should no less than Tensor B's.
 * e.g.
 * shape(A) = (2, 3, 4, 5), shape(B) = (4, 5), with axis=-1(default) or axis=2
 * shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
 * shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
 * shape(A) = (2, 3, 4, 5), shape(B) = (2, 1), with axis=0
 */
#define HLIR_DCL_BC_PE(name__)                                       \
  ir::Tensor name__(const ir::Tensor& A,                             \
                    const ir::Tensor& B,                             \
                    const std::string& out_name =                    \
                        cinn::common::UniqName("T_" #name__ "_out"), \
                    const Expr& axis = Expr());

//! Compute A + B with auto-broadcasting.
HLIR_DCL_BC_PE(Add);
//! Compute Atan2 with auto-broadcasting.
HLIR_DCL_BC_PE(Atan2);
//! Compute A - B with auto-broadcasting.
HLIR_DCL_BC_PE(Subtract);
//! Compute A * B with auto-broadcasting.
HLIR_DCL_BC_PE(Multiply);
//! Compute A / B with auto-broadcasting.
HLIR_DCL_BC_PE(Divide);
//! Compute Floor(A / B) with auto-broadcasting.
HLIR_DCL_BC_PE(FloorDivide);
//! Compute A % B with auto-broadcasting.
HLIR_DCL_BC_PE(Mod);
//! Compute A - floor_div(A, B) * B with auto-broadcasting.
HLIR_DCL_BC_PE(Remainder);
//! Compute Maximum(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Maximum);
//! Compute Minimum(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Minimum);
//! Compute A << B with auto-broadcasting.
HLIR_DCL_BC_PE(LeftShift);
//! Compute A >> B with auto-broadcasting.
HLIR_DCL_BC_PE(RightShift);
//! Compute A && B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalAnd);
//! Compute A || B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalOr);
//! Compute A ^ B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalXOr);
//! Compute A & B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseAnd);
//! Compute A | B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseOr);
//! Compute A ^ B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseXor);
//! Compute A > B with auto-broadcasting.
HLIR_DCL_BC_PE(Greater);
//! Compute A < B with auto-broadcasting.
HLIR_DCL_BC_PE(Less);
//! Compute A == B with auto-broadcasting.
HLIR_DCL_BC_PE(Equal);
//! Compute A != B with auto-broadcasting.
HLIR_DCL_BC_PE(NotEqual);
//! Compute A >= B with auto-broadcasting.
HLIR_DCL_BC_PE(GreaterEqual);
//! Compute A <= B with auto-broadcasting.
HLIR_DCL_BC_PE(LessEqual);
//! Compute  (unsigned)A >> B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalRightShift);
//! Compute  pow(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Pow);

ir::Tensor Pow(const ir::Tensor& A,
               const ir::Tensor& B,
               const std::string& output_name,
               const Expr& axis,
               const cinn::common::Target& target);

ir::Tensor BroadcastTo(
    const ir::Tensor& A,
    const std::vector<int>& out_shape,
    const std::vector<int>& broadcast_axes,
    const std::string& out_name = cinn::common::UniqName("T_broadcast_to_out"));

ir::Tensor BroadcastTo(
    const ir::Tensor& A,
    const std::vector<ir::Expr>& out_shape,
    const std::string& out_name = cinn::common::UniqName("T_broadcast_to_out"));

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
