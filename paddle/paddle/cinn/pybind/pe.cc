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

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"
#include "paddle/cinn/utils/string.h"

namespace py = pybind11;

namespace cinn {
namespace pybind {

using cinn::common::Type;
using lang::Placeholder;
using py::arg;
using utils::GetStreamCnt;
using utils::StringFormat;

void BindPE(py::module* m) {
#define BIND_UNARY(name__, fn__) \
  m->def(#name__,                \
         &hlir::pe::fn__,        \
         py::arg("x"),           \
         py::arg("out") = "T_" #name__ "_out")
  BIND_UNARY(exp, Exp);
  BIND_UNARY(erf, Erf);
  BIND_UNARY(sqrt, Sqrt);
  BIND_UNARY(log, Log);
  BIND_UNARY(log2, Log2);
  BIND_UNARY(log10, Log10);
  BIND_UNARY(floor, Floor);
  BIND_UNARY(ceil, Ceil);
  BIND_UNARY(round, Round);
  BIND_UNARY(trunc, Trunc);
  BIND_UNARY(cos, Cos);
  BIND_UNARY(cosh, Cosh);
  BIND_UNARY(tan, Tan);
  BIND_UNARY(sin, Sin);
  BIND_UNARY(sinh, Sinh);
  BIND_UNARY(acos, Acos);
  BIND_UNARY(acosh, Acosh);
  BIND_UNARY(asin, Asin);
  BIND_UNARY(asinh, Asinh);
  BIND_UNARY(atan, Atan);
  BIND_UNARY(atanh, Atanh);
  BIND_UNARY(isnan, IsNan);
  BIND_UNARY(tanh, Tanh);
  BIND_UNARY(isfinite, IsFinite);
  BIND_UNARY(isinf, IsInf);

  BIND_UNARY(negative, Negative);
  BIND_UNARY(identity, Identity);
  BIND_UNARY(logical_not, LogicalNot);
  BIND_UNARY(bitwise_not, BitwiseNot);
  BIND_UNARY(sigmoid, Sigmoid);
  BIND_UNARY(sign, Sign);
  BIND_UNARY(abs, Abs);
  BIND_UNARY(rsqrt, Rsqrt);

#define BIND_BINARY(name__, fn__) \
  m->def(#name__,                 \
         &hlir::pe::fn__,         \
         py::arg("x"),            \
         py::arg("y"),            \
         py::arg("out"),          \
         py::arg("axis") = Expr(-1))

  BIND_BINARY(add, Add);
  BIND_BINARY(atan2, Atan2);
  BIND_BINARY(subtract, Subtract);
  BIND_BINARY(multiply, Multiply);
  BIND_BINARY(divide, Divide);
  BIND_BINARY(floor_divide, FloorDivide);
  BIND_BINARY(mod, Mod);
  BIND_BINARY(remainder, Remainder);
  BIND_BINARY(max, Maximum);
  BIND_BINARY(min, Minimum);
  BIND_BINARY(left_shift, LeftShift);
  BIND_BINARY(right_shift, RightShift);
  BIND_BINARY(logical_and, LogicalAnd);
  BIND_BINARY(logical_or, LogicalOr);
  BIND_BINARY(logical_xor, LogicalXOr);
  BIND_BINARY(bitwise_and, BitwiseAnd);
  BIND_BINARY(bitwise_or, BitwiseOr);
  BIND_BINARY(bitwise_xor, BitwiseXor);
  BIND_BINARY(greater, Greater);
  BIND_BINARY(less, Less);
  BIND_BINARY(equal, Equal);
  BIND_BINARY(not_equal, NotEqual);
  BIND_BINARY(greater_equal, GreaterEqual);
  BIND_BINARY(less_equal, LessEqual);

#define BIND_REDUCE(name__, fn__)      \
  m->def(#name__,                      \
         &hlir::pe::fn__,              \
         py::arg("x"),                 \
         py::arg("axes"),              \
         py::arg("keep_dims") = false, \
         py::arg("out") = "T_" #name__ "_out")
  BIND_REDUCE(reduce_sum, ReduceSum);
  BIND_REDUCE(reduce_prod, ReduceProd);
  BIND_REDUCE(reduce_max, ReduceMax);
  BIND_REDUCE(reduce_min, ReduceMin);
  BIND_REDUCE(reduce_all, ReduceAll);
  BIND_REDUCE(reduce_any, ReduceAny);

  m->def("matmul",
         &hlir::pe::Matmul,
         py::arg("tensor_a"),
         py::arg("tensor_b"),
         py::arg("trans_a") = false,
         py::arg("trans_b") = false,
         py::arg("alpha") = 1,
         py::arg("out") = "T_Matmul_out");

  m->def("matmul_mkl",
         &hlir::pe::MatmulMKL,
         py::arg("tensor_a"),
         py::arg("tensor_b"),
         py::arg("trans_a") = false,
         py::arg("trans_b") = false,
         py::arg("alpha") = 1,
         py::arg("out") = "T_Matmul_mkl_out",
         py::arg("target") = cinn::common::DefaultHostTarget());
}

}  // namespace pybind
}  // namespace cinn
