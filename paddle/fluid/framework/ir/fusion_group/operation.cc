/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/operation.h"

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

OperationMap* OperationMap::map = nullptr;

OperationMap::OperationMap() {
  InsertUnaryElementwiseOperations();
  InsertBinaryElementwiseOperations();
  InsertMultivariateElementwiseOperations();
}

std::unordered_set<std::string> OperationMap::Find(int type) {
  std::unordered_set<std::string> res;
  for (auto& t : operations_) {
    if (t.second.type == type) {
      res.insert(t.first);
    }
  }
  return res;
}

void OperationMap::Insert(int type,
                          int num_operands,
                          std::string op_type,
                          std::string expr,
                          std::vector<std::string> grad_exprs,
                          std::vector<std::string> input_names,
                          std::vector<std::string> output_names) {
  Operation op(type, num_operands, op_type, {expr}, input_names, output_names);
  PADDLE_ENFORCE_EQ(op.IsValid(),
                    true,
                    platform::errors::InvalidArgument(
                        "Operation %s is invalid. Please set correct "
                        "expression for forward calculation.",
                        op_type));
  operations_[op_type] = op;

  if (grad_exprs.size() > 0U) {
    std::string grad_op_type = op_type + "_grad";
    // grad_inputs = inputs + outputs + grad of outputs
    std::vector<std::string> grad_input_names = input_names;

    for (auto name : output_names) {
      grad_input_names.push_back(name);
    }
    for (auto name : output_names) {
      grad_input_names.push_back(GradVarName(name));
    }
    // grad_output = grad of inputs
    std::vector<std::string> grad_output_names;
    for (auto name : input_names) {
      grad_output_names.push_back(GradVarName(name));
    }
    Operation grad_op(type,
                      num_operands,
                      grad_op_type,
                      grad_exprs,
                      grad_input_names,
                      grad_output_names);
    PADDLE_ENFORCE_EQ(grad_op.IsValid(),
                      true,
                      platform::errors::InvalidArgument(
                          "Operation %s is invalid. Please set correct "
                          "expression for backward calculation.",
                          grad_op_type));
    operations_[grad_op_type] = grad_op;
  }
}

void OperationMap::InsertUnaryElementwiseOperations() {
  // For unary elementwise operations:
  //  ${0} - x
  //  ${1} - out
  //  ${2} - dout
  auto insert_handler = [&](std::string op_type,
                            std::string expr,
                            std::vector<std::string> grad_exprs) {
    int type = 0;
    int num_oprands = 1;
    Insert(type, num_oprands, op_type, expr, grad_exprs, {"X"}, {"Out"});
  };

  // relu:
  //  out = f(x) = x > 0 ? x : 0
  //  dx = dout * (out > 0 ? 1 : 0)
  insert_handler(
      "relu", "${0} > %{0} ? ${0} : %{0.0}", {"${1} > %{0.0} ? ${2} : %{0.0}"});
  // sigmoid:
  //  out = f(x) = 1.0 / (1.0 + exp(-x))
  //  dx = dout * out * (1 - out)
  insert_handler("sigmoid",
                 "%{1.0} / (%{1.0} + Exp(- ${0}))",
                 {"${2} * ${1} * (%{1.0} - ${1})"});
  // tanh:
  //  out = f(x) = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
  //  dx = dout * (1 - out * out)
  insert_handler("tanh",
                 "%{2.0} / (%{1.0} + Exp(-%{2.0} * ${0})) - %{1.0}",
                 {"${2} * (%{1.0} - ${1} * ${1})"});

  // sqrt:
  //  out = x^(1/2)
  //  dx = dout * 0.5 / out
  insert_handler("sqrt", "Sqrt(${0})", {"${2} * %{0.5} / ${1}"});

  // square:
  //  out = x^2
  //  dx = dout * 2.0 * x
  insert_handler("square", "${0} * ${0}", {"${2} * %{2.0} * ${0}"});

  // assign:
  //  out = x
  insert_handler("assign", "${0}", {});

  // cast:
  //  out = static_cast<T>(x)
  // TODO(wangchaochaohu): This is not the compelete definition of
  //  cast Op, We need refine it later.
  insert_handler("cast", "${0}", {});

  // scale
  //  out = (bias_after_scale) ? scale * X +  bias : scale(X + bias)
  //  here we use '=' operator to separate th default value
  // TODO(wangchaochaohu): Later we need to support phi::DenseTensor input for
  // scale and
  //  bias.
  insert_handler(
      "scale",
      "${bias_after_scale=true} ? (${scale=%{1.0}} * ${0} + "
      "${bias=%{0.0}}) : (${scale=%{1.0}} * (${0} + ${bias=%{0.0}}))",
      {});
}

void OperationMap::InsertBinaryElementwiseOperations() {
  // For binary elementwise oprations:
  //  ${0} - x
  //  ${1} - y
  //  ${2} - out
  //  ${3} - dout
  auto insert_handler = [&](std::string op_type,
                            std::string expr,
                            std::vector<std::string> grad_exprs) {
    int type = 0;
    int num_oprands = 2;
    Insert(type, num_oprands, op_type, expr, grad_exprs, {"X", "Y"}, {"Out"});
  };

  // elementwise_add:
  //  out = x + y
  //  dx = dout * 1
  //  dy = dout * 1
  insert_handler("elementwise_add", "${0} + ${1}", {"${3}", "${3}"});
  // elementwise_sub:
  //  out = x - y
  //  dx = dout * 1
  //  dy = dout * (-1)
  insert_handler("elementwise_sub", "${0} - ${1}", {"${3}", "- ${3}"});
  // elementwise_mul:
  //  out = x * y
  //  dx = dout * y
  //  dy = dout * x
  insert_handler(
      "elementwise_mul", "${0} * ${1}", {"${3} * ${1}", "${3} * ${0}"});
  // elementwise_div:
  //  out = x / y
  //  dx = dout / y
  //  dy = - dout * out / y
  insert_handler("elementwise_div",
                 "${0} / ${1}",
                 {"${3} / ${1}", "- ${3} * ${2} / ${1}"});
  // elementwise_min:
  //  out = x < y ? x : y
  //  dx = dout * (x < y)
  //  dy = dout * (x >= y)
  insert_handler("elementwise_min",
                 "${0} < ${1} ? ${0} : ${1}",
                 {"${3} * (${0} < ${1})", "${3} * (${0} >= ${1})"});
  // elementwise_max:
  //  out = x > y ? x : y
  //  dx = dout * (x > y)
  //  dy = dout * (x <= y)
  insert_handler("elementwise_max",
                 "${0} > ${1} ? ${0} : ${1}",
                 {"${3} * (${0} > ${1})", "${3} * (${0} <= ${1})"});
}

void OperationMap::InsertMultivariateElementwiseOperations() {
  auto insert_handler = [&](std::string op_type,
                            std::string expr,
                            std::vector<std::string> grad_exprs) {
    int type = 0;
    int num_oprands = -1;
    Insert(type, num_oprands, op_type, expr, grad_exprs, {"X"}, {"Out"});
  };

  // sum:
  //  out = x_0 + x_1 + ... + x_N-1
  //
  // For sum with N inputs, the expression inside "[]" will be expanded
  //  N - 1 times. The ${?} represents the number of inputs starting with is 1.
  // For example, sum with 4 inputs, the expanded expression is:
  //  ${0} + ${1} + ${2} + ${3}
  insert_handler("sum", "${0}[ + ${?}]", {});

  auto insert_handler_without_input = [&](std::string op_type,
                                          std::string expr,
                                          std::vector<std::string> grad_exprs) {
    int type = 0;
    int num_oprands = 0;
    Insert(type, num_oprands, op_type, expr, grad_exprs, {}, {"Out"});
  };
  // fill_constant:
  insert_handler_without_input("fill_constant", "${str_value}", {});
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
