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

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

OperationMap* OperationMap::map = nullptr;

OperationMap::OperationMap() {
  InsertUnaryElementwiseOperations();
  InsertBinaryElementwiseOperations();
}

std::unordered_set<std::string> OperationMap::Find(int type, int num_operands) {
  std::unordered_set<std::string> res;
  for (auto& t : operations_) {
    if ((t.second.type == type) &&
        (num_operands < 0 || t.second.num_operands == num_operands)) {
      res.insert(t.first);
    }
  }
  return res;
}

void OperationMap::Insert(int type, int num_operands, std::string op_type,
                          std::string expr,
                          std::vector<std::string> grad_exprs) {
  Operation op(type, num_operands, op_type, {expr});
  PADDLE_ENFORCE_EQ(op.IsValid(), true, "Operation %s is invalid.", op_type);
  operations_[op_type] = op;

  if (grad_exprs.size() > 0U) {
    std::string grad_op_type = op_type + "_grad";
    Operation grad_op(type, num_operands, grad_op_type, grad_exprs);
    PADDLE_ENFORCE_EQ(grad_op.IsValid(), true, "Operation %s is invalid.",
                      grad_op_type);
    operations_[grad_op_type] = grad_op;
  }
}

void OperationMap::InsertUnaryElementwiseOperations() {
  int type = 0;
  int num_oprands = 1;
  // For unary elementwise operations:
  //  ${0} - x
  //  ${1} - out
  //  ${2} - dout

  // relu:
  //  out = f(x) = x > 0 ? x : 0
  //  dx = dout * (out > 0 ? 1 : 0) = dout * (x > 0 ? 1 : 0)
  Insert(type, num_oprands, "relu", "real_max(${0}, 0)",
         {"${0} > 0 ? ${2} : 0"});
  // sigmoid:
  //  out = f(x) = 1.0 / (1.0 + exp(-x))
  //  dx = dout * out * (1 - out)
  Insert(type, num_oprands, "sigmoid", "1.0 / (1.0 + real_exp(- ${0}))",
         {"${2} * ${1} * (1.0 - ${1})"});
  // tanh:
  //  out = f(x) = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
  //  dx = dout * (1 - out * out)
  Insert(type, num_oprands, "tanh", "2.0 / (1.0 + real_exp(-2.0 * ${0})) - 1.0",
         {"${2} * (1.0 - ${1} * ${1})"});
}

void OperationMap::InsertBinaryElementwiseOperations() {
  int type = 0;
  int num_oprands = 2;
  // For binary elementwise oprations:
  //  ${0} - x
  //  ${1} - y
  //  ${2} - out
  //  ${3} - dout

  // elementwise_add:
  //  out = x + y
  //  dx = dout * 1
  //  dy = dout * 1
  Insert(type, num_oprands, "elementwise_add", "${0} + ${1}", {"${3}", "${3}"});
  // elementwise_sub:
  //  out = x - y
  //  dx = dout * 1
  //  dy = dout * (-1)
  Insert(type, num_oprands, "elementwise_sub", "${0} - ${1}",
         {"${3}", "- ${3}"});
  // elementwise_mul:
  //  out = x * y
  //  dx = dout * y
  //  dy = dout * x
  Insert(type, num_oprands, "elementwise_mul", "${0} * ${1}",
         {"${3} * ${1}", "${3} * ${0}"});
  Insert(type, num_oprands, "elementwise_div", "${0} / ${1}", {});
  Insert(type, num_oprands, "elementwise_min", "real_min(${0}, ${1})", {});
  Insert(type, num_oprands, "elementwise_max", "real_max(${0}, ${1})", {});
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
