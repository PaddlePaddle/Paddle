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

  // relu:
  //  out = f(x) = x > 0 ? x : 0
  //  dx = dout * (out > 0 ? 1 : 0) = dout * (x > 0 ? 1 : 0)
  Insert(type, num_oprands, "relu", "real_max(var@, 0)", {"var@ > 0 ? 1 : 0"});
  // sigmoid:
  //  out = f(x) = 1.0 / (1.0 + exp(-x))
  //  dx = dout * out * (1 - out)
  Insert(type, num_oprands, "sigmoid", "1.0 / (1.0 + real_exp(-var@))", {});
  // tanh:
  //  out = f(x) = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
  //  dx = dout * (1 - out * out)
  Insert(type, num_oprands, "tanh", "2.0 / (1.0 + real_exp(-2.0 * var@)) - 1.0",
         {});
}

void OperationMap::InsertBinaryElementwiseOperations() {
  int type = 0;
  int num_oprands = 2;

  // elementwise_add:
  //  out = x + y
  //  dx = dout * 1
  //  dy = dout * 1
  Insert(type, num_oprands, "elementwise_add", "var@ + var$", {"1", "1"});
  // elementwise_sub:
  //  out = x - y
  //  dx = dout * 1
  //  dy = dout * (-1)
  Insert(type, num_oprands, "elementwise_sub", "var@ - var$", {"1", "-1"});
  // elementwise_mul:
  //  out = x * y
  //  dx = dout * y
  //  dy = dout * x
  Insert(type, num_oprands, "elementwise_mul", "var@ * var$", {"var$", "var@"});
  Insert(type, num_oprands, "elementwise_div", "var@ / var$", {});
  Insert(type, num_oprands, "elementwise_min", "real_min(var@, var$)", {});
  Insert(type, num_oprands, "elementwise_max", "real_max(var@, var$)", {});
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
