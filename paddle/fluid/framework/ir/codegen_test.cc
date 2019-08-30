// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/codegen.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/codegen_helper.h"
#ifdef PADDLE_WITH_CUDA
TEST(codegen, cuda) {
  std::vector<int> mul_input{1, 2};
  std::vector<int> add_input{3, 4};
  std::vector<int> sigmod_input{5};
  int mul_out = 3;
  int add_out = 5;
  int sigmod_out = 6;

  std::string op1 = "elementwise_mul";
  std::string op2 = "elementwise_add";
  std::string op3 = "sigmoid";
  paddle::framework::ir::OperationExpression opexp1(mul_input, mul_out, op1);
  paddle::framework::ir::OperationExpression opexp2(add_input, add_out, op2);
  paddle::framework::ir::OperationExpression opexp3(sigmod_input, sigmod_out,
                                                    op3);

  std::vector<paddle::framework::ir::OperationExpression> fused_op = {
      opexp1, opexp2, opexp3};
  paddle::framework::ir::CodeGen codegen;
  std::string result = codegen.GetKernelCode(fused_op);
  std::cout << result << std::endl;
}
#endif
