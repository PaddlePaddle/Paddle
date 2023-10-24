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

#include "test/cpp/cinn/benchmark/test_elementwise.h"

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn {
namespace tests {

TEST(test_elementwise_add, default_fp32) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> input_types{Float(32), Float(32)};
  std::vector<Type> output_types{Float(32)};
  auto input_tensors = add_tester.CreateInputTensors<float>();
  add_tester.TestOp("elementwise_add_default_fp32",
                    input_tensors,
                    attrs,
                    input_types,
                    output_types);
}

TEST(test_elementwise_add, default_int32) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> input_types{Int(32), Int(32)};
  std::vector<Type> output_types{Int(32)};
  auto input_tensors = add_tester.CreateInputTensors<int>();
  add_tester.TestOp("elementwise_add_default_int32",
                    input_tensors,
                    attrs,
                    input_types,
                    output_types);
  add_tester.Compare<int>();
}

}  // namespace tests
}  // namespace cinn
