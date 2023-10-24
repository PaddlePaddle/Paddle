// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/cinn/frontend/pass/pass_test_helper.h"
#include "paddle/cinn/frontend/pass/test_helper.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn::frontend {

TEST(DeadCodeEliminate, remove_single) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  //              <x>
  //           /  | |   \
  //     identity | |  identity
  //             /   \
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto reduce_sum_2 = builder.ReduceSum(x, {0, 1});
  auto program = builder.Build();

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {identity_1->id, reduce_sum_2->id};

  common::Target target = common::DefaultNVGPUTarget();
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"DeadCodeEliminate"}};
  CompareResult(
      &program, target, input_names, output_names, 2, passes, 123, true);
}

TEST(DeadCodeEliminate, remove_multiple) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  //              <x>
  //           /   |   \
  //     identity  |  reduce_sum
  //          \   /     |
  //           mul    <reduce_sum_1>
  //            |
  //         <mul_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Transpose(x, {1, 0});
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto mul_1 = builder.Matmul(x, identity_1);
  auto program = builder.Build();

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id};

  common::Target target = common::DefaultNVGPUTarget();
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"DeadCodeEliminate"}};
  CompareResult(
      &program, target, input_names, output_names, 2, passes, 123, true);
}

}  // namespace cinn::frontend
