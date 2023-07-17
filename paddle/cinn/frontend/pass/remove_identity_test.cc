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

#include "paddle/cinn/frontend/pass/test_helper.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"

namespace cinn::frontend {

TEST(RemoveIdentity, remove_single) {
  //              <x>
  //           /       \
  //     identity   identity
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(identity_1, {0});
  auto reduce_sum_2 = builder.ReduceSum(identity_2, {1});

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id};
  std::vector<std::string> program_passes = {"RemoveIdentity",
                                             "DeadCodeEliminate"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 3);
}

TEST(RemoveIdentity, remove_branch) {
  //              <x>
  //               |
  //            identity
  //           /        \
  //    reduce_sum  reduce_sum
  //          |          |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(identity_1, {0});
  auto reduce_sum_2 = builder.ReduceSum(identity_1, {1});

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id, reduce_sum_2->id};
  std::vector<std::string> program_passes = {"RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 1);
}

TEST(RemoveIdentity, remove_multiple) {
  //         <x>  <y>
  //          |    |
  //     identity  |
  //          |    |
  //     identity  |
  //          |    |
  //     identity  |
  //           \  /
  //           mul
  //            |
  //         <mul_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto y = builder.CreateInput(Float(32), {32, 16}, "y");
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(identity_1);
  auto identity_3 = builder.Identity(identity_2);
  auto mul_1 = builder.Add(identity_3, y);

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data(), y.id().data()};
  std::vector<std::string> output_names = {mul_1->id};
  std::vector<std::string> program_passes = {"RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 3);
}

TEST(RemoveIdentity, cannot_remove_fetch) {
  //         <x>  <y>
  //          |    |
  //        relu   |
  //          |    |
  //     identity  |
  //          |    |
  //     identity  |
  //           \  /
  //           mul
  //            |
  //         <mul_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto y = builder.CreateInput(Float(32), {32, 16}, "y");
  auto relu_1 = builder.Relu(x);
  auto identity_1 = builder.Identity(relu_1);
  auto identity_2 = builder.Identity(identity_1);
  auto mul_1 = builder.Add(identity_2, y);

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data(), y.id().data()};
  std::vector<std::string> output_names = {identity_2->id, mul_1->id};
  std::vector<std::string> program_passes = {"RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 1);
}

}  // namespace cinn::frontend
