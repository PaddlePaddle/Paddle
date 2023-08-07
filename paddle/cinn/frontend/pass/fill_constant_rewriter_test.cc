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

#include "paddle/cinn/frontend/pass/test_helper.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"

namespace cinn::frontend {

TEST(FillConstantRewriter, remove_reshape_single) {
  //              <x>
  //           /       \
  //     identity    reshape
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Identity(x);
  auto reshape_1 = builder.Reshape(x, {32, 16});
  auto reduce_sum_1 = builder.ReduceSum(identity_1, {0});
  auto reduce_sum_2 = builder.ReduceSum(reshape_1, {1});

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id, reduce_sum_2->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 2);
}

TEST(FillConstantRewriter, remove_reshape_with_fill_constant) {
  //  fill_constant({16, 32})   <x>
  //          |                  |
  //     reshape({32, 16}     reshape
  //           \                /
  //             elementwise_add
  //                   |
  //                <add_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto constant_1 = builder.FillConstant<float>(
      {16, 32}, static_cast<float>(1.0), "constant_1");
  auto reshape_1 = builder.Reshape(constant_1, {32, 16});
  auto reshape_2 = builder.Reshape(x, {32, 16});
  auto add_1 = builder.Add(reshape_1, reshape_2);

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {add_1->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 2);
}

TEST(FillConstantRewriter, remove_scale_single) {
  //              <x>
  //           /       \
  //     identity    scale
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1 = builder.Identity(x);
  auto scale_1 = builder.Scale(x, 1.0f);
  auto reduce_sum_1 = builder.ReduceSum(identity_1, {0});
  auto reduce_sum_2 = builder.ReduceSum(scale_1, {1});

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id, reduce_sum_2->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 2);
}

TEST(FillConstantRewriter, remove_scale_with_fill_constant) {
  //  fill_constant({16, 32})   <x>
  //          |                  |
  //        scale              scale
  //           \                /
  //             elementwise_add
  //                   |
  //                <add_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto constant_1 = builder.FillConstant<float>({32, 16}, 128.0f, "constant_1");
  auto scale_1 = builder.Scale(constant_1, -1.0f);
  auto scale_2 = builder.Scale(x, 1.0f);
  auto add_1 = builder.Add(scale_1, scale_2);

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {add_1->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 2);
}

TEST(FillConstantRewriter, remove_multi_scale_with_fill_constant) {
  //  fill_constant({16, 32})  x
  //          |                |
  //        scale             scale
  //          |                |
  //        scale              |
  //          |                |
  //        scale              |
  //          \                /
  //           elementwise_add
  //                   |
  //                <add_1>
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {32, 16}, "x");
  auto constant_1 = builder.FillConstant<float>({32, 16}, 128.0f, "constant_1");
  auto scale_1 = builder.Scale(constant_1, -1.0f);
  auto scale_2 = builder.Scale(scale_1, 2.0f, 10.0f);
  auto scale_3 = builder.Scale(scale_2, 3.0f, 1.0f, false);

  auto x_1 = builder.Scale(x, 1.0f);
  auto add_1 = builder.Add(scale_3, x_1);

  PassTest tester;
  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {add_1->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 4);
}

TEST(FillConstantRewriter, two_fill_constant) {
  //  fill_constant({16, 32})  fill_constant({16, 32})
  NetBuilder builder("net_builder");
  auto constant_1 = builder.FillConstant<float>({32, 16}, 128.0f, "constant_1");
  auto constant_2 =
      builder.FillConstant<float>({32, 16}, -128.0f, "constant_2");

  PassTest tester;
  std::vector<std::string> input_names = {};
  std::vector<std::string> output_names = {constant_1->id, constant_2->id};
  std::vector<std::string> program_passes = {"FillConstantRewriter",
                                             "RemoveIdentity"};
  int num_removed_ops =
      tester.RunAndCheck(&builder, program_passes, input_names, output_names);
  ASSERT_EQ(num_removed_ops, 0);
}

}  // namespace cinn::frontend
