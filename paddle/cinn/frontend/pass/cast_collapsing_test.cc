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

#include <cfloat>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/pass/pass_test_helper.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/data_util.h"

namespace cinn::frontend {

TEST(CastCollapsing, FuseTwoCast) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_t = builder.Cast(x, "float16");
  auto out = builder.Cast(x_t, "float32");
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(CastCollapsing, FuseThreeCast) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Cast(x, "int32");
  auto x_2t = builder.Cast(x_1t, "int64");
  auto out = builder.Cast(x_2t, "float32");
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(CastCollapsing, ReplaceUselessCastWithIndentity) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto out = builder.Cast(x, "float32");
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, true);
}

TEST(CastCollapsing, FuseCastToUseless) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Cast(x, "int32");
  auto x_2t = builder.Cast(x_1t, "int64");
  auto x_3t = builder.Cast(x_2t, "float32");
  auto out = builder.Add(x_3t, x_3t);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

TEST(TransposeCollapsing, FuseTransposeWithMultiOutput) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Cast(x, "int32");
  auto x_2t = builder.Cast(x_1t, "float32");
  auto x_3t = builder.Cast(x_2t, "int32");
  auto out1 = builder.Transpose(x_1t, {0, 2, 1});
  auto out2 = builder.Transpose(x_2t, {0, 2, 1});
  auto out3 = builder.Transpose(x_3t, {0, 2, 1});
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program,
                target,
                input_ids,
                {out1->id, out2->id, out3->id},
                1,
                passes,
                123,
                true);
}

TEST(TransposeCollapsing, FuseTwoSecTranspose) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Cast(x, "int32");
  auto x_2t = builder.Cast(x_1t, "float32");
  auto out1 = builder.Reshape(x_2t, {5, 3, 4});
  auto x_3t = builder.Cast(out1, "int32");
  auto x_4t = builder.Cast(x_3t, "float32");
  auto out2 = builder.Transpose(x_2t, {0, 2, 1});
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(
      &program, target, input_ids, {out1->id, out2->id}, 4, passes, 123, true);
}

TEST(TransposeCollapsing, FuseTwoHorizontalTranspose) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y_t1 = builder.Cast(x, "int32");
  auto y_t2 = builder.Cast(x, "int32");
  auto out = builder.Add(y_t1, y_t2);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, true);
}

TEST(TransposeCollapsing, FuseVerAndHorTranspose) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y_t1 = builder.Cast(x, "int32");
  auto y_t2 = builder.Cast(y_t1, "float32");
  auto y_t3 = builder.Cast(x, "float32");
  auto out = builder.Add(y_t2, y_t3);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer"}, {"CastCollapsing"}};
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

}  // namespace cinn::frontend
