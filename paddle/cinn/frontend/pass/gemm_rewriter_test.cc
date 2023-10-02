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

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/pass/pass_test_helper.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn::frontend {

TEST(GemmRwriter, BatchedTransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.CreateInput(Float(32), {3, 8, 7}, "E");
  auto out = builder.Add(d, e);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), e.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput", "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(GemmRwriter, BatchedTransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {3, 7, 6}, "B");
  auto c = builder.Transpose(b, {0, 2, 1});
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput", "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(GemmRwriter, BatchedTransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 7, 6}, "C");
  auto d = builder.Transpose(c, {0, 2, 1});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(GemmRwriter, BatchedNoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {3, 6, 7}, "B");
  auto e = builder.Matmul(a, b);
  auto f = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, true);
}

TEST(GemmRwriter, TransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.CreateInput(Float(32), {8, 7}, "E");
  auto out = builder.Add(d, e);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), e.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(GemmRwriter, TransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {7, 6}, "B");
  auto c = builder.Transpose(b, {1, 0});
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(GemmRwriter, TransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {7, 6}, "C");
  auto d = builder.Transpose(c, {1, 0});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(GemmRwriter, NoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {6, 7}, "B");
  auto e = builder.Matmul(a, b);
  auto f = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, true);
}

TEST(GemmRwriter, BatchedComplex) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b = builder.FillConstant<float>({16, 2, 20}, 2.0f, "B");
  auto c = builder.Transpose(b, {0, 2, 1});
  auto d = builder.CreateInput(Float(32), {121, 20}, "D");
  auto e = builder.BroadcastTo(d, {16, 121, 20}, {1, 2});
  auto f = builder.Matmul(e, c);
  auto x = builder.FillConstant<float>({16, 2, 20}, 1.0f, "X");
  auto y = builder.Transpose(x, {0, 2, 1});
  auto z = builder.CreateInput(Float(32), {16, 20, 121}, "Z");
  auto l = builder.Transpose(z, {0, 2, 1});
  auto m = builder.Matmul(l, y);
  auto n = builder.Matmul(d, a, false, true);
  auto o = builder.BroadcastTo(n, {16, n->shape[0], n->shape[1]}, {1, 2});
  auto p = builder.Subtract(f, o);
  auto q = builder.Add(f, m);
  auto out = builder.Add(p, q);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{d.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 4, passes, 123, false);
}

TEST(GemmRwriter, Complex) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {121, 20}, "C");
  auto d = builder.Matmul(c, b);
  auto x = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto y = builder.Transpose(x, {1, 0});
  auto z = builder.CreateInput(Float(32), {20, 121}, "Z");
  auto l = builder.Transpose(z, {1, 0});
  auto m = builder.Matmul(l, y);
  auto n = builder.Matmul(c, a, false, true);
  auto p = builder.Subtract(d, n);
  auto q = builder.Add(d, m);
  auto out = builder.Add(p, q);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{c.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes = std::make_pair(
      std::vector<std::string>{"Decomposer", "RemoveIdentity"},
      std::vector<std::string>{"TransposeFoldingInput", "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, false);
}

}  // namespace cinn::frontend
