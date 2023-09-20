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
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/pass/pass_test_helper.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn::frontend {

TEST(TransposeFoldingOutput, BatchedMatmulTransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.Transpose(d, {0, 2, 1});
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto out = builder.Subtract(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedGemmTransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.Transpose(d, {0, 2, 1});
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedMatmulTransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {3, 7, 6}, "B");
  auto c = builder.Transpose(b, {0, 2, 1});
  auto d = builder.Matmul(a, c);
  auto e = builder.Transpose(d, {0, 2, 1});
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto out = builder.Subtract(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedGemmTransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {3, 7, 6}, "B");
  auto c = builder.Transpose(b, {0, 2, 1});
  auto d = builder.Matmul(a, c);
  auto e = builder.Transpose(d, {0, 2, 1});
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedMatmulTransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 7, 6}, "C");
  auto d = builder.Transpose(c, {0, 2, 1});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto g = builder.Transpose(e, {0, 2, 1});
  auto out = builder.Subtract(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedGemmTransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b = builder.Transpose(a, {0, 2, 1});
  auto c = builder.CreateInput(Float(32), {3, 7, 6}, "C");
  auto d = builder.Transpose(c, {0, 2, 1});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto g = builder.Transpose(e, {0, 2, 1});
  auto out = builder.Add(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedMatmulNoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto c = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto g = builder.Transpose(e, {0, 2, 1});
  auto out = builder.Subtract(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedGemmNoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto c = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {3, 7, 8}, "F");
  auto g = builder.Transpose(e, {0, 2, 1});
  auto out = builder.Add(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingOutput, MatmulTransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.Transpose(d, {1, 0});
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto out = builder.Subtract(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, GemmTransLeft) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {6, 7}, "C");
  auto d = builder.Matmul(b, c);
  auto e = builder.Transpose(d, {1, 0});
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, MatmulTransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {7, 6}, "B");
  auto c = builder.Transpose(b, {1, 0});
  auto d = builder.Matmul(a, c);
  auto e = builder.Transpose(d, {1, 0});
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto out = builder.Subtract(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, GemmTransRight) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b = builder.CreateInput(Float(32), {7, 6}, "B");
  auto c = builder.Transpose(b, {1, 0});
  auto d = builder.Matmul(a, c);
  auto e = builder.Transpose(d, {1, 0});
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto out = builder.Add(e, f);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingOutput, MatmulTransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {7, 6}, "C");
  auto d = builder.Transpose(c, {1, 0});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto g = builder.Transpose(e, {1, 0});
  auto out = builder.Subtract(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

TEST(TransposeFoldingOutput, GemmTransTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {7, 6}, "C");
  auto d = builder.Transpose(c, {1, 0});
  auto e = builder.Matmul(b, d);
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto g = builder.Transpose(e, {1, 0});
  auto out = builder.Add(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, true);
}

TEST(TransposeFoldingOutput, MatmulNoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto c = builder.CreateInput(Float(32), {6, 7}, "C");
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto g = builder.Transpose(e, {1, 0});
  auto out = builder.Subtract(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingOutput, GemmNoTrans) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {8, 6}, "A");
  auto c = builder.CreateInput(Float(32), {6, 7}, "C");
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {7, 8}, "F");
  auto g = builder.Transpose(e, {1, 0});
  auto out = builder.Add(f, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingOutput, BatchedComplex) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.FillConstant<float>({20, 2}, 2.0f, "A");
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
  auto n = builder.Matmul(d, a);
  auto o = builder.BroadcastTo(n, {16, n->shape[0], n->shape[1]}, {1, 2});
  auto p = builder.Subtract(f, o);
  auto q = builder.Transpose(f, {0, 2, 1});
  auto u = builder.Transpose(m, {0, 2, 1});
  auto v = builder.Add(q, u);
  auto w = builder.Matmul(v, p);
  auto i = builder.Transpose(w, {2, 1, 0});
  auto j = builder.FillConstant<float>({2, 2, 16}, 3.14f, "I");
  auto out = builder.Add(i, j);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{d.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 5, passes, 123, false);
}

TEST(TransposeFoldingOutput, Complex) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b = builder.Transpose(a, {1, 0});  // 20 * 2
  auto c = builder.CreateInput(Float(32), {121, 20}, "C");
  auto f = builder.Matmul(c, b);  // 121 * 2
  auto x = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto z = builder.CreateInput(Float(32), {121, 20}, "Z");
  auto l = builder.Transpose(z, {1, 0});  // 20 * 121
  auto y = builder.Matmul(x, l);          // 2 * 121
  auto m = builder.Transpose(y, {1, 0});  // 121 * 2
  auto n = builder.Matmul(z, a, false, true);
  auto p = builder.Subtract(f, n);
  auto q = builder.Transpose(f, {1, 0});
  auto u = builder.Transpose(m, {1, 0});
  auto v = builder.Add(q, u);
  auto w = builder.Matmul(v, p);
  auto i = builder.Transpose(w, {1, 0});
  auto j = builder.FillConstant<float>({2, 2}, 3.14f, "I");
  auto out = builder.Add(i, j);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{c.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput"});
  CompareResult(&program, target, input_ids, {out->id}, 5, passes, 123, false);
}

TEST(TransposeFoldingOutput, MultiTransCaseOne) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {2, 10}, "A");
  auto b = builder.CreateInput(Float(32), {10, 50}, "B");
  auto c = builder.Matmul(a, b);          // 2 * 50
  auto d = builder.Transpose(c, {1, 0});  // 50 * 2
  auto e = builder.CreateInput(Float(32), {50, 2}, "E");
  auto f = builder.Add(d, e);
  auto g = builder.Transpose(f, {1, 0});
  auto h = builder.CreateInput(Float(32), {2, 50}, "H");
  auto out = builder.Add(h, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(
      std::vector<absl::string_view>{a.id(), b.id(), e.id(), h.id()},
      std::back_inserter(input_ids),
      [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingOutput, MultiTransCaseTwo) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {2, 10}, "A");
  auto b = builder.CreateInput(Float(32), {10, 50}, "B");
  auto c = builder.Matmul(a, b);          // 2 * 50
  auto d = builder.Transpose(c, {1, 0});  // 50 * 2
  auto g = builder.Transpose(d, {1, 0});
  auto h = builder.CreateInput(Float(32), {2, 50}, "H");
  auto out = builder.Add(h, g);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), h.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer", "RemoveIdentity"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

}  // namespace cinn::frontend
