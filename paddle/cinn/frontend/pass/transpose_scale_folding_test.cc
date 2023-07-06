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

TEST(ScaleFolding, FoldIntoDotBatchedCase1) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_x = builder.Scale(x);
  auto out = builder.Matmul(scale_x, y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, false);
}

TEST(ScaleFolding, FoldIntoDotBatchedCase2) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_x = builder.Scale(x, 2.0f);
  auto out = builder.Matmul(scale_x, y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, false);
}

TEST(ScaleFolding, FoldIntoDotBatchedCase3) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_x = builder.Scale(x, 2.0f, 1.0f);
  auto out = builder.Matmul(scale_x, y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, false);
}

TEST(ScaleFolding, FoldIntoDotBatchedCase4) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_y = builder.Scale(y, 2.0f);
  auto out = builder.Matmul(x, scale_y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, false);
}

TEST(ScaleFolding, FoldIntoDotBatchedCase5) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_x = builder.Scale(x, 2.0f);
  auto scale_y = builder.Scale(y, 2.0f);
  auto out = builder.Matmul(scale_x, scale_y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, false);
}

TEST(ScaleFolding, FoldIntoDotBatchedCase6) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto scale_x = builder.Scale(x, 2.0f);
  auto scale_y = builder.Scale(y, 2.0f);
  auto orig_out = builder.Matmul(scale_x, scale_y);
  auto out = builder.Scale(orig_out, 2.0f);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase1) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto scale_x = builder.Scale(transpose_x, 2.0f);
  auto transpose_y = builder.Transpose(y, {0, 2, 1});
  auto scale_y = builder.Scale(transpose_y, 2.0f);
  auto orig_out = builder.Matmul(scale_x, scale_y);
  auto scale_out = builder.Scale(orig_out, 2.0f);
  auto out = builder.Transpose(scale_out, {0, 2, 1});
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 6, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase2) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto scale_x = builder.Scale(x, 2.0f);
  auto transpose_x = builder.Transpose(scale_x, {0, 2, 1});
  auto scale_y = builder.Scale(y, 2.0f);
  auto transpose_y = builder.Transpose(scale_y, {0, 2, 1});
  auto orig_out = builder.Matmul(transpose_x, transpose_y);
  auto transpose_out = builder.Transpose(orig_out, {0, 2, 1});
  auto out = builder.Scale(transpose_out, 2.0f);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 6, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase3) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto scale_y = builder.Scale(y, 2.0f);
  auto out = builder.Matmul(transpose_x, scale_y);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase4) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto scale_x = builder.Scale(x, 2.0f);
  auto out = builder.Matmul(transpose_x, scale_x);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase5) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto z = builder.FillConstant({4, 3, 6}, 1.0f, "Z");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto scale_y = builder.Scale(y, 2.0f);
  auto out_matmul = builder.Matmul(transpose_x, scale_y);
  auto transpose_o = builder.Transpose(out_matmul, {0, 2, 1});
  auto out = builder.Matmul(transpose_o, z);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, false);
}

TEST(TransposeScaleFolding, BatchComplexCase6) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {20, 3}, "X");
  auto reshape_x = builder.Reshape(x, {4, 5, 3});
  auto scale_x = builder.Scale(reshape_x, 2.0f);
  auto transpose_x = builder.Transpose(scale_x, {0, 2, 1});
  auto out_matmul = builder.Matmul(scale_x, transpose_x);
  auto out = builder.Transpose(out_matmul, {0, 2, 1});
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 3, passes, 123, false);
}

TEST(TransposeBroadCastFolding, BatchComplexCase1) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {5, 6}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto scale_y = builder.Scale(y, 2.0f);
  auto broadcast_y = builder.BroadcastTo(scale_y, {4, 5, 6});
  auto out_matmul = builder.Matmul(transpose_x, broadcast_y);
  auto out_trans = builder.Transpose(out_matmul, {0, 2, 1});
  auto out = builder.Scale(out_trans, 2.0f);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 5, passes, 123, false);
}

TEST(TransposeBroadCastFolding, BatchComplexCase2) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {5, 6}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto cast_x = builder.Cast(transpose_x, "float32");
  auto scale_y = builder.Scale(y, 2.0f);
  auto broadcast_y = builder.BroadcastTo(scale_y, {4, 5, 6});
  auto out_matmul = builder.Matmul(cast_x, broadcast_y);
  auto out_cast = builder.Cast(out_matmul, "float32");
  auto out_trans = builder.Transpose(out_cast, {0, 2, 1});
  auto out = builder.Scale(out_trans, 2.0f);
  auto program = builder.Build();

  common::Target target = common::DefaultTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  auto passes =
      std::make_pair(std::vector<std::string>{"Decomposer"},
                     std::vector<std::string>{"TransposeFoldingInput",
                                              "GemmRewriter",
                                              "TransposeFoldingOutput",
                                              "GemmRewriter"});
  CompareResult(&program, target, input_ids, {out->id}, 5, passes, 123, false);
}

}  // namespace cinn::frontend
