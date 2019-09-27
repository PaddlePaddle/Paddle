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

#include <gtest/gtest.h>
#include <string>

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/mkldnn/conv_elementwise_add_mkldnn_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {
constexpr int nodes_removed = 3;
constexpr int nodes_added = 1;

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::pair<std::string, std::string>>& inputs,
           const std::pair<std::string, std::string>& output) {
  auto op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", true);

  for (const auto& input : inputs) {
    op->SetInput(input.first, {input.second});
  }

  op->SetOutput(output.first, {output.second});
}

struct TestIsReachable {
  using func = std::function<bool(const std::string&, const std::string&)>;

  auto operator()(const std::unique_ptr<ir::Graph>& graph) -> func {
    auto hash = [](const Node* node) -> std::string {
      return node->Name() + std::to_string(node->id());
    };

    auto find_node = [&](const std::unique_ptr<ir::Graph>& graph,
                         const std::string& name) -> Node* {
      for (auto& node : GraphTraits::DFS(*graph)) {
        if (name == hash(&node)) {
          return &node;
        }
      }

      return nullptr;
    };

    // update the from and to strings to hashed equivs in loop from graph traits
    return [&](std::string from, std::string to) -> bool {
      if (from == to) return true;

      std::map<std::string, bool> visited;

      for (auto& node : GraphTraits::DFS(*graph)) {
        auto hashed = hash(&node);
        if (node.Name() == from) from = hashed;
        if (node.Name() == to) to = hashed;
        visited[hashed] = false;
      }

      visited[from] = true;

      std::list<std::string> queue;
      queue.push_back(from);

      while (!queue.empty()) {
        auto cur = find_node(graph, queue.front());
        queue.pop_front();
        if (cur == nullptr) return false;

        for (auto n : cur->outputs) {
          auto hashed_name = hash(n);
          if (hashed_name == to) return true;

          if (!visited[hashed_name]) {
            visited[hashed_name] = true;
            queue.push_back(hashed_name);
          }
        }
      }
      return false;
    };
  }
};

void AssertOpsCount(const std::unique_ptr<ir::Graph>& graph,
                    int expected_conv_count,
                    int expected_elementwise_add_count = 0) {
  int conv_count = 0;
  int elementwise_add_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      ++conv_count;
    }
    if (node->IsOp() && node->Op()->Type() == "elementwise_add") {
      ++elementwise_add_count;
    }
  }
  EXPECT_EQ(conv_count, expected_conv_count);
  EXPECT_EQ(elementwise_add_count, expected_elementwise_add_count);
}

ProgramDesc BuildProgramDesc(const std::vector<std::string>& transient_vars,
                             const std::vector<std::string>& persistent_vars) {
  ProgramDesc prog;

  auto add_var_to_prog = [&prog](const std::string& var_name) -> VarDesc* {
    auto var = prog.MutableBlock(0)->Var(var_name);
    var->SetType(proto::VarType::LOD_TENSOR);

    return var;
  };

  for (const auto& v : transient_vars) {
    add_var_to_prog(v);
  }

  for (const auto& v : persistent_vars) {
    auto var = add_var_to_prog(v);
    var->SetPersistable(true);
  }

  return prog;
}

void RunPassAndAssert(ProgramDesc* prog, const std::string& from,
                      const std::string& to, int expected_conv_num) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(*prog));

  TestIsReachable is_reachable;
  EXPECT_TRUE(is_reachable(graph)(from, to));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(is_reachable(graph)(from, to));

  EXPECT_EQ(original_nodes_num - nodes_removed + nodes_added,
            current_nodes_num);

  AssertOpsCount(graph, expected_conv_num);
}
}  // namespace

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsYWithElementwiseAddRelu) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d",
        {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "c"});

  SetOp(&prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  RunPassAndAssert(&prog, "a", "relu", 1);
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionProjectionAsYWithElementwiseAddRelu) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e", "f"},
                               {"bias", "weights", "bias2", "weights2"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  // right branch
  SetOp(&prog, "conv2d",
        {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "c"});

  // left branch
  SetOp(&prog, "conv2d",
        {{"Input", "a"}, {"Bias", "bias2"}, {"Filter", "weights2"}},
        {"Output", "f"});

  SetOp(&prog, "elementwise_add", {{"X", "f"}, {"Y", "c"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  RunPassAndAssert(&prog, "a", "relu", 2);
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsYWithElementwiseAddReluNoBias) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
        {"Output", "c"});
  SetOp(&prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  RunPassAndAssert(&prog, "a", "relu", 1);
}

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsXWithElementwiseAddRelu) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d",
        {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "c"});

  SetOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  RunPassAndAssert(&prog, "a", "relu", 1);
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsXWithElementwiseAddReluNoBias) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
        {"Output", "c"});
  SetOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  RunPassAndAssert(&prog, "a", "relu", 1);
}

TEST(ConvElementwiseAddMKLDNNFusePass, NoFusion) {
  auto prog =
      BuildProgramDesc({"a", "b", "c", "d", "e", "f", "g"}, {"weights"});

  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
        {"Output", "c"});

  SetOp(&prog, "conv2d", {{"Input", "d"}, {"Filter", "weights"}},
        {"Output", "e"});

  SetOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "e"}}, {"Out", "f"});
  SetOp(&prog, "relu", {{"X", "f"}}, {"Out", "g"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  TestIsReachable is_reachable;
  EXPECT_TRUE(is_reachable(graph)("a", "g"));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(is_reachable(graph)("a", "g"));
  EXPECT_EQ(original_nodes_num, current_nodes_num);

  AssertOpsCount(graph, 2, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
