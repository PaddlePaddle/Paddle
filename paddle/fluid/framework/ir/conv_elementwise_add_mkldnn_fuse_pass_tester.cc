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

#include "paddle/fluid/framework/ir/conv_elementwise_add_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_traits.h"

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

struct IsReachable {
  using func = std::function<bool(const std::string&, const std::string&)>;

  auto operator()(const std::unique_ptr<ir::Graph>& graph) -> func {
    auto find_node = [](const std::unique_ptr<ir::Graph>& graph,
                        const std::string& name) -> Node* {
      for (auto& node : GraphTraits::DFS(*graph)) {
        if (name == node.Name()) {
          return &node;
        }
      }

      return nullptr;
    };

    return [&](std::string from, const std::string to) -> bool {
      if (from == to) return true;

      std::map<std::string, bool> visited;

      for (auto& node : GraphTraits::DFS(*graph)) {
        visited[node.Name()] = false;
      }

      visited[from] = true;

      std::list<std::string> queue;
      queue.push_back(from);

      while (!queue.empty()) {
        auto cur = find_node(graph, queue.front());
        queue.pop_front();

        if (cur == nullptr) return false;

        for (auto n : cur->outputs) {
          if (n->Name() == to) return true;

          if (!visited[n->Name()]) {
            visited[n->Name()] = true;
            queue.push_back(n->Name());
          }
        }
      }
      return false;
    };
  }
};

void AssertOpsCount(const std::unique_ptr<ir::Graph>& graph) {
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
  EXPECT_EQ(conv_count, 1);
  EXPECT_EQ(elementwise_add_count, 0);
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
}  // namespace

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionWithElementwiseAddRelu) {
  auto prog =
      BuildProgramDesc({"a", "b", "c", "d", "e", "f"}, {"bias", "weights"});

  SetOp(&prog, "conv2d",
        {{"Input", "a"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "b"});
  SetOp(&prog, "elementwise_add", {{"X", "b"}, {"Y", "c"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  IsReachable is_reachable;
  EXPECT_TRUE(is_reachable(graph)("a", "relu"));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(is_reachable(graph)("a", "relu"));

  EXPECT_EQ(original_nodes_num - nodes_removed + nodes_added,
            current_nodes_num);

  AssertOpsCount(graph);
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionWithElementwiseAddReluNoBias) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});
  SetOp(&prog, "conv2d", {{"Input", "a"}, {"Filter", "weights"}},
        {"Output", "b"});
  SetOp(&prog, "elementwise_add", {{"X", "b"}, {"Y", "c"}}, {"Out", "d"});
  SetOp(&prog, "relu", {{"X", "d"}}, {"Out", "e"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  IsReachable is_reachable;

  EXPECT_TRUE(is_reachable(graph)("a", "relu"));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(is_reachable(graph)("a", "relu"));

  EXPECT_EQ(original_nodes_num - nodes_removed + nodes_added,
            current_nodes_num);

  AssertOpsCount(graph);
}

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionElementwiseAdd) {
  auto prog = BuildProgramDesc({"a", "b", "c", "d"}, {"bias", "weights"});
  SetOp(&prog, "conv2d",
        {{"Input", "a"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "b"});
  SetOp(&prog, "elementwise_add", {{"X", "b"}, {"Y", "c"}}, {"Out", "d"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  IsReachable is_reachable;
  EXPECT_TRUE(is_reachable(graph)("a", "d"));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_FALSE(is_reachable(graph)("a", "d"));

  EXPECT_EQ(original_nodes_num - nodes_removed + nodes_added,
            current_nodes_num);
  AssertOpsCount(graph);
}

TEST(ConvElementwiseAddMKLDNNFusePass, SigmoidConvolutionAddElementwiseRelu) {
  auto prog =
      BuildProgramDesc({"a", "b", "c", "d", "e", "f"}, {"bias", "weights"});
  SetOp(&prog, "sigmoid", {{"X", "a"}}, {"Out", "b"});
  SetOp(&prog, "conv2d",
        {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
        {"Output", "c"});
  SetOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "d"}}, {"Out", "e"});
  SetOp(&prog, "relu", {{"X", "e"}}, {"Out", "f"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  IsReachable is_reachable;

  EXPECT_TRUE(is_reachable(graph)("a", "f"));

  auto pass =
      PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(is_reachable(graph)("a", "f"));

  EXPECT_EQ(original_nodes_num - nodes_removed + nodes_added,
            current_nodes_num);
  AssertOpsCount(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
