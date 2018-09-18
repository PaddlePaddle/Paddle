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

constexpr int nodes_removed = 3;
constexpr int nodes_added = 1;

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);

  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", true);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetOutput("Output", outputs);
  } else if (type == "elementwise_add") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", outputs);
  } else if (type == "relu" || type == "sigmoid") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", outputs);
  }
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

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionWithElementwiseAddRelu) {
  auto build_program_desc = [&]() -> ProgramDesc {
    ProgramDesc prog;
    for (auto& v :
         std::vector<std::string>({"a", "b", "weights", "c", "d", "e"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v == "weights") {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "conv2d", {"a", "weights"}, {"b"});
    SetOp(&prog, "elementwise_add", {"c", "b"}, {"d"});
    SetOp(&prog, "relu", {"d"}, {"e"});

    return prog;
  };

  auto prog = build_program_desc();
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
  // Assert conv_relu op in newly generated graph
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

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionElementwiseAdd) {
  auto build_program_desc = [&]() -> ProgramDesc {
    ProgramDesc prog;
    for (auto& v : std::vector<std::string>({"a", "b", "weights"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v == "weights" || v == "bias") {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "conv2d", {"a", "weights"}, {"b"});
    SetOp(&prog, "elementwise_add", {"c", "b"}, {"d"});

    return prog;
  };

  auto prog = build_program_desc();
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
  // Assert conv_relu op in newly generated graph
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

TEST(ConvElementwiseAddMKLDNNFusePass, SigmoidConvolutionAddElementwiseRelu) {
  auto build_program_desc = [&]() -> ProgramDesc {
    ProgramDesc prog;
    for (auto& v :
         std::vector<std::string>({"a", "b", "weights", "c", "d", "e", "f"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v.find("weights")) {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "sigmoid", {"a"}, {"b"});
    SetOp(&prog, "conv2d", {"b", "weights"}, {"c"});
    SetOp(&prog, "elementwise_add", {"d", "c"}, {"e"});
    SetOp(&prog, "relu", {"e"}, {"f"});

    return prog;
  };

  auto prog = build_program_desc();
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
  // Assert conv_relu op in newly generated graph
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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
