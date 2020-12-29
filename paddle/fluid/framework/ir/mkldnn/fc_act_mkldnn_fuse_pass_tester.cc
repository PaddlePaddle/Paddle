// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <exception>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/mkldnn/fc_act_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace ir {

// -------------------------- helper functions --------------------------------
namespace {

using InOutVarNamePair = std::pair<std::string, std::string>;
using OpTypeCountPair = std::pair<std::string, int>;

///
/// @brief      Creates the specified operator and sets up its inputs/outputs.
///
/// @param      prog          The program descriptor to which we add new op.
/// @param[in]  op_type_name  The operator type name.
/// @param[in]  inputs        The vector of input pairs: {input_name, variable
///                           name}
/// @param[in]  outputs       The vector of output pairs {output_name, variable}
/// @param[in]  use_mkldnn    The flag deciding whether or not to set
///                           'use_mkldnn' attribute.
///
/// @return     Returns pointer to the created operator descriptor.
///
OpDesc* CreateOp(ProgramDesc* prog, const std::string& op_type_name,
                 const std::vector<InOutVarNamePair>& inputs,
                 const std::vector<InOutVarNamePair>& outputs,
                 bool use_mkldnn = true) {
  auto op = prog->MutableBlock(0)->AppendOp();
  op->SetType(op_type_name);
  op->SetAttr("use_mkldnn", use_mkldnn);

  for (const auto& input : inputs) {
    op->SetInput(input.first, {input.second});
  }
  for (const auto& output : outputs) {
    op->SetOutput(output.first, {output.second});
  }

  return op;
}

///
/// @brief      Check whether node 'to' is reachable from node 'from' in graph.
///
/// @param[in]  graph  The graph we're checking for reachability.
/// @param[in]  from   The 'from' node name.
/// @param[in]  to     The 'to' node name.
///
/// @return     True if there is connection between nodes 'from' and 'to'.
///
bool TestIsReachable(const Graph& graph, std::string from, std::string to) {
  auto hash = [](const Node* node) -> std::string {
    return node->Name() + std::to_string(node->id());
  };

  auto find_node = [&](const Graph& graph, const std::string& name) -> Node* {
    for (auto& node : GraphTraits::DFS(graph)) {
      if (name == hash(&node)) {
        return &node;
      }
    }

    return nullptr;
  };

  if (from == to) return true;

  std::map<std::string, bool> visited;
  // update the from and to strings to hashed equivs in loop from graph traits
  for (auto& node : GraphTraits::DFS(graph)) {
    auto hashed = hash(&node);
    if (node.Name() == from) {
      from = hashed;
    }
    if (node.Name() == to) {
      to = hashed;
    }
    visited[hashed] = false;
  }

  visited[from] = true;

  std::list<std::string> queue;
  queue.push_back(from);

  while (!queue.empty()) {
    auto cur = find_node(graph, queue.front());
    queue.pop_front();
    if (cur == nullptr) {
      return false;
    }

    for (auto n : cur->outputs) {
      auto hashed_name = hash(n);
      if (hashed_name == to) {
        return true;
      }

      if (!visited[hashed_name]) {
        visited[hashed_name] = true;
        queue.push_back(hashed_name);
      }
    }
  }
  return false;
}

///
/// @brief      Search through graph and counts provided operator occurences.
///
/// @param[in]  graph          The graph we search through.
/// @param[in]  op_type_count  The vector of pairs {op_type_name, op count}
///
/// @note       After going through all graph nodes this function asserts
///             whether counted number for each requested op is as expected.
///
void AssertOpsCount(const Graph& graph,
                    std::vector<OpTypeCountPair> op_type_count) {
  for (auto* node : graph.Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    const std::string op_type_name = node->Op()->Type();
    auto op_it =
        std::find_if(std::begin(op_type_count), std::end(op_type_count),
                     [op_type_name](const OpTypeCountPair& p) {
                       return op_type_name == p.first;
                     });
    if (op_it != std::end(op_type_count)) {
      op_it->second--;
    }
  }

  for (const OpTypeCountPair& p : op_type_count) {
    EXPECT_EQ(p.second, 0);
  }
}

///
/// @brief      Builds a program descriptor.
///
/// @param[in]  transient_vars   The vector of transient variables names.
/// @param[in]  persistent_vars  The vector of persistent variables names. Those
///                              will have persistable attribute set to true.
///
/// @return     The program descriptor object.
///
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
    auto* var = add_var_to_prog(v);
    var->SetPersistable(true);
  }

  return prog;
}

///
/// @brief      Execute pass on provided graph and perform checks.
///
/// @param      graph                The graph we run pass on.
/// @param[in]  from                 The name of a 'starting' node sequence in a
///                                  graph. This would be used to test for
///                                  correct node connections.
/// @param[in]  to                   The name of a 'ending' node sequence in a
///                                  graph. This would be used to test for
///                                  correct node connections.
/// @param[in]  removed_nodes_count  The number of nodes we expect will be
///                                  removed/fused after pass execution.
/// @param[in]  added_nodes_count    The number of nodes we expect will be
///                                  added after pass execution.
///
void RunPassAndAssert(Graph* graph, const std::string& from,
                      const std::string& to, int removed_nodes_count,
                      int added_nodes_count = 0) {
  EXPECT_TRUE(TestIsReachable(*graph, from, to));
  int original_nodes_num = graph->Nodes().size();
  auto pass = PassRegistry::Instance().Get("fc_act_mkldnn_fuse_pass");
  pass->Apply(graph);
  int current_nodes_num = graph->Nodes().size();

  EXPECT_TRUE(TestIsReachable(*graph, from, to));
  EXPECT_EQ(original_nodes_num - removed_nodes_count + added_nodes_count,
            current_nodes_num);
}

}  // namespace

// ------------------------------ Test cases -----------------------------------

TEST(FuseFCActOneDNNPass, ThrowUseMkldnn) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}}, false);
  CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseFCActOneDNNPass, FuseWithGeluTanh) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}});
  auto* act_op =
      CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);
  act_op->SetAttr("approximate", true);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count);
  AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}});

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_TRUE(act_type.compare("gelu_tanh") == 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithGeluErf) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}});
  auto* act_op =
      CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);
  act_op->SetAttr("approximate", false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count);
  AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}});

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_TRUE(act_type.compare("gelu_erf") == 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithGeluAuto) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}});
  CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count);
  AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}});

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_TRUE(act_type.compare("gelu") == 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithTanh) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}});
  CreateOp(&prog, "tanh", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count);
  AssertOpsCount(graph, {{"fc", 1}, {"tanh", 0}});

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_TRUE(act_type.compare("tanh") == 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithSigmoid) {
  auto prog = BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  CreateOp(&prog, "fc",
           {
               {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
           },
           {{"Out", "fc_y"}});
  CreateOp(&prog, "sigmoid", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  RunPassAndAssert(&graph, "x", "act_y", removed_nodes_count);
  AssertOpsCount(graph, {{"fc", 1}, {"sigmoid", 0}});

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_TRUE(act_type.compare("sigmoid") == 0);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_act_mkldnn_fuse_pass);
