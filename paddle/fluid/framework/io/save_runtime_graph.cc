/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/io/save_runtime_graph.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/phi/common/port.h"

namespace paddle::framework {

void save_string(std::string content,
                 std::string type,
                 std::string saved_path) {
  VLOG(6) << type << " will be saved to " << saved_path;
  MkDirRecursively(DirName(saved_path).c_str());

  std::ofstream fout(saved_path);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      common::errors::Unavailable("Cannot open %s to save ", saved_path));
  fout << content;
  fout.close();
}

void save_graph_compilation_key(int64_t graph_compilation_key,
                                std::string type,
                                std::string saved_path) {
  VLOG(6) << type << " will be saved to " << saved_path;
  MkDirRecursively(DirName(saved_path).c_str());

  std::ofstream fout(saved_path);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      common::errors::Unavailable("Cannot open %s to save ", saved_path));
  fout << std::to_string(graph_compilation_key);
  fout.close();
}

std::string node_format(const ir::Node& node, int number) {
  return "node_" + std::to_string(number) + " : " + "[" + node.Name() + ", " +
         (node.IsOp() ? "op" : "var") + "]";
}

void save_graph(const ir::Graph& graph,
                std::string type,
                std::string saved_path) {
  VLOG(6) << type << " will be saved to " << saved_path;
  MkDirRecursively(DirName(saved_path).c_str());

  std::ofstream fout(saved_path);
  std::stringstream nodes_content;
  std::stringstream edges_content;
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      common::errors::Unavailable("Cannot open %s to save ", saved_path));
  // record all nodes[var, op]
  // format
  int index = 0;
  std::unordered_map<const ir::Node*, std::string> nodes;
  for (const ir::Node* n : graph.Nodes()) {
    index++;
    nodes_content << node_format(*n, index) << "\n";
    nodes[n] = "node_" + std::to_string(index);
  }
  for (const ir::Node* n : graph.Nodes()) {
    std::string node_id = nodes[n];
    for (const ir::Node* out : n->outputs) {
      edges_content << node_id << " -> " << nodes[out] << "\n";
    }
  }

  // recode nodes' edges
  fout << "nodes: \n";
  fout << nodes_content.str();
  fout << "edges: \n";
  fout << edges_content.str();
  fout.close();
}

void save_runtime_cinn_graph(const ir::Graph& graph,
                             int64_t graph_compilation_key,
                             std::string clusters_ops,
                             std::string clusters_inputs,
                             std::string cluster_outputs,
                             std::string cluster_intervals,
                             std::string saved_path) {
  save_string(clusters_ops, "cluster_ops", saved_path + "/cluster_ops.txt");
  save_string(
      clusters_inputs, "cluster_inputs", saved_path + "/cluster_inputs.txt");
  save_string(
      cluster_outputs, "cluster_outputs", saved_path + "/cluster_outputs.txt");
  save_string(cluster_intervals,
              "cluster_intervals",
              saved_path + "/cluster_intervals.txt");
  save_graph_compilation_key(graph_compilation_key,
                             "graph_compilation_key",
                             saved_path + "/graph_compilation_key.txt");
  save_graph(graph, "graph", saved_path + "/subgraph.txt");
}

}  // namespace paddle::framework
