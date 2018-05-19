/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/subgraph_splitter.h"

namespace paddle {
namespace inference {
namespace analysis {

const char *SubGraphSplitter::kMarkerAttrName =
    "_sub_graph_splitter_inside_sub_graph";

std::vector<std::vector<Node *>> SubGraphSplitter::operator()() {
  MarkNodesInsideSubGraph();
  return ExtractSubGraphs();
}

// Mark the output variables inside a subgraph with the func.
inline void MarkOutLinksInSubGraph(const Function *func) {
  for (auto *var : func->outlinks) {
    var->attr(SubGraphSplitter::kMarkerAttrName).Bool() = true;
  }
}

void SubGraphSplitter::MarkNodesInsideSubGraph() {
  for (auto &node : GraphTraits<DataFlowGraph>(graph_).nodes()) {
    if (node_inside_subgraph_teller_(&node)) {
      node.attr(kMarkerAttrName).Bool() = true;
      if (node.type() == Node::Type::kFunction) {
        // If a function is inside the sub-graph, mark all the output variables
        // to be inside too, so that two marked functions will be inside a same
        // sub-graph, lets take a example:  A_function->var->B_function, if
        // A_function is marked, var should also be marked, so that B_function
        // will be in the same sub-graph with A_function if B_function is
        // marked.
        MarkOutLinksInSubGraph(static_cast<const Function *>(&node));
      }
    }
  }
}

const char *kUnionFindParent = "_sub_graph_splitter_union_find_parent_";

// Use the Union Find(UF) algorithm to find fully connected sub-graphs, if node
// a's output is node b, that is a and b is in the same sub-graph. The UF
// algorithm will group them to the same cluster.
using node_map_t = std::unordered_map<int, Node *>;
// Find the ancestor id of a node.
int UnionFindGetAncestor(const node_map_t &node_map, size_t id) {
  int tmp = id;
  do {
    tmp = node_map.at(tmp)->attr(kUnionFindParent).Int32();
  } while (node_map.at(tmp)->attr(kUnionFindParent).Int32() != tmp);
  return tmp;
}
// Make this two node share the same ancestor.
// TODO(Superjom) bad performance, make a balanced tree latter.
void UnionFindCombine(const node_map_t &node_map, size_t a, size_t b) {
  int a_ancestor = UnionFindGetAncestor(node_map, a);
  int b_ancestor = UnionFindGetAncestor(node_map, b);
  node_map.at(b_ancestor)->attr(kUnionFindParent).Int32() = a_ancestor;
  node_map.at(a)->attr(kUnionFindParent).Int32() = a_ancestor;
  node_map.at(b)->attr(kUnionFindParent).Int32() = a_ancestor;
}

std::vector<std::vector<Node *>> SubGraphSplitter::ExtractSubGraphs() {
  std::vector<Node *> marked_nodes;
  for (auto &node : GraphTraits<DataFlowGraph>(graph_).nodes()) {
    if (node.attr(kMarkerAttrName).Bool()) {
      marked_nodes.push_back(&node);
    }
  }
  // extract sub-graphs in the marked node set, use Union Find algorithm.
  node_map_t node_map;  // id to ptr
  for (auto *n : marked_nodes) {
    // n's parent == n.id means it is the ancestor
    n->attr(kUnionFindParent).Int32() = n->id();
    node_map[n->id()] = n;
  }
  std::unordered_set<Node *> visited;
  for (auto *n : marked_nodes) {
    for (auto *out : n->outlinks) {
      if (node_map.count(out->id())) {
        UnionFindCombine(node_map, n->id(), out->id());
      }
    }
  }

  std::unordered_map<int /*ancestor*/, std::vector<Node *>> clusters;
  for (auto *n : marked_nodes) {
    if (n->type() == Node::Type::kFunction) {
      clusters[UnionFindGetAncestor(node_map,
                                    n->attr(kUnionFindParent).Int32())]
          .push_back(n);
    }
  }
  std::vector<std::vector<Node *>> result;
  std::for_each(clusters.begin(), clusters.end(),
                [&](const decltype(clusters)::value_type &it) {
                  result.push_back(it.second);
                });

  return result;
}

void SubGraphFuse::operator()() { ReplaceNodesWithSubGraphs(); }

void SubGraphFuse::ReplaceNodesWithSubGraphs() {
  auto subgraphs = SubGraphSplitter(graph_, node_inside_subgraph_teller_)();
  for (auto &subgraph : subgraphs) {
    // replace this sub-graph with the first node. Two steps: 1. Create a Block
    // Node that contains this subgraph 2. Mark the nodes inside the sub-graph
    // as deleted. 3. Replace the deleted node with the new Block Node.
    auto *block_node = graph_->nodes.Create(Node::Type::kFunctionBlock);
    auto io = ExtractInputAndOutputOfSubGraph(subgraph);
    block_node->inlinks = std::move(io.first);
    block_node->outlinks = std::move(io.second);
    for (auto *node : subgraph) {
      // TODO(Superjomn) need a unified mechanism to treat deleted node in each
      // pass.
      node->SetDeleted();
    }

    std::unordered_map<Node *, Node *>
        delelte_node_map;  // deleted node to BlockNode
    for (auto *n : block_node->inlinks) {
      n->inlinks.clear();
    }
    for (auto *n : block_node->outlinks) {
      n->outlinks.clear();
    }
    for (auto *n : block_node->inlinks) {
      n->outlinks.push_back(block_node);
    }
    for (auto *n : block_node->outlinks) {
      n->inlinks.push_back(n);
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
