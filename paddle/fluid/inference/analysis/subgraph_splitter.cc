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

const char *SubGraphSplitter::kMarkerAttrName = "sub_graph_splitter_";

std::vector<std::vector<Node *>> SubGraphSplitter::operator()() {
  MarkNodesInsideSubGraph();
  return ExtractSubGraphs();
}

void SubGraphSplitter::MarkNodesInsideSubGraph() {
  auto trait = GraphTraits<DataFlowGraph>(graph_);
  auto nodes = trait.nodes();
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    if (node_inside_subgraph_teller_(&(*it))) {
      NodeAttr &attr = it->NewAttr<NodeAttr>(kMarkerAttrName);
      attr.is_in_subgraph = true;
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
    tmp = node_map.at(tmp)->NewAttr<int>(kUnionFindParent);
  } while (node_map.at(tmp)->NewAttr<int>(kUnionFindParent) != tmp);
  return tmp;
}
// Make this two node share the same ancestor.
// TODO(Superjom) bad performance, make a balanced tree latter.
void UnionFindCombine(const node_map_t &node_map, size_t a, size_t b) {
  int a_ancestor = UnionFindGetAncestor(node_map, a);
  int b_ancestor = UnionFindGetAncestor(node_map, b);
  node_map.at(b_ancestor)->NewAttr<int>(kUnionFindParent) = a_ancestor;
}

std::vector<std::vector<Node *>> SubGraphSplitter::ExtractSubGraphs() {
  std::vector<Node *> marked_nodes;
  auto trait = GraphTraits<DataFlowGraph>(graph_);
  auto nodes = trait.nodes();
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto &attr = it->NewAttr<NodeAttr>(kMarkerAttrName);
    if (attr.is_in_subgraph) {
      marked_nodes.push_back(&(*it));
    }
  }
  // extract sub-graphs in the marked node set, use Union Find algorithm.
  node_map_t node_map;  // id to ptr
  std::unordered_set<Node *> marked;
  for (auto *n : marked_nodes) {
    n->NewAttr<int>(kUnionFindParent) =
        n->id();  // n's parent == n.id means it is the ancestor
    node_map[n->id()] = n;
    marked.insert(n);
  }
  std::unordered_set<Node *> visited;
  for (auto *n : marked_nodes) {
    for (auto *out : n->outlinks) {
      if (marked.count(out)) {
        UnionFindCombine(node_map, n->id(), out->id());
      }
    }
  }

  std::unordered_map<int /*ancestor*/, std::vector<Node *>> clusters;
  for (auto *n : marked_nodes) {
    clusters[n->NewAttr<int>(kUnionFindParent)].push_back(n);
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
