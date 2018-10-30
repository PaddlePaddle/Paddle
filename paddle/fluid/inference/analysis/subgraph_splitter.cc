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
  for (auto &node : GraphTraits<DataFlowGraph>(*graph_).nodes()) {
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

// This is a simple representation of a graph.
// The BriefNode hold the pointer of the Node.
// This is to avoid changing the original graph
// in the process of trt graph analysis.
struct BriefNode {
  explicit BriefNode(Node *n) { node = n; }
  Node *node;
  std::vector<BriefNode *> inlinks;
  std::vector<BriefNode *> outlinks;
};

// Union two adjacent BriefNode.
// Suppose we have two adjacent nodes src and dst.
// We will perform the following operations:
// 1. add all inputs(except src) of dst to src inlinks.
// 2. add all outputs of dst to src outlinks.
// 3. change all the dst's inputs and outputs
// corresponding inlinks and outlinks to src node.
// 4. delete all dst's inlinks and outlinks.
void UnionContractedNodes(const std::unordered_map<int, BriefNode *> &node_map,
                          int src_id, int dst_id) {
  // merge the two adjacent nodes into one node.
  BriefNode *src_node = node_map.at(src_id);
  BriefNode *dst_node = node_map.at(dst_id);

  std::unordered_set<BriefNode *> inputs(src_node->inlinks.begin(),
                                         src_node->inlinks.end());
  std::unordered_set<BriefNode *> outputs;

  for (auto *n : src_node->outlinks) {
    if (n != dst_node) outputs.insert(n);
  }

  // Add the inlinks and outlinks of dst node to src node.
  std::vector<BriefNode *> dst_in_nodes = dst_node->inlinks;
  for (BriefNode *node : dst_in_nodes) {
    if (node != src_node) {
      inputs.insert(node);
    }
  }

  std::vector<BriefNode *> dst_out_nodes = dst_node->outlinks;
  for (BriefNode *node : dst_out_nodes) {
    outputs.insert(node);
  }

// update the dst and src node's inlinks and outlinks.
#ifdef __clang__
  src_node->inlinks = std::vector<BriefNode *>(inputs.begin(), inputs.end());
  src_node->outlinks = std::vector<BriefNode *>(outputs.begin(), outputs.end());
  dst_node->inlinks.clear();
  dst_node->outlinks.clear();
#else
  src_node->inlinks =
      std::move(std::vector<BriefNode *>(inputs.begin(), inputs.end()));
  src_node->outlinks =
      std::move(std::vector<BriefNode *>(outputs.begin(), outputs.end()));
  dst_node->inlinks.clear();
  dst_node->outlinks.clear();
#endif

  auto inlink_or_outlink_cleaner = [&](std::vector<BriefNode *> &nodes) {
    for (auto *&n : nodes) {
      if (n == src_node || n == dst_node) {
        n = src_node;
      }
    }
  };
  // Change all the dst inputs and outputs corresponding inlink and
  // outlink to the src node.
  for (auto *node : src_node->inlinks) {
    inlink_or_outlink_cleaner(node->outlinks);
  }

  for (auto *node : src_node->outlinks) {
    inlink_or_outlink_cleaner(node->inlinks);
  }
}

// FlexibleDFS
// If reverse is true, do reverse dfs.
// If enter func is not nullptr, calls enter(node) before visiting any children
// of node.
// If leave func not nullptr, calls leave(node) after visiting all parents of
// node.
void FlexibleDFS(const std::vector<BriefNode *> &source, bool reverse,
                 const std::function<bool(const BriefNode *)> &enter,
                 const std::function<bool(const BriefNode *)> &leave) {
  typedef struct {
    const BriefNode *node;
    bool leave;
  } FNode;

  std::vector<FNode> stack;
  for (auto &node : source) {
    stack.push_back(FNode{node, false});
  }
  std::unordered_set<const BriefNode *> visited;
  while (!stack.empty()) {
    auto fnode = stack.back();
    stack.pop_back();

    if (fnode.leave) {
      if (leave && !leave(fnode.node)) return;
    }
    if (visited.count(fnode.node)) continue;
    visited.insert(fnode.node);

    if (enter && !enter(fnode.node)) return;

    if (leave) stack.push_back(FNode{fnode.node, true});
    const std::vector<BriefNode *> iter_nodes =
        reverse == true ? fnode.node->inlinks : fnode.node->outlinks;
    for (const BriefNode *node : iter_nodes) {
      if (!visited.count(node)) {
        stack.push_back(FNode{node, false});
      }
    }
  }
}

std::vector<std::vector<Node *>> SubGraphSplitter::ExtractSubGraphs() {
  // Run the Extract algorithm to find all subgraphs.
  std::vector<Node *> marked_nodes;
  //  We use brief_node_map to represent the original graph in order to avoid
  //  changing the original graph.
  std::unordered_map<int, BriefNode *> brief_node_map;

  for (auto &node : GraphTraits<DataFlowGraph>(*graph_).nodes_in_TS()) {
    brief_node_map[node.id()] = new BriefNode(&node);
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

  // create breif node map
  for (auto &itr : brief_node_map) {
    for (Node *node : itr.second->node->inlinks) {
      itr.second->inlinks.push_back(brief_node_map[node->id()]);
    }

    for (Node *node : itr.second->node->outlinks) {
      itr.second->outlinks.push_back(brief_node_map[node->id()]);
    }
  }

  for (auto &itr : brief_node_map) {
    BriefNode *brief_node = itr.second;

    if (!brief_node->node->attr(kMarkerAttrName).Bool()) {
      VLOG(4) << brief_node->node->id() << " node not a trt candicate.";
      continue;
    }

    //  Our algorithm must guarantee that:
    //  1. The graph is always directed acyclic graph（DAG）.
    //  2. If there is a path in the subgraph from X to Y (X and Y are both
    //  nodes in the subgraph), then all paths from X to Y are in the
    //  subgraph.
    //
    //  In order to achieve the above guarantee.
    //  For adjacent nodes src -> dst.
    //  1. Get all dst input nodes except src.
    //  2. Reverse DFS from those input nodes
    //  3. If there is a path from input nodes to src,
    //  then the src and dst nodes can not be fused into one node,
    //  otherwise it can be done.

    while (true) {
      std::unordered_set<BriefNode *> contract_nodes;
      for (auto *out : brief_node->outlinks) {
        // must be an trt candidate
        if (!out->node->attr(kMarkerAttrName).Bool()) continue;
        // get all dst input nodes except src.
        std::vector<BriefNode *> source_nodes;
        for (auto *n : out->inlinks) {
          if (n != brief_node) {
            source_nodes.push_back(n);
          }
        }

        // Reverse DFS from the source_nodes.
        bool have_excess_path = false;
        FlexibleDFS(source_nodes, true, nullptr,
                    [&have_excess_path, brief_node](const BriefNode *n) {
                      if (n == brief_node) {
                        have_excess_path = true;
                        return false;
                      }
                      return true;
                    });
        if (have_excess_path) continue;
        contract_nodes.insert(out);
      }
      if (contract_nodes.empty()) break;

      for (auto dst_node : contract_nodes) {
        UnionFindCombine(node_map, brief_node->node->id(),
                         dst_node->node->id());
        UnionContractedNodes(brief_node_map, brief_node->node->id(),
                             dst_node->node->id());
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
    if (subgraph.size() <= argument_->Get<int>("minimum_subgraph_size"))
      continue;
    std::unordered_set<Node *> subgraph_uniq(subgraph.begin(), subgraph.end());
    // replace this sub-graph with the first node. Two steps: 1. Create a Block
    // Node that contains this subgraph 2. Mark the nodes inside the sub-graph
    // as deleted. 3. Replace the deleted node with the new Block Node.
    auto *block_node = static_cast<FunctionBlock *>(
        graph_->nodes.Create(Node::Type::kFunctionBlock));
    auto io = ExtractInputAndOutputOfSubGraph(subgraph);
    block_node->inlinks = std::move(io.first);
    block_node->outlinks = std::move(io.second);

    for (auto *node : subgraph) {
      // TODO(Superjomn) need a unified mechanism to treat deleted node in each
      // pass.
      node->SetDeleted();
      block_node->subgraph.push_back(node);
    }

    // Change all the sub-graph's inputs and outputs corresponding inlink and
    // outlink to this sub-graph node.
    auto inlink_or_outlink_cleaner = [&](std::vector<Node *> &nodes) {
      for (auto *&n : nodes) {
        if (subgraph_uniq.count(n)) {
          n = block_node;
        }
      }
      std::unordered_set<Node *> uniq(nodes.begin(), nodes.end());
      nodes.assign(uniq.begin(), uniq.end());
    };
    for (auto *i : block_node->inlinks) {
      inlink_or_outlink_cleaner(i->outlinks);
    }
    for (auto *&o : block_node->outlinks) {
      inlink_or_outlink_cleaner(o->inlinks);
    }
  }
  FilterRedundantOutputOfSubGraph(graph_);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
