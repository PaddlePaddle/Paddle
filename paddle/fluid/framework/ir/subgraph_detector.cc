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

#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;
static int ii =0;
std::pair<std::vector<Node *>, std::vector<Node *>>
ExtractInputAndOutputOfSubGraph(std::vector<Node *> &graph,
                       paddle::framework::ir::Graph *global_graph) {  // NOLINT
  std::unordered_set<Node *> nodes(graph.begin(), graph.end());
  std::unordered_set<Node *> inputs;
  std::unordered_set<Node *> outputs;
  // Input a Value, check whether its inlink is in the subgraph.
  auto inlink_in_subgraph = [&](Node *n) {
    for (auto *in : n->inputs) {
      if (nodes.count(in)) return true;
    }
    return false;
  };


  framework::BlockDesc *block_cast{nullptr};
  for (auto *op_node : framework::ir::TopologySortOperations(*global_graph)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed") block_cast = op_node->Op()->Block();
  }


  for (auto &node : graph) {
    for (auto *in : node->inputs) {
      // The Value that is written by nodes inside a sub-graph shouldn't be the
      // input of the sub-graph.
      if (!nodes.count(in) && in->IsVar() && !inlink_in_subgraph(in)) {
        inputs.insert(in);
      }
    }
    for (auto *out : node->outputs) {
      if (!nodes.count(out) && out->IsVar()) {
        outputs.insert(out);
      }
    }
  }

  auto copy_inputs = inputs;
  for (auto in_var : copy_inputs) {
    if (in_var->Var()->GetDataType() == framework::proto::VarType::INT64 && 0) {
      std::cout << "哈哈哈" << std::endl;
      // We should place a `cast` op between in_var and `node` op.
      // node muse be within tensorrt sub-graph.
      //        |                      |
      //      in_var                 in_var   int64
      //        |                      |
      //        |        ->        cast_node
      //        |                      |
      //        |            cast_output_node int32
      //        |                      |
      //     other ops            other ops   they must be in sub-graph
      //        |                      |
      framework::OpDesc cast_desc(block_cast);
      cast_desc.SetType("cast");
      cast_desc.SetInput("X", {in_var->Name()});
      std::string cast_out_name = in_var->Name() + "_out321" + std::to_string(ii);
      ii++;
      cast_desc.SetOutput("Out", {cast_out_name});
      cast_desc.SetAttr("in_dtype", 3);
      cast_desc.SetAttr("out_dtype", 2);
      cast_desc.Flush();
      auto *cast_node = global_graph->CreateOpNode(&cast_desc);
      auto *cast_output_vardesc = block_cast->Var(cast_out_name);
      cast_output_vardesc->SetPersistable(false);
      cast_output_vardesc->SetDataType(framework::proto::VarType::INT32);
      cast_output_vardesc->SetShape(in_var->Var()->GetShape());
      auto *cast_output_node = global_graph->CreateVarNode(cast_output_vardesc);

      std::vector<paddle::framework::ir::Node *> new_in_var_outputs;
      for (auto op : in_var->outputs) {
        if (!nodes.count(op))
        new_in_var_outputs.push_back(op);
      }
      new_in_var_outputs.push_back(cast_node);
      
      std::vector<Node*> other_ops;
      for (auto op : in_var->outputs) {
        if(nodes.count(op)) {
          other_ops.push_back(op);
        }
      }

      in_var->outputs = new_in_var_outputs;

      cast_node->inputs = std::vector<paddle::framework::ir::Node *>{in_var};
      cast_node->outputs = std::vector<paddle::framework::ir::Node *>{cast_output_node};
      cast_output_node->inputs = std::vector<paddle::framework::ir::Node *>{cast_node};

      cast_output_node->outputs = other_ops;
      for (auto other_op : other_ops) {
        // in `other_op->Op()`, we need replace `in_var->Name()` with cast_out_name
        for (auto iter : other_op->Op()->Inputs()) {
          auto iter_inputs = iter.second;
          for (auto &var_name : iter_inputs) {
            if (var_name == in_var->Name()) {
              var_name = cast_out_name;
            }
          }
          other_op->Op()->SetInput(iter.first, iter_inputs);
        }
        if (std::find(other_op->inputs.begin(), other_op->inputs.end(), in_var) !=
            other_op->inputs.end()) {
          *(std::find(other_op->inputs.begin(), other_op->inputs.end(), in_var)) =
              cast_output_node;
        }
      }
      inputs.erase(in_var);
      inputs.emplace(cast_output_node);
    }
  }

  return std::make_pair(std::vector<Node *>(inputs.begin(), inputs.end()),
                        std::vector<Node *>(outputs.begin(), outputs.end()));
}

// Filter the Intermediate results of the subgraph node.
void FilterRedundantOutputOfSubGraph(Graph *graph) {
  std::vector<Node *> op_nodes;
  for (auto &node : TopologicalSort(*graph)) {
    if (node.IsVar() || Agent(&node).deleted()) {
      continue;
    }
    op_nodes.push_back(&node);
  }
  size_t op_num = op_nodes.size();
  for (size_t i = 0; i < op_num; i++) {
    if (op_nodes[i]->IsOp()) continue;
    std::unordered_set<std::string> follow_up_input_names;
    for (size_t j = i + 1; j < op_num; j++) {
      for (auto *in : op_nodes[j]->inputs) {
        follow_up_input_names.insert(in->Name());
      }
    }
    std::vector<Node *> filtered_subgraph_outlinks;
    for (auto *out : op_nodes[i]->outputs) {
      if (follow_up_input_names.count(out->Name())) {
        filtered_subgraph_outlinks.push_back(out);
      } else {
        Agent(out).set_deleted(true);
      }
    }
    // The filtered_subgraph_outlinks may be empty.
    op_nodes[i]->outputs = filtered_subgraph_outlinks;
  }
}

std::vector<std::vector<Node *>> SubgraphDetector::operator()() {
  MarkNodesInsideSubGraph();
  return ExtractSubGraphs();
}

// Mark the output variables inside a subgraph with the func.
inline void MarkOutLinksInSubGraph(const Node *func) {
  for (auto *var : func->outputs) {
    Agent(var).set_marked(true);
  }
}

void SubgraphDetector::MarkNodesInsideSubGraph() {
  for (auto &node : framework::ir::GraphTraits::DFS(*graph_)) {
    if (node_inside_subgraph_teller_(&node)) {
      Agent(&node).set_marked(true);
      if (node.IsOp()) {
        // If a function is inside the sub-graph, mark all the output variables
        // to be inside too, so that two marked functions will be inside a same
        // sub-graph, lets take a example:  A_function->var->B_function, if
        // A_function is marked, var should also be marked, so that B_function
        // will be in the same sub-graph with A_function if B_function is
        // marked.
        MarkOutLinksInSubGraph(&node);
      }
    }
  }
}

// Use the Union Find(UF) algorithm to find fully connected sub-graphs, if node
// a's output is node b, that is a and b is in the same sub-graph. The UF
// algorithm will group them to the same cluster.
using node_map_t = std::map<int, Node *>;
// Find the ancestor id of a node.
int UnionFindGetAncestor(const node_map_t &node_map, size_t id) {
  int tmp = id;
  do {
    tmp = Agent(node_map.at(tmp)).union_find_parent();
  } while (Agent(node_map.at(tmp)).union_find_parent() != tmp);
  return tmp;
}
// Make this two node share the same ancestor.
// TODO(Superjom) bad performance, make a balanced tree latter.
void UnionFindCombine(const node_map_t &node_map, size_t a, size_t b) {
  int a_ancestor = UnionFindGetAncestor(node_map, a);
  int b_ancestor = UnionFindGetAncestor(node_map, b);
  Agent(node_map.at(b_ancestor)).set_union_find_parent(a_ancestor);
  Agent(node_map.at(a)).set_union_find_parent(a_ancestor);
  Agent(node_map.at(b)).set_union_find_parent(a_ancestor);
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
void UnionContractedNodes(const std::map<int, BriefNode *> &node_map,
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

std::vector<std::vector<Node *>> SubgraphDetector::ExtractSubGraphs() {
  // Run the Extract algorithm to find all subgraphs.
  std::vector<Node *> marked_nodes;
  //  We use brief_node_map to represent the original graph in order to avoid
  //  changing the original graph.
  std::map<int, BriefNode *> brief_node_map;

  std::unordered_set<int32_t> valid_node_ids;
  for (auto *node : graph_->Nodes()) {
    valid_node_ids.insert(node->id());
  }

  for (auto &node : framework::ir::GraphTraits::TS(*graph_)) {
    brief_node_map[node.id()] = new BriefNode(&node);
    if (Agent(&node).marked()) {
      marked_nodes.push_back(&node);
    }
  }

  // extract sub-graphs in the marked node set, use Union Find algorithm.
  node_map_t node_map;  // id to ptr
  for (auto *n : marked_nodes) {
    // n's parent == n.id means it is the ancestor
    Agent(n).set_union_find_parent(n->id());
    node_map[n->id()] = n;
  }

  // create breif node map
  for (auto &itr : brief_node_map) {
    for (Node *node : itr.second->node->inputs) {
      if (!valid_node_ids.count(node->id())) {
        LOG(INFO) << "invalid node id " << node->id();
        continue;
      }
      itr.second->inlinks.push_back(brief_node_map.at(node->id()));
    }

    for (Node *node : itr.second->node->outputs) {
      if (!valid_node_ids.count(node->id())) {
        LOG(INFO) << "invalid node id " << node->id();
        continue;
      }
      itr.second->outlinks.push_back(brief_node_map.at(node->id()));
    }
  }

  for (auto &itr : brief_node_map) {
    BriefNode *brief_node = itr.second;

    if (!Agent(brief_node->node).marked()) {
      VLOG(4) << brief_node->node->id() << " node named "
              << brief_node->node->Name() << " is not a trt candidate.";
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
        if (!Agent(out->node).marked()) continue;
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
    if (n->IsOp()) {
      clusters[UnionFindGetAncestor(node_map, Agent(n).union_find_parent())]
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

void SubGraphFuser::operator()() { ReplaceNodesWithSubGraphs(); }

void RemoveIntermediateOutputInSubgraph(const std::vector<Node *> &subgraph,
                                        Graph *graph,
                                        std::vector<Node *> *outputs) {
  std::unordered_set<Node *> subgraph_set(subgraph.begin(), subgraph.end());
  std::unordered_set<Node *> valid_output;

  for (auto *output : *outputs) {
    int num_used = 0;
    for (auto *node : output->outputs) {
      if (!subgraph_set.count(node)) ++num_used;
      if (num_used > 0) valid_output.insert(output);
    }
  }

  outputs->assign(valid_output.begin(), valid_output.end());
}

void DetachDeletedNodes(framework::ir::Graph *graph) {
  std::unordered_set<const Node *> nodes;
  for (auto *node : graph->Nodes()) {
    if (Agent(node).deleted()) {
      node->inputs.clear();
      node->outputs.clear();
    }
  }
}

void SubGraphFuser::ReplaceNodesWithSubGraphs() {
  auto subgraphs = SubgraphDetector(graph_, node_inside_subgraph_teller_)();
  for (auto &subgraph : subgraphs) {
    if (subgraph.size() <= (size_t)min_subgraph_size_) continue;
    std::unordered_set<Node *> subgraph_uniq(subgraph.begin(), subgraph.end());
    // replace this sub-graph with the first node. Two steps: 1. Create a Block
    // Node that contains this subgraph 2. Mark the nodes inside the sub-graph
    // as deleted. 3. Replace the deleted node with the new Block Node.
    framework::OpDesc empty_desc;
    empty_desc.SetType(name_);
    auto *block_node = graph_->CreateOpNode(&empty_desc);
    Agent(block_node).set_subgraph({});
    auto io = ExtractInputAndOutputOfSubGraph(subgraph, graph_);
    block_node->inputs = std::move(io.first);
    block_node->outputs = std::move(io.second);

    RemoveIntermediateOutputInSubgraph(subgraph, graph_, &block_node->outputs);

  framework::BlockDesc *block_cast{nullptr};
  for (auto *op_node : framework::ir::TopologySortOperations(*graph_)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed") block_cast = op_node->Op()->Block();
  }

    // We should place a `cast` op between out_var and `node` op.
    //        |                         |
    //     out_var                   out_var.         int64 but infact int32
    //        |                         |
    //        |           ->        cast_node
    //        |                         |
    //        |                     cast_output_node  int64
    //        |                         |
    //    other ops                  other ops      they must be outside sub-graph
    //        |                         |
  std::unordered_set<Node *> nodes(subgraph.begin(), subgraph.end());
  for (size_t i = 0; i < block_node->outputs.size() * 0; i++) {
    auto out_var = block_node->outputs[i];
    if (out_var->Var()->GetDataType() == framework::proto::VarType::INT64) {
      framework::OpDesc cast_desc(block_cast);
      cast_desc.SetType("cast");
      cast_desc.SetInput("X", {out_var->Name()});
      std::string cast_out_name = out_var->Name() + "_out321" + std::to_string(ii);
      ii++;
      cast_desc.SetOutput("Out", {cast_out_name});
      cast_desc.SetAttr("in_dtype", 2);
      cast_desc.SetAttr("out_dtype", 3);
      cast_desc.Flush();
      auto *cast_node = graph_->CreateOpNode(&cast_desc);
      auto *cast_output_vardesc = block_cast->Var(cast_out_name);
      cast_output_vardesc->SetPersistable(false);
      cast_output_vardesc->SetDataType(framework::proto::VarType::INT64);
      cast_output_vardesc->SetShape(out_var->Var()->GetShape());
      auto *cast_output_node = graph_->CreateVarNode(cast_output_vardesc);

      std::vector<paddle::framework::ir::Node *> new_out_var_outputs;
      for (auto op : out_var->outputs) {
        if (nodes.count(op))
        new_out_var_outputs.push_back(op);
      }
      new_out_var_outputs.push_back(cast_node);
      
      std::vector<Node*> other_ops;
      for (auto op : out_var->outputs) {
        if(!nodes.count(op)) {
          other_ops.push_back(op);
        }
      }

      out_var->outputs = new_out_var_outputs;

      cast_node->inputs = std::vector<paddle::framework::ir::Node *>{out_var};
      cast_node->outputs = std::vector<paddle::framework::ir::Node *>{cast_output_node};
      cast_output_node->inputs = std::vector<paddle::framework::ir::Node *>{cast_node};

      cast_output_node->outputs = other_ops;
      for (auto other_op : other_ops) {
        // in `other_op->Op()`, we need replace `in_var->Name()` with cast_out_name
        for (auto iter : other_op->Op()->Inputs()) {
          auto iter_inputs = iter.second;
          for (auto &var_name : iter_inputs) {
            if (var_name == out_var->Name()) {
              var_name = cast_out_name;
            }
          }
          other_op->Op()->SetInput(iter.first, iter_inputs);
        }
        if (std::find(other_op->inputs.begin(), other_op->inputs.end(), out_var) !=
            other_op->inputs.end()) {
          *(std::find(other_op->inputs.begin(), other_op->inputs.end(), out_var)) =
              cast_output_node;
        }
      }
    }
  }

    for (auto *node : subgraph) {
      // TODO(Superjomn) need a unified mechanism to treat deleted node in each
      // pass.
      Agent(node).set_deleted(true);
      Agent(block_node).subgraph()->push_back(node);
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
    for (auto *i : block_node->inputs) {
      inlink_or_outlink_cleaner(i->outputs);
    }
    for (auto *&o : block_node->outputs) {
      inlink_or_outlink_cleaner(o->inputs);
    }
  }
  // DetachDeletedNodes(graph_);
  FilterRedundantOutputOfSubGraph(graph_);
}

inline bool CheckNodeIndegreeEquals(const Node &node, size_t n) {
  return node.inputs.size() == n;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
