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

#include "paddle/fluid/framework/ir/graph_helper.h"
#include <queue>
#include <stack>
#include "paddle/fluid/framework/op_proto_maker.h"

DECLARE_bool(convert_all_blocks);
PADDLE_DEFINE_EXPORTED_string(print_sub_graph_dir, "",
                              "FLAGS_print_sub_graph_dir is used "
                              "to print the nodes of sub_graphs.");

namespace paddle {
namespace framework {
namespace ir {
namespace {

template <class NodeComparator = ir::NodeComp>
void SortHelper(const std::map<ir::Node *, std::set<ir::Node *, NodeComparator>,
                               NodeComparator> &adj_list,
                ir::Node *node, std::unordered_set<ir::Node *> *visited,
                std::vector<ir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper<NodeComparator>(adj_list, adj, visited, ret);
    }
  }

  VLOG(5) << "topology sort insert: " << node->Name() << " "
          << reinterpret_cast<void *>(node) << " input " << node->inputs.size();
  ret->push_back(node);
}

template <class NodeComparator = ir::NodeComp>
bool HasCircleHelper(
    ir::Node *node,
    const std::map<ir::Node *, std::set<ir::Node *, NodeComparator>,
                   NodeComparator> &adj_list,
    std::unordered_set<ir::Node *> *visited,
    std::unordered_set<ir::Node *> *in_trace,
    std::vector<std::vector<ir::Node *>> *circles) {
  if (visited->find(node) == visited->end()) {
    visited->insert(node);
    in_trace->insert(node);

    for (ir::Node *in : adj_list.at(node)) {
      if (visited->find(in) == visited->end() &&
          HasCircleHelper<NodeComparator>(in, adj_list, visited, in_trace,
                                          circles)) {
        return true;
      } else if (in_trace->find(in) != in_trace->end()) {
        if (circles != nullptr) {
          std::vector<ir::Node *> circle;
          circle.emplace_back(in);
          ir::Node *p = in;
          for (auto &adj : adj_list.at(p)) {
            if (in_trace->count(adj)) {
              circle.emplace_back(adj);
              p = adj;
            }
          }
          circles->emplace_back(circle);
        }
        return true;
      }
    }
  }
  in_trace->erase(node);
  return false;
}

template <class NodeComparator = ir::NodeComp>
bool HasCircleInternal(
    const std::map<ir::Node *, std::set<ir::Node *, NodeComparator>,
                   NodeComparator> &adj_list,
    std::vector<std::vector<ir::Node *>> *circles) {
  std::unordered_set<ir::Node *> visited;
  std::unordered_set<ir::Node *> in_trace;
  for (auto &adj : adj_list) {
    if (HasCircleHelper<NodeComparator>(adj.first, adj_list, &visited,
                                        &in_trace, circles)) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool HasCircle(const Graph &graph) {
  return HasCircleInternal(BuildOperationAdjList(graph), nullptr);
}

bool VarDescIsConsistency(const Graph &graph) {
  std::unordered_map<std::string, std::unordered_set<ir::Node *>>
      var_name2node_set;
  for (ir::Node *node : graph.Nodes()) {
    if (node->IsVar() && node->Var()) {
      var_name2node_set[node->Var()->Name()].emplace(node);
    }
  }
  for (auto &iter : var_name2node_set) {
    auto &first_node = *iter.second.begin();
    bool is_persistable = std::any_of(iter.second.begin(), iter.second.end(),
                                      [&first_node](const ir::Node *node) {
                                        return node->Var()->Persistable();
                                      });
    if (is_persistable) {
      bool is_consistency =
          std::all_of(iter.second.begin(), iter.second.end(),
                      [&first_node](const ir::Node *node) {
                        return *node->Var() == *first_node->Var();
                      });
      if (!is_consistency) return false;
    }
  }
  return true;
}
bool FindCircleSubGraph(const Graph &graph,
                        std::vector<std::vector<ir::Node *>> *circles) {
  return HasCircleInternal(BuildOperationAdjList(graph), circles);
}

std::vector<ir::Node *> TopologySortOperations(const Graph &graph) {
  std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
      adj_list = BuildOperationAdjList(graph);
  PADDLE_ENFORCE_EQ(HasCircleInternal(adj_list, nullptr), false,
                    platform::errors::InvalidArgument(
                        "Generated graph shouldn't contain cycle."));
  std::unordered_set<ir::Node *> visited;
  std::vector<ir::Node *> ret;
  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper<ir::NodeComp>(adj_list, adj.first, &visited, &ret);
    }
  }

  return ret;
}

bool IsTopologySortOperationsUnique(const Graph &graph) {
  auto nodes = TopologySortOperations(graph);
  size_t n = nodes.size();
  for (size_t i = 1; i < n; ++i) {
    auto *prev_op = nodes[i - 1];
    auto *cur_op = nodes[i];

    std::unordered_set<Node *> prev_op_outputs;
    for (auto *output : prev_op->outputs) {
      prev_op_outputs.insert(output);
    }

    bool found = false;
    for (auto *input : cur_op->inputs) {
      if (prev_op_outputs.count(input) > 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

// Build operator outlink edge table.
std::map<ir::Node *, std::unordered_set<ir::Node *>> BuildOperationOutAdjList(
    const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list;

  for (auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::unordered_set<ir::Node *>();
    }
    for (auto &var : n->outputs) {
      for (auto &adj_n : var->outputs) {
        PADDLE_ENFORCE_EQ(
            adj_n->NodeType(), ir::Node::Type::kOperation,
            platform::errors::InvalidArgument(
                "Node(%s)'s type(%d) must be kOperation type.", adj_n->Name(),
                static_cast<int>(adj_n->NodeType())));
        VLOG(40) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                 << " -> " << n->Name() << reinterpret_cast<void *>(n)
                 << "  via " << var->Name() << reinterpret_cast<void *>(var);
        adj_list[n].insert(adj_n);
      }
    }
  }
  return adj_list;
}

std::vector<ir::Node *> OpDFSSort(const Graph &graph) {
  auto edge_table = BuildOperationOutAdjList(graph);
  std::stack<Node *> stack;
  for (auto &ele : edge_table) {
    if (ele.first->inputs.empty()) {
      // find the input ops (those without input vars)
      stack.push(ele.first);
    } else {
      // find the ops with only persistable vars as inputs.
      bool all_persistable = true;
      for (auto *input : ele.first->inputs) {
        if (!(input->IsVar() && input->Var() && input->Var()->Persistable())) {
          all_persistable = false;
        }
      }
      if (all_persistable) {
        stack.push(ele.first);
      }
    }
  }

  std::vector<Node *> res;
  // start from the feed op and DFS
  std::unordered_set<Node *> unique_set;
  while (!stack.empty()) {
    // will start from the last feed by default.
    auto cur = stack.top();
    stack.pop();
    unique_set.insert(cur);
    res.push_back(cur);

    for (auto *op : edge_table[cur]) {
      if (!unique_set.count(op)) {
        stack.push(op);
      }
    }
  }
  return res;
}

std::vector<ir::Node *> TopologyDfsSortOperations(const Graph &graph) {
  std::vector<ir::Node *> nodes;
  std::unordered_map<Node *, int> in_degree;

  auto set_out_ops_ready = [&](Node *var) {
    for (auto *op : var->outputs) {
      --in_degree[op];
    }
  };
  // build in_degree
  for (auto *node : graph.Nodes()) {
    if (node->IsOp()) {
      in_degree[node] += node->inputs.size();
    } else if (node->IsVar() && node->inputs.empty()) {
      // put all the inputs of the whole graph ready.
      set_out_ops_ready(node);
    }
  }

  std::deque<Node *> op_queue;
  // first visit
  for (auto &node : OpDFSSort(graph)) {
    if (node->IsOp()) {
      op_queue.push_back(node);
    }
  }

  // traverse the graph
  int num_ops = op_queue.size();
  while (num_ops) {
    for (auto it = op_queue.begin(); it != op_queue.end(); it++) {
      auto *&cur_op = *it;
      if (!cur_op || in_degree[cur_op] > 0) continue;
      // visit this node
      // put all the output var of this op valid.
      for (auto *out_var : cur_op->outputs) {
        if (!out_var) continue;
        set_out_ops_ready(out_var);
      }
      VLOG(8) << "visit " << cur_op->Name();
      nodes.push_back(cur_op);

      cur_op = nullptr;
      num_ops--;
    }
  }

  return nodes;
}

size_t GraphNum(const Graph &graph) {
  std::unordered_set<ir::Node *> nodes(graph.Nodes());
  std::unordered_set<ir::Node *> visited_nodes;
  visited_nodes.reserve(nodes.size());
  std::deque<ir::Node *> q_nodes;
  std::vector<std::unordered_set<ir::Node *>> graph_nodes;
  std::unordered_set<ir::Node *> g_nodes;
  // q_set used to record records in the queue.
  std::unordered_set<ir::Node *> q_set;
  size_t graph_count = 0;

  auto traverse_nodes = [&visited_nodes, &q_nodes,
                         &q_set](const std::vector<ir::Node *> &nodes) {
    for (auto n : nodes) {
      if (visited_nodes.count(n) == 0 && q_set.count(n) == 0) {
        q_nodes.push_back(n);
        q_set.insert(n);
      }
    }
  };

  while (visited_nodes.size() != nodes.size()) {
    if (!q_nodes.empty()) {
      auto cur_node = q_nodes.front();
      q_nodes.pop_front();
      q_set.erase(cur_node);
      visited_nodes.insert(cur_node);
      g_nodes.insert(cur_node);
      traverse_nodes(cur_node->inputs);
      traverse_nodes(cur_node->outputs);
    } else {
      ++graph_count;
      if (g_nodes.size()) {
        graph_nodes.emplace_back(g_nodes);
      }
      g_nodes.clear();
      for (auto &n : nodes) {
        if (visited_nodes.count(n) == 0) {
          q_nodes.push_back(n);
          q_set.insert(n);
          break;
        }
      }
    }
  }

  if (g_nodes.size()) {
    graph_nodes.emplace_back(g_nodes);
  }

  if (FLAGS_print_sub_graph_dir.size()) {
    if (graph_nodes.size() > 1) {
      std::stringstream out;
      for (auto &g_n : graph_nodes) {
        out << "graph_nodes: " << g_n.size() << "\n";
      }
      out << "\n\n";
      for (auto &g_n : graph_nodes) {
        out << "graph_nodes: " << g_n.size();
        for (auto &node : g_n) {
          out << "\nNode: " << node->Name() << " in [";
          for (auto &n : node->inputs) {
            out << n->Name() << ", ";
          }
          out << "], out[";
          for (auto &n : node->outputs) {
            out << n->Name() << ", ";
          }
          out << "]";
        }
        out << "\n\n\n";
      }
      std::unique_ptr<std::ostream> fout(
          new std::ofstream(FLAGS_print_sub_graph_dir));
      PADDLE_ENFORCE_EQ(fout->good(), true,
                        platform::errors::Unavailable(
                            "Can not open file %s for printing the graph.",
                            FLAGS_print_sub_graph_dir));
      *fout << out.str();
    }
  }

  return graph_count;
}

void CleanIndividualNodes(Graph *graph) {
  std::unordered_set<Node *> nodes2rm;
  for (auto *node : graph->Nodes()) {
    if (node->inputs.empty() && node->outputs.empty()) {
      nodes2rm.insert(node);
    }
  }

  for (auto *node : nodes2rm) {
    graph->RemoveNode(node);
  }
}

std::vector<Node *> TopologyVarientSort(const Graph &graph,
                                        SortKind sort_kind) {
  switch (sort_kind) {
    case SortKind::TS:
      return framework::ir::TopologySortOperations(graph);
    default:
      return framework::ir::TopologyDfsSortOperations(graph);
  }
}

class DescOrderComparator {
 public:
  bool operator()(Node *const &n1, Node *const &n2) const {
    if (n1->DescOrder() < n2->DescOrder()) {
      return true;
    } else if (n1->DescOrder() == n2->DescOrder()) {
      return n1->id() < n2->id() ||
             (n1->id() == n2->id() && n1->ToString() < n2->ToString());
    }
    return false;
  }
};

std::vector<ir::Node *> TopologySortGraphByDescOrder(const Graph &graph) {
  std::map<ir::Node *, std::set<ir::Node *, DescOrderComparator>,
           DescOrderComparator>
      adj_list = BuildOperationAdjList<DescOrderComparator>(graph);
  PADDLE_ENFORCE_EQ(HasCircleInternal<DescOrderComparator>(adj_list, nullptr),
                    false, platform::errors::InvalidArgument(
                               "Generated graph shouldn't contain cycle."));
  std::unordered_set<ir::Node *> visited;
  std::vector<ir::Node *> ret;
  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper<DescOrderComparator>(adj_list, adj.first, &visited, &ret);
    }
  }

  return ret;
}

static OpDesc *ReplaceScaleLossGradOp(const Node &node, OpDesc *desc) {
  desc->SetType("fill_constant");
  desc->SetAttr(
      OpProtoAndCheckerMaker::OpRoleAttrName(),
      (static_cast<int>(OpRole::kBackward) | static_cast<int>(OpRole::kLoss)));
  desc->SetAttr("value", 1.0f);
  std::vector<std::string> output_names;
  for (auto out : node.outputs) {
    output_names.emplace_back(out->Name());
  }
  desc->SetOutput("Out", output_names);
  return desc;
}

static void GetGraphOpDesc(const std::vector<Node *> &nodes,
                           std::vector<OpDesc> *ops) {
  for (Node *n : nodes) {
    // if node is not Op, skip
    if (!n->IsOp()) continue;

    // create fill_constant op
    if (n->Name() == "scale_loss_grad") {
      ops->emplace_back();
      auto &desc = ops->back();
      ReplaceScaleLossGradOp(*n, &desc);
    } else if (n->Op()) {
      ops->emplace_back(*n->Op());
    }
    // delete no OpDesc op
  }
}

static void GraphToBlock(const Graph &graph, proto::BlockDesc *block,
                         const SortKind *sort_kind) {
  // Remove the unneeded variables after memory optimization.
  std::unordered_set<std::string> vars2remove;
  if (graph.Has(kGraphToProgramVarsToRemove)) {
    vars2remove =
        graph.Get<std::unordered_set<std::string>>(kGraphToProgramVarsToRemove);
    VLOG(2) << "graph (id: " << block->idx() << ") to program remove "
            << vars2remove.size() << " nodes";
  }

  block->clear_vars();
  std::unordered_set<std::string> visited_vars;
  for (Node *n : graph.Nodes()) {
    if (n->IsVar()) {
      if (n->Var() && visited_vars.count(n->Var()->Name()) == 0 &&
          !vars2remove.count(n->Var()->Name()) &&
          n->GetVarNodeBlockId() == graph.GetBlockId()) {
        visited_vars.insert(n->Var()->Name());
        block->add_vars()->MergeFrom(*n->Var()->Proto());
      }
    }
  }
  block->clear_ops();

  std::vector<Node *> nodes;
  if (sort_kind != nullptr) {
    // Inference Memory Optimize relays on this branch.
    nodes = TopologyVarientSort(graph, *sort_kind);
  } else {
    if (FLAGS_convert_all_blocks) {
      nodes = TopologySortGraphByDescOrder(graph);
    } else {
      nodes = TopologySortOperations(graph);
    }
  }

  std::vector<OpDesc> ops;
  GetGraphOpDesc(nodes, &ops);
  for (auto &op : ops) {
    block->add_ops()->MergeFrom(*op.Proto());
  }
}

void GraphToProgram(const Graph &graph, ProgramDesc *program,
                    const SortKind *sort_kind) {
  PADDLE_ENFORCE_EQ(graph.IsMainGraph(), true,
                    platform::errors::InvalidArgument(
                        "This graph is a sub_graph, "
                        "and can't convert to program individually"));
  PADDLE_ENFORCE_NOT_NULL(
      program,
      platform::errors::InvalidArgument(
          "program must not be nullptr when converting graph to program"));

  proto::ProgramDesc program_pb(*(program->Proto()));
  auto block = program_pb.mutable_blocks(kRootBlockIndex);
  block->set_idx(kRootBlockIndex);

  if (FLAGS_convert_all_blocks) {
    GraphToBlock(*graph.GetSubGraph(kRootBlockIndex), block, sort_kind);

    VLOG(3) << "Graph to program need convert " << graph.SubGraphsSize()
            << " sub graph";
    for (size_t idx = 0; idx < graph.SubGraphsSize(); ++idx) {
      // avoid kRootBlockIndex not 0
      if (idx == kRootBlockIndex) continue;

      block = program_pb.add_blocks();
      block->set_idx(idx);
      block->set_parent_idx(kRootBlockIndex);
      GraphToBlock(*graph.GetSubGraph(idx), block, sort_kind);
    }
  } else {
    GraphToBlock(graph, block, sort_kind);
  }

  program->CopyFrom(program_pb);
}

static std::vector<std::vector<ir::Node::Dep>> GetOpDependencies(
    const BlockDesc &block, const std::unordered_set<ir::Node *> &nodes) {
  auto block_ops = block.AllOps();
  size_t op_num = block_ops.size();
  std::unordered_map<const ir::Node *, std::unordered_set<const ir::Node *>>
      preceding_ops(op_num);
  std::unordered_map<const ir::Node *, size_t> preceding_deps(op_num);
  std::unordered_map<const ir::Node *, std::unordered_set<const ir::Node *>>
      pending_ops(op_num);

  std::queue<const ir::Node *> ready_ops;
  for (const auto *node : nodes) {
    if (!node->IsOp()) continue;

    auto &tmp_preceding_ops = preceding_ops[node];
    for (const auto *in_var : node->inputs) {
      for (const auto *in_op : in_var->inputs) {
        tmp_preceding_ops.insert(in_op);
      }
    }
    if (tmp_preceding_ops.empty()) {
      ready_ops.push(node);
    }
    preceding_deps[node] = tmp_preceding_ops.size();

    auto &tmp_pending_ops = pending_ops[node];
    for (const auto *out_var : node->outputs) {
      for (const auto *out_op : out_var->outputs) {
        tmp_pending_ops.insert(out_op);
      }
    }
  }

  std::unordered_map<const ir::Node *, std::unordered_set<const ir::Node *>>
      all_preceding_ops;
  while (!ready_ops.empty()) {
    const auto *cur_op = ready_ops.front();
    ready_ops.pop();

    auto &all_preceding_ops_of_cur_op = all_preceding_ops[cur_op];
    for (const auto *preceding_op : preceding_ops.at(cur_op)) {
      all_preceding_ops_of_cur_op.insert(preceding_op);
      auto &prev_preceding_ops = all_preceding_ops[preceding_op];
      all_preceding_ops_of_cur_op.insert(prev_preceding_ops.begin(),
                                         prev_preceding_ops.end());
    }

    for (const auto *pending_op : pending_ops.at(cur_op)) {
      if (--preceding_deps.at(pending_op) == 0) {
        ready_ops.push(pending_op);
      }
    }
  }

  std::unordered_map<uint64_t, size_t> op_id_to_idx(op_num);
  for (const auto *op_desc : block_ops) {
    size_t op_idx = op_id_to_idx.size();
    PADDLE_ENFORCE_EQ(
        op_id_to_idx.emplace(op_desc->OriginalId(), op_idx).second, true,
        platform::errors::InvalidArgument(
            "There should not be duplicate op id: %d", op_desc->OriginalId()));
  }

  std::vector<std::vector<ir::Node::Dep>> dep_matrix(op_num);
  for (size_t i = 0; i < op_num; ++i) {
    dep_matrix[i].resize(op_num, ir::Node::Dep::kNoDep);
    dep_matrix[i][i] = ir::Node::Dep::kSame;
  }

  auto get_op_idx_by_id = [&op_id_to_idx](uint64_t op_id) {
    auto iter = op_id_to_idx.find(op_id);
    PADDLE_ENFORCE_NE(iter, op_id_to_idx.end(),
                      platform::errors::InvalidArgument(
                          "Cannot find OpDesc with id %d", op_id));
    return iter->second;
  };

  for (const auto &pair : all_preceding_ops) {
    const auto *cur_op_node = pair.first;
    size_t op_idx_1 = get_op_idx_by_id(cur_op_node->Op()->OriginalId());
    for (const auto *preceding_op_node : pair.second) {
      size_t op_idx_2 = get_op_idx_by_id(preceding_op_node->Op()->OriginalId());
      dep_matrix[op_idx_1][op_idx_2] = ir::Node::Dep::kAfter;
      dep_matrix[op_idx_2][op_idx_1] = ir::Node::Dep::kBefore;
    }
  }
  return dep_matrix;
}

std::vector<std::vector<std::vector<ir::Node::Dep>>> GetOpDependencies(
    const ProgramDesc &program) {
  ir::Graph graph(program);
  size_t block_num = program.Size();
  std::vector<std::vector<std::vector<ir::Node::Dep>>> deps;
  deps.reserve(block_num);
  for (size_t i = 0; i < block_num; ++i) {
    deps.emplace_back(
        GetOpDependencies(program.Block(i), graph.GetSubGraph(i)->Nodes()));
  }
  return deps;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
