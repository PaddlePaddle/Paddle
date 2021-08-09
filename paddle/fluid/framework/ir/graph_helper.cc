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
#include "paddle/fluid/framework/operator.h"

DEFINE_string(print_sub_graph_dir, "",
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
      // return n1->ToString() < n2->ToString();
      return n1->id() > n2->id() ||
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

bool NodeCanBlockWrite(Node *node) {
  return !node->outputs.empty() && !node->IsCtrlVar() &&
         node->Name() != kEmptyVarName;
  // && node->Var() != nullptr &&
  // node->Var()->GetType() != proto::VarType::LOD_TENSOR_ARRAY;
}
/*
bool SimulateRunNode(Node* node, std::set<std::string> *var_cannot_overwrite) {
  // Check if we run node op, what changes var_cannot_overwrite
    for (Node *in_var : node->inputs) {
      if (var_cannot_overwrite->find(in_var->Name()) !=
          var_cannot_overwrite->end()) {
        // If all output op of in_var is in sorted_op, remove in_var from
        // var_cannot_overwrite
        bool all_read = true;
        for (Node *read_op : in_var->outputs) {
          if (std::find(sorted_ops->begin(), sorted_ops->end(), read_op) ==
              sorted_ops->end()) {
            all_read = false;
            break;
          }
        }
        if (all_read) {
          var_cannot_overwrite->erase(in_var->Name());
        }
      }
    }
    // all output vars should not be overwritten now.
    bool out_has_conflict = false;
    for (Node *out_var : node->outputs) {
      if (var_cannot_overwrite->find(out_var->Name()) !=
          var_cannot_overwrite->end()) {
        out_has_conflict = true;
        break;
      } else if (NodeCanBlockWrite(out_var)) {
        var_cannot_overwrite->insert(out_var->Name());
      }
    }
    return !out_has_conflict;
}

template <class NodeComparator = ir::NodeComp>
bool ResolveHazardSortHelper(const std::map<ir::Node *, std::set<ir::Node *,
NodeComparator>,
                               NodeComparator> &adj_list,
                std::set<ir::Node*, NodeComparator>* out_nodes,
                std::unordered_set<ir::Node *> *visited,
                std::vector<ir::Node *> *ret,
                std::set<std::string> *var_cannot_overwrite) {

   while (!out_nodes.empty()) {
     bool has_update = false;
     std::unordered_set<ir::Node *> backup_visited(*visited);
     std::vector<ir::Node *> backup_ret(*ret);
     std::set<std::string> backup_var_cannot_overwrite(*var_cannot_overwrite);
     for (auto iter = out_nodes.begin(); iter != out_nodes.end(); ++iter) {
       Node* node = *iter;
       if (visited->find(node) != visited->end()) {
         has_update = true;
         out_nodes.erase(node);
       }

       visited->insert(node);
       std::set<ir::Node *, NodeComparator> adj = adj_list.at(node);
       bool success = ResolveHazardSortHelper<NodeComparator>(adj_list, &adj,
visited, var_cannot_overwrite);
       if (success && SimulateRunNode(node, var_cannot_overwrite)) {
           ret->push_back(node);
           has_update = true;
           out_nodes.erase(node);
       } else {
         visited = backup_visited;
         ret = backup_ret;
         var_cannot_overwrite = backup_var_cannot_overwrite;
       }
     }
     if (!has_update) {
       return false;
     }
   }

   return true;
}

std::vector<ir::Node *> TopologySortGraphByDescOrderResolveHazard(const Graph
&graph) {
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
      ResolveHazardSortHelper<DescOrderComparator>(adj_list, adj.first,
&visited, &ret);
    }
  }

  return ret;
}
*/
Node *CheckHazardVar(const Graph &graph, std::vector<Node *> *sorted_ops) {
  std::set<std::string> var_cannot_overwrite;
  for (size_t i = 0; i < sorted_ops->size(); ++i) {
    Node *node = sorted_ops->at(i);
    // Check if we run node op, what changes var_cannot_overwrite
    for (Node *in_var : node->inputs) {
      if (var_cannot_overwrite.find(in_var->Name()) !=
          var_cannot_overwrite.end()) {
        // If all output op of in_var is in sorted_op, remove in_var from
        // var_cannot_overwrite
        bool all_read = true;
        for (Node *read_op : in_var->outputs) {
          if (std::find(sorted_ops->begin(), sorted_ops->begin() + i + 1,
                        read_op) == sorted_ops->end()) {
            all_read = false;
            break;
          }
        }
        if (all_read) {
          var_cannot_overwrite.erase(in_var->Name());
        }
      }
    }
    // all output vars should not be overwritten now.
    for (Node *out_var : node->outputs) {
      if (var_cannot_overwrite.find(out_var->Name()) !=
          var_cannot_overwrite.end()) {
        return out_var;
      } else if (NodeCanBlockWrite(out_var)) {
        var_cannot_overwrite.insert(out_var->Name());
      }
    }
  }
  return nullptr;
}

template <class NodeComparator>
bool TopologySortSolveHazardHelper(
    std::vector<Node *> *sorted_ops, std::vector<Node *> *candidates,
    std::map<Node *, int, NodeComparator> *in_degree,
    std::set<std::string> *var_cannot_overwrite) {
  std::set<std::string> save_var_cannot_overwrite(*var_cannot_overwrite);
  for (Node *node : *candidates) {
    // for (auto iter = candidates->rbegin(); iter != candidates->rend();
    // ++iter) {
    //  Node* node = *iter;
    /*
    LOG(WARNING) << "sorted_ops:";
    for (Node *op : *sorted_ops) {
      LOG(WARNING) << "sorted node->id() = " << op->id()
                   << ", node->Name() = " << op->Name()
                   << ", node pointer = " << op;
    }
    LOG(WARNING) << "checking node->id() = " << node->id()
                 << ", node->Name() = " << node->Name()
                 << ", node pointer = " << node;
    LOG(WARNING) << "Before checking, var_cannot_overwrite = ";
    std::stringstream before_ss;
    for (std::string var : *var_cannot_overwrite) {
      before_ss << var << ", ";
    }
    LOG(WARNING) << before_ss.str();
    */
    sorted_ops->push_back(node);
    std::cout << "sorted_ops.size() = " << sorted_ops->size()
              << ", in_degree->size() = " << in_degree->size() << std::endl;
    std::cout << "current node id = " << node->id()
              << ", node->Name() = " << node->Name() << std::endl;
    // Check if we run node op, what changes var_cannot_overwrite
    for (Node *in_var : node->inputs) {
      if (var_cannot_overwrite->find(in_var->Name()) !=
          var_cannot_overwrite->end()) {
        // If all output op of in_var is in sorted_op, remove in_var from
        // var_cannot_overwrite
        bool all_read = true;
        for (Node *read_op : in_var->outputs) {
          if (std::find(sorted_ops->begin(), sorted_ops->end(), read_op) ==
              sorted_ops->end()) {
            all_read = false;
            break;
          }
        }
        if (all_read) {
          var_cannot_overwrite->erase(in_var->Name());
        }
      }
    }
    // all output vars should not be overwritten now.
    bool out_has_conflict = false;
    for (Node *out_var : node->outputs) {
      if (var_cannot_overwrite->find(out_var->Name()) !=
          var_cannot_overwrite->end()) {
        out_has_conflict = true;
        break;
      } else if (NodeCanBlockWrite(out_var)) {
        var_cannot_overwrite->insert(out_var->Name());
      }
    }

    /*
    LOG(WARNING) << "After checking, var_cannot_overwrite = ";
    std::stringstream after_ss;
    for (std::string var : *var_cannot_overwrite) {
      after_ss << var << ", ";
    }
    LOG(WARNING) << after_ss.str();
    LOG(WARNING) << "out_has_conflict = " << out_has_conflict;
    */

    if (!out_has_conflict) {
      if (sorted_ops->size() == in_degree->size()) {
        return true;
      }

      std::set<ir::Node *, DescOrderComparator> out_candidates;
      for (Node *out_var : node->outputs) {
        for (Node *out_op : out_var->outputs) {
          if (out_op->Name() == "fetch" ||
              in_degree->find(out_op) == in_degree->end()) {
            continue;
          }
          --(in_degree->at(out_op));
          if (in_degree->at(out_op) == 0) {
            out_candidates.insert(out_op);
          }
        }
      }

      std::vector<ir::Node *> next_step_candidates(*candidates);
      auto cur_node_iter = std::find(next_step_candidates.begin(),
                                     next_step_candidates.end(), node);
      if (cur_node_iter != next_step_candidates.end()) {
        next_step_candidates.erase(cur_node_iter);
      }

      if (out_candidates.size() == 1) {
        next_step_candidates.insert(next_step_candidates.begin(),
                                    *(out_candidates.begin()));
      } else {
        next_step_candidates.insert(next_step_candidates.end(),
                                    out_candidates.begin(),
                                    out_candidates.end());
      }
      bool success = TopologySortSolveHazardHelper(
          sorted_ops, &next_step_candidates, in_degree, var_cannot_overwrite);
      if (success) {
        return true;
      }
      // if not success, restore
      for (Node *out_var : node->outputs) {
        for (Node *out_op : out_var->outputs) {
          if (out_op->Name() == "fetch" ||
              in_degree->find(out_op) == in_degree->end()) {
            continue;
          }
          ++(in_degree->at(out_op));
        }
      }
    }
    sorted_ops->pop_back();
    *var_cannot_overwrite = save_var_cannot_overwrite;
  }
  return false;
}

std::vector<ir::Node *> TopologySortSolveHazard(const Graph &graph) {
  std::set<ir::Node *, DescOrderComparator> candidates;
  std::map<ir::Node *, int, DescOrderComparator> in_degree;
  // Becuase fetch op has async optimize, running fetch ops doesn't mean that
  // you can overwrite the input of fetch_ops. We have to handle fetch op topo
  // sort specially: put them at the end;
  std::set<ir::Node *, DescOrderComparator> fetch_ops;

  for (Node *n : graph.Nodes()) {
    if (!n->IsOp()) {
      continue;
    }
    if (n->Name() == "fetch") {
      fetch_ops.insert(n);
    } else {
      in_degree[n] = 0;
    }
  }
  for (Node *n : graph.Nodes()) {
    if (!n->IsOp()) {
      continue;
    }
    for (Node *var : n->outputs) {
      for (Node *out : var->outputs) {
        if (out->Name() != "fetch" && in_degree.find(out) != in_degree.end()) {
          ++in_degree[out];
        }
      }
    }
  }

  for (std::pair<ir::Node *, int> node_in_degree : in_degree) {
    if (node_in_degree.second == 0) {
      candidates.insert(node_in_degree.first);
    }
  }

  std::vector<ir::Node *> sorted_ops;
  if (in_degree.size() > 0) {
    std::set<std::string> var_cannot_overwrite;
    LOG(WARNING) << "in_degree.size() = " << in_degree.size()
                 << ", fetch_ops.size() = " << fetch_ops.size();
    std::vector<Node *> priority_candidates(candidates.begin(),
                                            candidates.end());
    bool sort_success = TopologySortSolveHazardHelper<DescOrderComparator>(
        &sorted_ops, &priority_candidates, &in_degree, &var_cannot_overwrite);

    PADDLE_ENFORCE_EQ(sort_success, true,
                      platform::errors::PreconditionNotMet(
                          "The Graph contains circle or unsolve Hazard in "
                          "TopologySortSolveHazard"));
  }
  sorted_ops.insert(sorted_ops.end(), fetch_ops.begin(), fetch_ops.end());
  return sorted_ops;
}

std::vector<ir::Node *> NewTopologySortGraphByDescOrder(const Graph &graph) {
  std::vector<ir::Node *> sorted_ops;
  std::priority_queue<Node *, std::vector<Node *>, DescOrderComparator> q;
  std::unordered_map<Node *, std::unordered_set<Node *>> in_ops;
  std::unordered_map<Node *, std::unordered_set<Node *>> out_ops;

  // ensure all op node in 'in_ops' and 'out_ops'
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    in_ops.emplace(n, std::unordered_set<Node *>());
    out_ops.emplace(n, std::unordered_set<Node *>());
  }

  // record all op's input op and output op
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    // traverse all input op
    for (const auto &var : n->inputs) {
      for (const auto &in : var->inputs) {
        // use at instead of [] to prevent no unrecorded op node
        in_ops.at(n).insert(in);
        out_ops.at(in).insert(n);
      }
    }
  }

  // find topology entrance
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    if (in_ops.at(n).empty()) {
      q.push(n);
    }
  }

  // topological sorting
  while (!q.empty()) {
    // Do not get by reference!!! The element will pop later.
    const auto cur_op = q.top();
    q.pop();

    sorted_ops.push_back(cur_op);
    for (const auto &out : out_ops.at(cur_op)) {
      PADDLE_ENFORCE_GT(in_ops.at(out).count(cur_op), 0,
                        platform::errors::InvalidArgument(
                            "We find %s in %s's output list, "
                            "but cannot find %s in %s's input list. "
                            "Please ensure graph completely.",
                            out->Name().c_str(), cur_op->Name().c_str(),
                            cur_op->Name().c_str(), out->Name().c_str()));
      in_ops.at(out).erase(cur_op);

      // push if in-degree is 0
      if (in_ops.at(out).empty()) {
        q.push(out);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      sorted_ops.size(), in_ops.size(),
      platform::errors::InvalidArgument("Topological sorting incompletely, "
                                        "only sorted %zd op but total %zd.",
                                        sorted_ops.size(), in_ops.size()));

  return sorted_ops;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
