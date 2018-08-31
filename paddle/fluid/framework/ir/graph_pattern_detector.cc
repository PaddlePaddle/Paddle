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

#include <array>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

size_t PDPattern::id_ = 0UL;

PDNode* PDPattern::NewNode(PDNode::teller_t&& teller, const std::string& name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(node_map_.count(name), 0,
                      "PDNode's name should be unique, get duplicate [%s]",
                      name);
  }

  nodes_.emplace_back(new PDNode(std::move(teller), this, name));
  auto* cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode* PDPattern::RetriveNode(const std::string& id) const {
  auto it = node_map_.find(id);
  if (it == node_map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PDPattern::AddEdge(PDNode* a, PDNode* b) {
  PADDLE_ENFORCE(a);
  PADDLE_ENFORCE(b);
  PADDLE_ENFORCE(a != b, "can't connect to the same nodes.");
  edges_.emplace_back(a, b);
}

void GraphPatternDetector::operator()(Graph* graph,
                                      GraphPatternDetector::handle_t handler) {
  if (!MarkPDNodesInGraph(*graph)) return;
  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);

  LOG(INFO) << "detect " << subgraphs.size() << " subgraph matches the pattern";
  int id = 0;
  for (auto& g : subgraphs) {
    VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool GraphPatternDetector::MarkPDNodesInGraph(const ir::Graph& graph) {
  VLOG(4) << "mark pdnodes in graph";
  if (graph.Nodes().empty()) return false;

  for (auto& node : GraphTraits::DFS(graph)) {
    for (const auto& pdnode : pattern_.nodes()) {
      if (pdnode->Tell(&node)) {
        VLOG(4) << "pdnode " << pdnode->name() << " marked";
        pdnodes2nodes_[pdnode.get()].insert(&node);
      }
    }
  }
  VLOG(3) << pdnodes2nodes_.size() << " nodes marked";
  return !pdnodes2nodes_.empty();
}

struct HitGroup {
  std::unordered_map<PDNode*, Node*> roles;

  bool Match(Node* node, PDNode* pat) {
    if (nodes_.count(node)) {
      if (!roles.count(pat)) return false;
      return roles[pat] == node;
    }
    return !roles.count(pat) || roles.at(pat) == node;
  }

  void Register(Node* node, PDNode* pat) {
    roles[pat] = node;
    nodes_.insert(node);
  }

 private:
  std::unordered_set<Node*> nodes_;
};

// Tell whether Node a links to b.
bool IsNodesLink(Node* a, Node* b) {
  for (auto* node : a->outputs) {
    if (b == node) {
      return true;
    }
  }
  return false;
}

std::vector<GraphPatternDetector::subgraph_t>
GraphPatternDetector::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<GraphPatternDetector::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  // PADDLE_ENFORCE(!pattern_.edges().empty(), "At least one edge is needed");
  auto* first_pnode = pattern_.edges().empty() ? pattern().nodes().front().get()
                                               : pattern_.edges().front().first;
  if (!pdnodes2nodes_.count(first_pnode)) return result;
  for (auto* node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[first_pnode] = node;
    init_groups.emplace_back(group);
  }

  int step = 0;
  bi_records[0] = std::move(init_groups);

  // Extend a PDNode to subgraphs by deducing the connection relations defined
  // in edges of PDNodes.
  for (const auto& edge : pattern_.edges()) {
    VLOG(4) << "check " << edge.first->name() << " -> " << edge.second->name();
    // Each role has two PDNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.
    auto& pre_groups = bi_records[step % 2];
    auto& cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (Node* source : pdnodes2nodes_[edge.first]) {
      for (Node* target : pdnodes2nodes_[edge.second]) {
        // TODO(Superjomn) add some prune strategies.
        for (const auto& group : pre_groups) {
          HitGroup new_group = group;
          if (IsNodesLink(source, target) &&
              new_group.Match(source, edge.first)) {
            new_group.Register(source, edge.first);
            if (new_group.Match(target, edge.second)) {
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
              // TODO(Superjomn) need to unique
            }
          }
        }
      }
    }
    VLOG(3) << "step " << step << " get records: " << cur_groups.size();
  }

  for (auto& group : bi_records[step % 2]) {
    GraphPatternDetector::subgraph_t subgraph;
    for (auto& role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

void GraphPatternDetector::UniquePatterns(
    std::vector<GraphPatternDetector::subgraph_t>* subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<GraphPatternDetector::subgraph_t> result;

  std::unordered_set<size_t> set;
  for (auto& g : *subgraphs) {
    size_t key = 0;
    for (auto& item : g) {
      key ^= std::hash<void*>{}(item.first);
      key ^= std::hash<void*>{}(item.second);
    }
    if (!set.count(key)) {
      result.emplace_back(g);
      set.insert(key);
    }
  }
  *subgraphs = result;
}

void GraphPatternDetector::RemoveOverlappedMatch(
    std::vector<subgraph_t>* subgraphs) {
  std::vector<subgraph_t> result;
  std::unordered_set<Node*> node_set;

  for (const auto& subgraph : *subgraphs) {
    bool valid = true;
    for (auto& item : subgraph) {
      if (node_set.count(item.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto& item : subgraph) {
        node_set.insert(item.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

std::string PDPattern::DotString() const {
  using inference::analysis::Dot;
  Dot dot;
  int id = 0;
  // Create Nodes
  std::unordered_map<PDNode*, std::string> node2dot;
  for (const auto& node : nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    dot.AddNode(node_id, {}, node->name());
    node2dot[node.get()] = node_id;
  }
  // Create Edges
  for (const auto& edge : edges()) {
    if (!node2dot.count(edge.first) || !node2dot.count(edge.second)) {
      LOG(ERROR) << "no node " << edge.first << " " << edge.second;
      continue;
    }
    auto& src = node2dot.at(edge.first);
    auto& trg = node2dot.at(edge.second);
    dot.AddEdge(src, trg, {});
  }
  return dot.Build();
}

PDNode& PDNode::LinksTo(const std::vector<PDNode*>& others) {
  // extend outlinks.
  for (PDNode* x : others) {
    pattern_->AddEdge(this, x);
  }
  return *this;
}

PDNode& PDNode::LinksFrom(const std::vector<PDNode*>& others) {
  // extend outlinks.
  for (PDNode* x : others) {
    pattern_->AddEdge(x, this);
  }
  return *this;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
