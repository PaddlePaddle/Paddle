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

#include "paddle/fluid/inference/analysis/fuse.h"
#include "paddle/fluid/inference/analysis/graph_traits.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace fuse {

FusePatternNode *Pattern::AddNode() {
  return dynamic_cast<FusePatternNode *>(
      pattern_graph_.nodes.Create(Node::Type::kFusePattern));
}

void Pattern::AddEdge(FusePatternNode *source, FusePatternNode *target) {
  source->outlinks.push_back(target);
  target->inlinks.push_back(source);
  edges_.emplace_back(source, target);
}

void Pattern::SetHandle(const Pattern::handle_t &handle) { handle_ = handle; }

// Whether two nodes links with each other.
bool Links(const Node &a, const Node &b) {
  std::unordered_set<Node *> outlinks(a.outlinks.begin(), a.outlinks.end());
  return outlinks.count(const_cast<Node *>(&b));
}

std::vector<PatternRecord> Pattern::Match(DataFlowGraph *graph) {
  MarkNodesInPattern(graph);
  std::vector<PatternRecord> pattern_records;
  // auto two_gram_hits = Extract2GramPatterns(graph);
  // Create an initial FuseRecord for each 2-gram pattern.
  /*
  std::transform(two_gram_hits.begin(), two_gram_hits.end(),
                 std::back_inserter(pattern_records),
                 [](std::pair<hit_t, hit_t> &x) {
                   return PatternRecord(x.first, x.second);
                 });
                 */
  auto patterns = ExtractPatterns(graph);
  // check results
  // The pattern should have the same number of nodes with pattern.

  // help to determine whether a node in already in the pattern.
  std::unordered_set<const Node *> nodes_in_pattern;
  std::vector<PatternRecord> result;
  for (auto &pattern : patterns) {
    if (pattern.symbol_table.size() != pattern_graph_.nodes.size()) {
      LOG(INFO) << "invalid pattern removed, node size: "
                << pattern.symbol_table.size();
      pattern.set_invalid();
      continue;
    }

    // Is all the nodes in a pattern is free(not included in other patterns).
    bool all_free = true;
    for (auto &item : pattern.symbol_table) {
      if (nodes_in_pattern.count(item.second)) {
        LOG(INFO)
            << "invalid pattern which node in other pattern found, node size: "
            << pattern.symbol_table.size();
        all_free = false;
        break;
      }
    }
    if (!all_free) continue;  // drop current pattern
    // mark the nodes inside current pattern.
    for (auto &item : pattern.symbol_table) {
      nodes_in_pattern.insert(item.second);
    }
    result.emplace_back(pattern);
  }
  return result;
}

void Pattern::Fuse(DataFlowGraph *graph) {
  auto patterns = Match(graph);
  for (const auto &pattern : patterns) {
    handle_(pattern, graph);
  }
}

void Pattern::MarkNodesInPattern(DataFlowGraph *graph) {
  for (auto &node : graph->nodes.nodes()) {
    for (auto &pnode : pattern_graph_.nodes.nodes()) {
      // TODO(Superjomn) check the teller is valid.
      if (dynamic_cast<FusePatternNode *>(&(*pnode))->teller(&(*node))) {
        node->attr("fuse-marker")
            .Int32s()
            .push_back(pnode->attr("fuse-id").Int32());
        pattern_to_node_map_[pnode->attr("fuse-id").Int32()].insert(node.get());
      }
    }
  }
}

struct PairHitHasher {
 public:
  std::size_t operator()(const std::pair<hit_t, hit_t> &x) {
    return std::hash<size_t>()(reinterpret_cast<size_t>(x.first.first)) ^
           std::hash<int32_t>()(x.first.second) ^
           std::hash<size_t>()(reinterpret_cast<size_t>(x.second.first)) ^
           std::hash<int32_t>()(x.second.second);
  }
};

/*
std::vector<std::pair<hit_t, hit_t>> Pattern::Extract2GramPatterns(
    DataFlowGraph *graph) {
  std::vector<std::pair<hit_t, hit_t>> result;
  std::unordered_set<size_t> pattern_set;
  auto hasher = PairHitHasher();

  for (auto &node : graph->nodes.nodes()) {
    auto &in_fuse_ids = node->attr("fuse-marker").Int32s();
    if (in_fuse_ids.empty()) continue;

    for (auto pid : in_fuse_ids) {
      hit_t in_hit({node.get(), pid});

      for (auto *outlink : node->outlinks) {
        auto &out_fuse_ids = outlink->attr("fuse-marker").Int32s();
        for (int32_t out_id : out_fuse_ids) {
          hit_t out_hit({outlink, out_id});
          auto key = hasher(std::make_pair(in_hit, out_hit));
          if (!pattern_set.count(key)) {
            result.emplace_back(in_hit, out_hit);
            pattern_set.insert(key);
          }
        }
      }
    }
  }
  return result;
}
 */

std::vector<PatternRecord> Pattern::ExtractPatterns(DataFlowGraph *graph) {
  // Init pattern with the first node.
  PADDLE_ENFORCE(!edges_.empty(),
                 "a pattern should be defined with at least edges");
  auto *start_point_pnode = edges_.front().first;

  std::vector<PatternRecord> init_patterns;
  for (auto *node : pattern_to_node_map_[start_point_pnode->id()]) {
    init_patterns.emplace_back(start_point_pnode->id(), node);
  }

  // Extend the records from beginning
  auto pre_pattern = init_patterns;
  auto fits_pnode = [&](PatternRecord &record, FusePatternNode *pnode,
                        std::vector<PatternRecord> *pattern_records) {
    for (auto *node : pattern_to_node_map_[pnode->id()]) {
      if (record.Match(node, pnode->id())) {
        pattern_records->emplace_back(record);
        pattern_records->back().MatchOrInsert(node, pnode->id());
      }
    }
  };
  for (auto &edge : edges_) {
    auto *source_pnode = edge.first;
    auto *target_pnode = edge.second;

    std::vector<PatternRecord> patterns0, patterns1;
    // match source pattern node.
    for (auto &pattern : pre_pattern) {
      fits_pnode(pattern, source_pnode, &patterns0);
    }
    for (auto &pattern : patterns0) {
      fits_pnode(pattern, target_pnode, &patterns1);
    }
    // check the nodes in this pattern links to each other.
    patterns1.erase(
        std::remove_if(patterns1.begin(), patterns1.end(),
                       [&](PatternRecord &pattern) {
                         auto *source_node =
                             pattern.symbol_table.at(source_pnode->id());
                         auto *target_node =
                             pattern.symbol_table.at(target_pnode->id());
                         return !Links(*source_node, *target_node);
                       }),
        patterns1.end());
    pre_pattern = std::move(patterns1);
  }
  return pre_pattern;
}

}  // namespace fuse
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
