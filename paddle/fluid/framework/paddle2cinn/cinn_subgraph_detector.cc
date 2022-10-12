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

#include "paddle/fluid/framework/paddle2cinn/cinn_subgraph_detector.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

std::unordered_set<Node*> GetProducerOps(Node* node) {
  CHECK(node->IsOp());
  std::unordered_set<Node*> producers;

  for (auto input_var : node->inputs) {
    CHECK(input_var->IsVar());
    for (auto input_op : input_var->inputs) {
      CHECK(input_op->IsOp());
      producers.insert(input_op);
    }
  }

  return producers;
}

std::unordered_set<Node*> GetConsumerOps(Node* node) {
  CHECK(node->IsOp());
  std::unordered_set<Node*> consumers;

  for (auto output_var : node->outputs) {
    CHECK(output_var->IsVar());
    for (auto output_op : output_var->outputs) {
      CHECK(output_op->IsOp());
      consumers.insert(output_op);
    }
  }

  return consumers;
}

void CinnSubGraph::Insert(Node* op) {
  nodes.push_back(op);
  node_set.insert(op);

  auto producers = GetProducerOps(op);
  for (auto producer : producers) {
    input_nodes.insert(producer);
  }
  input_nodes.erase(op);
}

void CinnSubgraphDetector::DoOpFusion() {
  // sort node from input to output
  for (auto& node : TopologicalSort(*graph_)) {
    if (node.IsVar()) {
      continue;
    }
    nodes_.push_back(&node);
  }
  // reverse from output to input
  std::reverse(nodes_.begin(), nodes_.end());

  // do fusion
  for (auto* node : nodes_) {
    auto subgraph =
        subgraph_map_.count(node)
            ? subgraph_map_[node]
            : std::make_shared<CinnSubGraph>(node, node_classifier_(node));
    if (!subgraph_map_.count(node)) {
      subgraph_map_[node] = subgraph;
    }
    auto producers = GetProducerOps(node);

    for (auto producer : producers) {
      if (node_classifier_(producer) != subgraph->substitute) {
        continue;
      }

      bool can_fused = true;
      auto consumers = GetConsumerOps(producer);
      for (auto consumer : consumers) {
        if (!subgraph->node_set.count(consumer)) {
          can_fused = false;
          break;
        }
      }

      if (!can_fused) {
        continue;
      }

      // fuse producer to sub-graph
      subgraph->Insert(producer);
      subgraph_map_[producer] = subgraph;
    }
  }
}

void CinnSubgraphDetector::BuildSubGraph() {
  std::unordered_set<CinnSubGraph*> subgraph_set;
  for (auto node : nodes_) {
    CHECK(subgraph_map_.count(node));
    auto& subgraph = subgraph_map_[node];
    if (subgraph_set.count(subgraph.get())) {
      continue;
    }

    subgraph_set.insert(subgraph.get());
    subgraph_list_.push_back(subgraph);
  }

  for (auto& subgraph : subgraph_list_) {
    for (auto& input_node : subgraph->input_nodes) {
      CHECK(subgraph_map_.count(input_node));
      auto& producer = subgraph_map_[input_node];
      subgraph->producers.insert(producer);
      producer->consumers.insert(subgraph);
    }
  }

  // init group depth.
  for (auto& subgraph : subgraph_list_) {
    for (auto& consumer : subgraph->consumers) {
      // update depth.
      subgraph->depth = std::max(subgraph->depth, consumer->depth + 1);
    }
    subgraph->max_depth = subgraph->depth;
    subgraph->min_depth = subgraph->depth;
  }

  // reverse to keep fusion group in order.
  std::reverse(subgraph_list_.begin(), subgraph_list_.end());
}

void CinnSubgraphDetector::DoSubGraphFusion() {
  while (true) {
    bool update = false;
    for (auto& subgraph : subgraph_list_) {
      // sub graph is not substitute
      if (!subgraph->substitute) {
        continue;
      }
      // do fusion
      update |= FuseSubGraph(&subgraph);
    }
    if (!update) {
      break;
    }
  }
}

bool CinnSubgraphDetector::FuseSubGraph(CinnSubGraphPtr* subgraph_ptr) {
  auto producer = *subgraph_ptr;
  auto& consumers = producer->consumers;
  std::vector<CinnSubGraphPtr> candidates;
  for (auto& consumer : consumers) {
    if (!consumer->substitute) {
      continue;
    }
    // fast depency check.
    if (IsDependencySimplify(producer, consumer, consumers)) {
      continue;
    }
    // global depency check.
    if (IsDependency(producer, consumer, consumers)) {
      continue;
    }

    candidates.push_back(consumer);
  }

  if (!candidates.size()) {
    return false;
  }

  // fuse candidate to producer
  for (auto& candidate : candidates) {
    candidate->substitute = false;

    // merge nodes
    producer->nodes.insert(producer->nodes.end(),
                           candidate->nodes.begin(),
                           candidate->nodes.end());
    producer->node_set.insert(candidate->node_set.begin(),
                              candidate->node_set.end());

    // update bound for check depency
    producer->max_depth = std::max(producer->max_depth, candidate->max_depth);
    producer->min_depth = std::min(producer->min_depth, candidate->min_depth);

    // merge producer/consumer
    producer->producers.insert(candidate->producers.begin(),
                               candidate->producers.end());
    producer->consumers.insert(candidate->consumers.begin(),
                               candidate->consumers.end());
    // update producers's consumer
    for (auto& tmp : candidate->producers) {
      if (tmp.get() == producer.get()) {
        continue;
      }
      tmp->consumers.insert(producer);
      tmp->consumers.erase(candidate);
    }
    // update consumers's producer
    for (auto& tmp : candidate->consumers) {
      tmp->producers.insert(producer);
      tmp->producers.erase(candidate);
    }

    // remove candicate in producer/consumer
    producer->producers.erase(candidate);
    producer->consumers.erase(candidate);

    // merge input nodes
    producer->input_nodes.insert(candidate->input_nodes.begin(),
                                 candidate->input_nodes.end());
  }

  // remove input nodes that is in node set
  auto input_nodes = producer->input_nodes;
  for (auto input_node : input_nodes) {
    if (producer->node_set.count(input_node)) {
      producer->input_nodes.erase(input_node);
    }
  }

  // remove producer from set.
  producer->producers.erase(producer);
  producer->consumers.erase(producer);

  return true;
}

bool CinnSubgraphDetector::IsDependency(
    const CinnSubGraphPtr& producer_g,
    const CinnSubGraphPtr& consumer,
    const std::unordered_set<CinnSubGraphPtr>& consumers) {
  std::queue<CinnSubGraphPtr> candidates;
  candidates.push(consumer);

  std::unordered_set<CinnSubGraphPtr> visited_set;
  while (!candidates.empty()) {
    auto& candidate = candidates.front();
    candidates.pop();
    for (auto& producer : candidate->producers) {
      if (producer.get() == producer_g.get()) {
        continue;
      }
      if (consumers.count(producer)) {
        return true;
      }
      if (!visited_set.count(producer)) {
        visited_set.insert(producer);
        candidates.push(producer);
      }
    }
  }
  return false;
}

bool CinnSubgraphDetector::IsDependencySimplify(
    const CinnSubGraphPtr& producer_g,
    const CinnSubGraphPtr& consumer,
    const std::unordered_set<CinnSubGraphPtr>& consumers) {
  std::queue<CinnSubGraphPtr> candidates;
  candidates.push(consumer);
  // check upper bound.
  int check_upper_depth = producer_g->max_depth;
  std::unordered_set<CinnSubGraphPtr> visited_set;
  while (!candidates.empty()) {
    auto& candidate = candidates.front();
    candidates.pop();
    for (auto& producer : candidate->producers) {
      if (producer.get() == producer_g.get()) {
        continue;
      }
      if (producer->min_depth > check_upper_depth) {
        continue;
      }
      if (consumers.count(producer)) {
        return true;
      }
      if (!visited_set.count(producer)) {
        visited_set.insert(producer);
        candidates.push(producer);
      }
    }
  }
  return false;
}

std::vector<std::vector<Node*>> CinnSubgraphDetector::operator()() {
  DoOpFusion();
  BuildSubGraph();
  DoSubGraphFusion();

  std::vector<std::vector<Node*>> clusters;
  for (auto& subgraph : subgraph_list_) {
    if (!subgraph->substitute) {
      continue;
    }
    clusters.push_back(subgraph->nodes);
  }

  return clusters;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
