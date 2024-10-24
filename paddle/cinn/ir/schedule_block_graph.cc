// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/schedule_block_graph.h"
#include "paddle/cinn/common/dfs_topo_walker.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

ScheduleBlockNode::ScheduleBlockNode(Expr block, const IRSchedule& ir_sch)
    : ir_sch_(ir_sch) {
  PADDLE_ENFORCE_NE(
      block.As<ScheduleBlockRealize>(),
      nullptr,
      ::common::errors::InvalidArgument("Expr is not a ScheduleBlockRealize."));
  id_ = block.As<ScheduleBlockRealize>()
            ->schedule_block.As<ScheduleBlock>()
            ->name;
  VLOG(5) << "create schedule_block node: " << id_;
}

Expr ScheduleBlockNode::Block() const { return ir_sch_.GetBlock(id_); }

std::vector<Expr> ScheduleBlockNode::GetLoops() const {
  return ir_sch_.GetLoops(id_);
}

bool EdgeCompare(const cinn::common::Shared<cinn::common::GraphEdge>& a,
                 const cinn::common::Shared<cinn::common::GraphEdge>& b) {
  PADDLE_ENFORCE_NOT_NULL(
      a.get(), ::common::errors::InvalidArgument("The a is nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      b.get(), ::common::errors::InvalidArgument("The b is nullptr."));
  return a->index() < b->index();
}
std::vector<cinn::common::Shared<cinn::common::GraphEdge>>
ScheduleBlockNode::OrderedInLinks() const {
  std::vector<cinn::common::Shared<cinn::common::GraphEdge>> ordered_links;
  for (auto& in_edge : this->inlinks()) {
    ordered_links.push_back(in_edge);
    PADDLE_ENFORCE_GE(in_edge->index(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The index of a node's inlinks should be >= 0!"));
  }
  std::sort(ordered_links.begin(), ordered_links.end(), EdgeCompare);
  return ordered_links;
}

std::vector<cinn::common::Shared<cinn::common::GraphEdge>>
ScheduleBlockNode::OrderedOutLinks() const {
  std::vector<cinn::common::Shared<cinn::common::GraphEdge>> ordered_links;
  for (auto& out_edge : this->outlinks()) {
    ordered_links.push_back(out_edge);
    PADDLE_ENFORCE_GE(
        out_edge->index(),
        0,
        ::common::errors::InvalidArgument("The index of a node's outlinks "
                                          "should be >= 0!"));
  }
  std::sort(ordered_links.begin(), ordered_links.end(), EdgeCompare);
  return ordered_links;
}

std::vector<ScheduleBlockNode*> ScheduleBlockNode::Producers() const {
  std::vector<ScheduleBlockNode*> producers;
  for (const auto& link : this->OrderedInLinks()) {
    producers.push_back(dynamic_cast<ScheduleBlockNode*>(link->source()));
  }
  return producers;
}
std::vector<ScheduleBlockNode*> ScheduleBlockNode::Consumers() const {
  std::vector<ScheduleBlockNode*> consumers;
  for (const auto& link : this->OrderedOutLinks()) {
    consumers.push_back(dynamic_cast<ScheduleBlockNode*>(link->sink()));
  }
  return consumers;
}

ScheduleBlockGraph::ScheduleBlockGraph(const IRSchedule& ir_sch) {
  Update(ir_sch);
}

void ScheduleBlockGraph::Update(const IRSchedule& ir_sch) {
  nodes_.clear();
  registry_.clear();
  std::vector<Expr> all_blocks = ir_sch.GetAllBlocks();
  Expr root_block = ir_sch.GetRootBlock(all_blocks[0]);
  for (Expr block : all_blocks) {
    PADDLE_ENFORCE_NE(block.As<ScheduleBlockRealize>(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "Expr is not a ScheduleBlockRealize."));
    std::string id = block.As<ScheduleBlockRealize>()
                         ->schedule_block.As<ScheduleBlock>()
                         ->name;
    if (id == "root") {
      continue;
    }
    ScheduleBlockNode* node = new ScheduleBlockNode(block, ir_sch);
    RegisterNode(id, node);
    VLOG(5) << "register schedule_block node: " << id;
    block_ids_in_order_.push_back(id);

    std::vector<Expr> producers = GetProducers(block, root_block);
    for (Expr producer : producers) {
      PADDLE_ENFORCE_NE(producer.As<ScheduleBlockRealize>(),
                        nullptr,
                        ::common::errors::InvalidArgument(
                            "Expr is not a ScheduleBlockRealize."));
      std::string producer_id = producer.As<ScheduleBlockRealize>()
                                    ->schedule_block.As<ScheduleBlock>()
                                    ->name;
      ScheduleBlockNode* producer_node = RetrieveNode(producer_id);
      PADDLE_ENFORCE_NOT_NULL(
          producer_node,
          ::common::errors::InvalidArgument(
              "producer node: %s does not exist in the graph", producer_id));
      producer_node->Controls(node);
      for (const std::string& upstream_node_id :
           producer_node->UpstreamNodes()) {
        node->AddUpstreamNode(upstream_node_id);
      }
      node->AddUpstreamNode(producer_id);
    }

    for (const std::string& upstream_node_id : node->UpstreamNodes()) {
      RetrieveNode(upstream_node_id)->AddDownstreamNode(id);
    }
  }
}

std::vector<ScheduleBlockNode*> ScheduleBlockGraph::StartPoints() {
  std::vector<ScheduleBlockNode*> res;
  for (cinn::common::GraphNode* node : nodes()) {
    if (node->inlinks().empty()) {
      res.push_back(dynamic_cast<ScheduleBlockNode*>(node));
    }
  }
  return res;
}

std::vector<ScheduleBlockNode*> ScheduleBlockGraph::EndPoints() {
  std::vector<ScheduleBlockNode*> res;
  for (cinn::common::GraphNode* node : nodes()) {
    if (node->outlinks().empty()) {
      res.push_back(dynamic_cast<ScheduleBlockNode*>(node));
    }
  }
  return res;
}

void ScheduleBlockGraph::NodesWalk(const NodeHandlerType& NodeHandler) {
  for (cinn::common::GraphNode* node : nodes()) {
    ScheduleBlockNode* cur_node = dynamic_cast<ScheduleBlockNode*>(node);
    NodeHandler(cur_node);
  }
}

void ScheduleBlockGraph::DFSTopoWalk(const NodeHandlerType& NodeHandler,
                                     bool is_reverse) {
  auto VisitPreNodes = [&](const ScheduleBlockNode* node,
                           const NodeHandlerType& PreNodeHandler) {
    std::vector<ScheduleBlockNode*> pre_nodes =
        is_reverse ? node->Consumers() : node->Producers();
    for (ScheduleBlockNode* pre_node : pre_nodes) {
      PreNodeHandler(pre_node);
    }
  };
  auto VisitNextNodes = [&](const ScheduleBlockNode* node,
                            const NodeHandlerType& NextNodeHandler) {
    std::vector<ScheduleBlockNode*> next_nodes =
        is_reverse ? node->Producers() : node->Consumers();
    for (ScheduleBlockNode* next_node : next_nodes) {
      NextNodeHandler(next_node);
    }
  };
  cinn::common::DfsTopoWalker<ScheduleBlockNode*> walker(VisitPreNodes,
                                                         VisitNextNodes);
  std::vector<ScheduleBlockNode*> starts =
      is_reverse ? EndPoints() : StartPoints();
  walker(starts.begin(), starts.end(), NodeHandler);
}

}  // namespace ir
}  // namespace cinn
