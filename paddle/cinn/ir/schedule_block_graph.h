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

#pragma once

#include <list>
#include <stack>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace ir {

// Node in units of ScheduleBlock.
class ScheduleBlockNode : public cinn::common::GraphNode {
 public:
  ScheduleBlockNode(Expr block, const IRSchedule& ir_sch);

  // Get the id of this node, which is same as the name of ScheduleBlock.
  std::string id() const { return id_; }

  // Get the ScheduleBlockRealize expr
  Expr Block() const;

  // Get all control stmts containing the schedule_block, now only the For node
  // is being considered.
  std::vector<Expr> GetLoops() const;

  // Get all the upstream nodes that this node depends on.
  std::unordered_set<std::string> UpstreamNodes() const {
    return upstream_nodes_;
  }

  // Get all downstream nodes that depend on this node.
  std::unordered_set<std::string> DownstreamNodes() const {
    return downstream_nodes_;
  }

  // Get the producer node that this node directly depends on
  std::vector<ScheduleBlockNode*> Producers() const;

  // Get consumer nodes that directly depend on this node.
  std::vector<ScheduleBlockNode*> Consumers() const;

  void AddUpstreamNode(const std::string& node_id) {
    upstream_nodes_.insert(node_id);
  }
  void AddDownstreamNode(const std::string& node_id) {
    downstream_nodes_.insert(node_id);
  }

 private:
  std::vector<cinn::common::Shared<cinn::common::GraphEdge>> OrderedInLinks()
      const;
  std::vector<cinn::common::Shared<cinn::common::GraphEdge>> OrderedOutLinks()
      const;

 private:
  std::string id_;
  std::unordered_set<std::string> upstream_nodes_;
  std::unordered_set<std::string> downstream_nodes_;
  const IRSchedule& ir_sch_;
};

// Graph in units of ScheduleBlockNode, each node corresponds to a ScheduleBlock
// in IR.
class ScheduleBlockGraph : public cinn::common::Graph {
 public:
  explicit ScheduleBlockGraph(const IRSchedule& ir_sch);

  // Update graph information according to the new IRSchedule.
  void Update(const IRSchedule& ir_sch);

  // Retrieve a node in the graph by id, the id is same as the name of
  // ScheduleBlock.
  ScheduleBlockNode* RetrieveNode(const std::string& id) {
    return dynamic_cast<ScheduleBlockNode*>(
        cinn::common::Graph::RetrieveNode(id));
  }

  // Get all block name in order,
  // this sequence may become invalid after some schedule operations,
  // and an Update() operation is required.
  std::list<std::string> BlockIdsInOrder() const { return block_ids_in_order_; }

  // Get all nodes without input node.
  std::vector<ScheduleBlockNode*> StartPoints();

  // Get all nodes without output node.
  std::vector<ScheduleBlockNode*> EndPoints();

  // Function used to define the operations to be performed on each node.
  using NodeHandlerType = std::function<void(ScheduleBlockNode*)>;

  // Walk through each node
  // and perform some operations defined by NodeHandler on it.
  void NodesWalk(const NodeHandlerType& NodeHandler);

  // Walk through each node topological dfs topo order
  // and perform some operations defined by NodeHandler on it.
  void DFSTopoWalk(const NodeHandlerType& NodeHandler, bool is_reverse = true);

 private:
  std::list<std::string> block_ids_in_order_;
};

/**
 * The mutator used to construct the order of blocks and their control
 * statements
 *
 * Example:
 * for0:
 *   for1:
 *     block0
 *     block1
 *   block2
 *   for2:
 *     block3
 *     block4
 *
 * the result is:
 *   [0]: for0
 *   [0, 0]: for1
 *   [0, 0, 0]: block0
 *   [0, 0, 1]: block1
 *   [0, 1]: block2
 *   [0, 2]: for2
 *   [0, 2, 0]: block3
 *   [0, 2, 1]: block4
 */
struct BlockOrderConstructor : public IRMutator<Expr*> {
  std::map<std::vector<int>, Expr> operator()(ir::Expr* expr) {
    IRMutator::Visit(expr, expr);
    return block_order_with_ctrl_structure_;
  }

 private:
  void Visit(const For* x, Expr* op) {
    if (global_idx_.empty() ||
        block_order_with_ctrl_structure_.rbegin()->first.size() ==
            global_idx_.size()) {
      cur_idx_ = -1;
    }
    global_idx_.push_back(++cur_idx_);
    block_order_with_ctrl_structure_.insert(std::make_pair(global_idx_, *op));
    IRMutator<Expr*>::Visit(x, op);
    cur_idx_ = global_idx_.back();
    global_idx_.pop_back();
  }

  void Visit(const ScheduleBlockRealize* x, Expr* op) {
    if (global_idx_.empty() ||
        block_order_with_ctrl_structure_.rbegin()->first.size() ==
            global_idx_.size()) {
      cur_idx_ = -1;
    }
    global_idx_.push_back(++cur_idx_);
    block_order_with_ctrl_structure_.insert(std::make_pair(global_idx_, *op));
    if (x->schedule_block.As<ScheduleBlock>()->name.substr(0, 4) == "root") {
      IRMutator<Expr*>::Visit(x, op);
    }
    global_idx_.pop_back();
  }

  void Visit(const IfThenElse* x, Expr* op) {
    if (global_idx_.empty() ||
        block_order_with_ctrl_structure_.rbegin()->first.size() ==
            global_idx_.size()) {
      cur_idx_ = -1;
    }
    global_idx_.push_back(++cur_idx_);
    block_order_with_ctrl_structure_.insert(std::make_pair(global_idx_, *op));
    IRMutator<Expr*>::Visit(x, op);
    cur_idx_ = global_idx_.back();
    global_idx_.pop_back();
  }

 private:
  int cur_idx_;
  std::vector<int> global_idx_;
  std::map<std::vector<int>, Expr> block_order_with_ctrl_structure_;
};

}  // namespace ir
}  // namespace cinn
