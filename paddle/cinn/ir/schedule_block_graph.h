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
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_schedule.h"

using Group = cinn::hlir::framework::Graph::Group;

namespace cinn {
namespace ir {

class ScheduleBlockNode : public common::GraphNode {
 public:
  ScheduleBlockNode(Expr block, const IRSchedule& ir_sch);

  std::string id() const { return id_; }
  Expr Block() const;
  std::vector<Expr> ControlStmts() const;
  std::unordered_set<std::string> UpstreamNodes() const {
    return upstream_nodes_;
  }
  std::unordered_set<std::string> DownstreamNodes() const {
    return downstream_nodes_;
  }
  std::vector<ScheduleBlockNode*> Producers() const;
  std::vector<ScheduleBlockNode*> Consumers() const;

  void Update();

  void AddUpstreamNode(const std::string& node_id) {
    upstream_nodes_.insert(node_id);
  }
  void AddDownstreamNode(const std::string& node_id) {
    downstream_nodes_.insert(node_id);
  }

 private:
  std::vector<common::Shared<common::GraphEdge>> OrderedInLinks() const;
  std::vector<common::Shared<common::GraphEdge>> OrderedOutLinks() const;

 private:
  std::string id_;
  std::unordered_set<std::string> upstream_nodes_;
  std::unordered_set<std::string> downstream_nodes_;
  const IRSchedule& ir_sch_;
};

class ScheduleBlockGraph : public common::Graph {
 public:
  explicit ScheduleBlockGraph(const IRSchedule& ir_sch);

  void Update(const IRSchedule& ir_sch);

  ScheduleBlockNode* RetrieveNode(const std::string& id) {
    return dynamic_cast<ScheduleBlockNode*>(common::Graph::RetrieveNode(id));
  }

  std::list<std::string> BlockIdsInOrder() const { return block_ids_in_order_; }

  std::vector<ScheduleBlockNode*> StartPoints();
  std::vector<ScheduleBlockNode*> EndPoints();

  using NodeHandlerType = std::function<void(ScheduleBlockNode*)>;
  void NodesWalk(const NodeHandlerType& NodeHandler);
  void DFSTopoWalk(const NodeHandlerType& NodeHandler, bool is_reverse = true);

  ScheduleBlockNode* GetGlobalMasterNode() const;

 private:
  std::list<std::string> block_ids_in_order_;
};

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
