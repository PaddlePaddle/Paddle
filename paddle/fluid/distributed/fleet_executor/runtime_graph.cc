// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace distributed {
namespace {

using OperatorBase = RuntimeGraph::OperatorBase;
using OpRole = paddle::framework::OpRole;
using OpRegistry = paddle::framework::OpRegistry;
using ProgramDesc = paddle::framework::ProgramDesc;

bool IsForward(int32_t op_role) {
  return (op_role == static_cast<int32_t>(OpRole::kForward)) ||
         (op_role == (static_cast<int32_t>(OpRole::kForward) |
                      static_cast<int32_t>(OpRole::kLoss)));
}

bool IsLRSched(int32_t op_role) {
  return op_role == static_cast<int32_t>(OpRole::kLRSched);
}

bool IsBackward(int32_t op_role) {
  return (op_role == static_cast<int32_t>(OpRole::kBackward)) ||
         (op_role == (static_cast<int32_t>(OpRole::kBackward) |
                      static_cast<int32_t>(OpRole::kLoss)));
}

bool IsOptimize(int32_t op_role) {
  return op_role == static_cast<int32_t>(OpRole::kOptimize);
}

struct DistCoord {
  int32_t dp_idx;
  int32_t pp_idx;
  int32_t mp_idx;
};

class DistCoordSys final {
 public:
  DistCoordSys(int32_t dp_degree, int32_t pp_degree, int32_t mp_degree)
      : dp_degree_(dp_degree), pp_degree_(pp_degree), mp_degree_(mp_degree) {}
  DistCoord RankToCoord(int64_t rank) const;
  int64_t CoordToRank(const DistCoord& coord) const;

 private:
  DISABLE_COPY_AND_ASSIGN(DistCoordSys);
  bool InvalidCoord(const DistCoord& coord) const;
  int32_t dp_degree_;
  int32_t pp_degree_;
  int32_t mp_degree_;
};

DistCoord DistCoordSys::RankToCoord(int64_t rank) const {
  DistCoord coord;
  coord.mp_idx = rank % mp_degree_;
  rank /= mp_degree_;
  coord.pp_idx = rank % pp_degree_;
  rank /= pp_degree_;
  coord.dp_idx = rank % dp_degree_;
  return coord;
}

int64_t DistCoordSys::CoordToRank(const DistCoord& coord) const {
  if (InvalidCoord(coord)) {
    return -1;
  }
  return coord.dp_idx * pp_degree_ * mp_degree_ + coord.pp_idx * mp_degree_ +
         coord.mp_idx;
}

bool DistCoordSys::InvalidCoord(const DistCoord& coord) const {
  return coord.mp_idx < 0 || coord.mp_idx >= mp_degree_ || coord.pp_idx < 0 ||
         coord.pp_idx >= pp_degree_ || coord.dp_idx < 0 ||
         coord.dp_idx >= dp_degree_;
}

}  // namespace

std::vector<OpRole> RuntimeGraph::functionality_order = {
    OpRole::kLRSched, OpRole::kForward, OpRole::kBackward, OpRole::kOptimize};

RuntimeGraph::RuntimeGraph(const ProgramDesc& program,
                           const FleetExecutorDesc& exe_desc)
    : exe_desc_(exe_desc) {
  if (exe_desc.grain() == "coarse") {
    SplitProgramBasedFunctionality(program);
    AssignTaskToIntercepter();
    FakeDependence();
    FakeRuntimeInfo();
  }
}

void RuntimeGraph::SplitProgramBasedFunctionality(const ProgramDesc& program) {
  for (const auto& op_desc : program.Block(0).AllOps()) {
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
  }
  std::unordered_map<int32_t, std::vector<OperatorBase*>> role_to_ops;
  for (const auto& op : ops_) {
    int32_t op_role = op->Attr<int32_t>("op_role");
    OpRole new_op_role;
    if (IsLRSched(op_role)) {
      new_op_role = OpRole::kLRSched;
    } else if (IsForward(op_role)) {
      new_op_role = OpRole::kForward;
    } else if (IsBackward(op_role)) {
      new_op_role = OpRole::kBackward;
    } else if (IsOptimize(op_role)) {
      new_op_role = OpRole::kOptimize;
    } else {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "The op %s is None of LRSched, Forward, Backward or Optimize.",
          op->Type()));
    }
    int32_t new_op_role_id = static_cast<int32_t>(new_op_role);
    if (role_to_ops.find(new_op_role_id) == role_to_ops.end()) {
      role_to_ops.insert({new_op_role_id, {}});
    }
    role_to_ops.at(new_op_role_id).emplace_back(op.get());
  }
  int64_t cur_rank = exe_desc_.cur_rank();
  DistCoordSys coord_sys(exe_desc_.dp_degree(), exe_desc_.pp_degree(),
                         exe_desc_.mp_degree());
  const auto& coord = coord_sys.RankToCoord(cur_rank);
  int pipeline_stage = coord.pp_idx;
  int64_t num_pipeline_stages = exe_desc_.pp_degree();
  // TODO(fleet_executor dev): start up steps should be a config `num_slots`
  int64_t start_up_steps = num_pipeline_stages - pipeline_stage - 1;
  int64_t num_micro_batches = exe_desc_.num_micro_batches();
  int64_t task_id = cur_rank * functionality_order.size();
  for (std::size_t i = 0; i < functionality_order.size(); ++i) {
    OpRole role = functionality_order[i];
    int32_t role_id = static_cast<int64_t>(role);
    int64_t max_run_times = num_micro_batches;
    int64_t max_slot_nums = start_up_steps;
    if (IsLRSched(role_id) || IsOptimize(role_id)) {
      max_run_times = 1;
      max_slot_nums = 1;
    }
    if (role_to_ops.find(role_id) == role_to_ops.end()) {
      task_nodes_.emplace_back(TaskNode::CreateEmptyTaskNode(
          role_id, cur_rank, task_id, max_run_times, max_slot_nums));
    } else {
      task_nodes_.emplace_back(
          TaskNode::CreateTaskNode(role_id, role_to_ops.at(role_id), cur_rank,
                                   task_id, max_run_times, max_slot_nums));
    }
    ++task_id;
  }
}

void RuntimeGraph::FakeDependence() {
  int64_t cur_rank = exe_desc_.cur_rank();
  DistCoordSys coord_sys(exe_desc_.dp_degree(), exe_desc_.pp_degree(),
                         exe_desc_.mp_degree());
  const auto& coord = coord_sys.RankToCoord(cur_rank);
  DistCoord upstream_coord = coord, downstream_coord = coord;
  upstream_coord.pp_idx -= 1;
  downstream_coord.pp_idx += 1;
  int64_t pp_upstream = coord_sys.CoordToRank(upstream_coord);
  int64_t pp_downstream = coord_sys.CoordToRank(downstream_coord);
  int32_t num_of_functionality = functionality_order.size();
  // lr -> forward -> backward -> optimize
  //         |          |
  // lr -> forward -> backward -> optimize
  for (std::size_t i = 0; i < task_nodes_.size(); ++i) {
    if (i != 0) {
      task_nodes_[i]->AddUpstreamTask(cur_rank * num_of_functionality + i - 1);
    }
    if (i != task_nodes_.size() - 1) {
      task_nodes_[i]->AddDownstreamTask(cur_rank * num_of_functionality + i +
                                        1);
    }
    if (IsForward(task_nodes_[i]->role())) {
      if (pp_upstream != -1) {
        task_nodes_[i]->AddUpstreamTask(pp_upstream * num_of_functionality + i);
      }
      if (pp_downstream != -1) {
        task_nodes_[i]->AddDownstreamTask(pp_downstream * num_of_functionality +
                                          i);
      }
    } else if (IsBackward(task_nodes_[i]->role())) {
      if (pp_downstream != -1) {
        task_nodes_[i]->AddUpstreamTask(pp_downstream * num_of_functionality +
                                        i);
      }
      if (pp_upstream != -1) {
        task_nodes_[i]->AddDownstreamTask(pp_upstream * num_of_functionality +
                                          i);
      }
    }
  }
}

void RuntimeGraph::AssignTaskToIntercepter() {
  for (const auto& task : task_nodes_) {
    int64_t intercepter_id = task->task_id();
    if (intercepter_id_to_node_.find(intercepter_id) !=
        intercepter_id_to_node_.end()) {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "Repeated intercepter id: %d", intercepter_id));
    }
    intercepter_id_to_node_.insert({intercepter_id, task.get()});
  }
}

void RuntimeGraph::FakeRuntimeInfo() {
  int64_t nrank = exe_desc_.cluster_info().size();
  int32_t num_of_functionality = functionality_order.size();
  for (int64_t i = 0; i < nrank; ++i) {
    for (int32_t j = 0; j < num_of_functionality; ++j) {
      int64_t intercepter_id = i * num_of_functionality + j;
      intercepter_id_to_rank_.insert({intercepter_id, i});
    }
  }
}

std::string RuntimeGraph::DebugString() const {
  std::ostringstream os;
  os << "\nRuntime Graph Debug: \n";
  for (const auto& task : task_nodes_) {
    os << task->DebugString();
    os << "\n";
  }
  return os.str();
}

}  // namespace distributed
}  // namespace paddle
