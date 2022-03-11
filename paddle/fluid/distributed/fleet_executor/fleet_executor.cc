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
#include <algorithm>

#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace distributed {

FleetExecutor::FleetExecutor(const std::string& exe_desc_str) {
  bool parse_flag = exe_desc_.ParseFromString(exe_desc_str);
  PADDLE_ENFORCE(parse_flag, platform::errors::PreconditionNotMet(
                                 "Error occurs while parsing string to proto"));
  // Message bus will be created and inited only once
  GlobalVal<MessageBus>::Create();
  InitMessageBus();
}

FleetExecutor::FleetExecutor(const FleetExecutorDesc& exe_desc)
    : exe_desc_(exe_desc) {
  // Message bus will be created and inited only once
  GlobalVal<MessageBus>::Create();
  InitMessageBus();
}

FleetExecutor::~FleetExecutor() {
  for (const auto& carrier_id : carrier_ids_) {
    GlobalMap<std::string, Carrier>::Get(carrier_id)->Release();
  }
}

void FleetExecutor::Init(
    const std::string& carrier_id, const framework::ProgramDesc& program_desc,
    framework::Scope* scope, const platform::Place& place,
    int64_t num_micro_batches, const std::vector<TaskNode*>& task_nodes,
    const std::unordered_map<int64_t, int64_t>& task_id_to_rank,
    const std::vector<std::string>& inference_root_scope_vars) {
  PADDLE_ENFORCE_GT(task_nodes.size(), 0,
                    platform::errors::InvalidArgument(
                        "Fleet executor is inited with empty task node"));
  // TODO(fleet_exe devs): the unused_vars should be got from run time graph
  std::vector<std::unique_ptr<framework::OperatorBase>> ops;
  for (auto task_node : task_nodes) {
    for (auto op : task_node->ops()) {
      ops.emplace_back(std::unique_ptr<framework::OperatorBase>(op));
    }
  }
  auto unused_vars = framework::GetUnusedVars(program_desc.Block(0), ops, {});
  // NOTE: For inference, the vars in inference_root_scope_vars
  // shouldn't be deleted during inf, for that they may be the result of the
  // inf. If they are GCed, it will cause error during ZeroCopy the result.
  std::vector<const framework::OperatorBase*> changed_ops;
  for (auto pair : unused_vars) {
    const framework::OperatorBase* op = pair.first;
    std::vector<std::string> unused = pair.second;
    for (auto name : inference_root_scope_vars) {
      auto iter = std::find(unused.begin(), unused.end(), name);
      if (iter != unused.end()) {
        VLOG(3) << "Removing var: [" << name
                << "] from the unused vars list of op: [" << op->Type() << "]";
        unused.erase(iter);
        if (std::find(changed_ops.begin(), changed_ops.end(), op) ==
            changed_ops.end()) {
          // record the op whose unused vars have been updated
          changed_ops.emplace_back(op);
        }
      }
    }
    // update the unused vars list in the map
    unused_vars[op] = unused;
  }
  for (auto op : changed_ops) {
    auto iter = unused_vars.find(op);
    if (iter->second.empty()) {
      // remove those ops in the map that have empty unused vars list
      VLOG(3) << "Removing op: [" << op->Type() << "] from unused_vars map.";
      unused_vars.erase(iter);
    }
  }
  runtime_graph_ = std::make_shared<RuntimeGraph>();
  std::unordered_map<int64_t, TaskNode*> interceptor_id_to_task;
  for (auto task_node : task_nodes) {
    task_node->SetUnusedVars(unused_vars);
    int64_t interceptor_id = task_node->task_id();
    interceptor_id_to_task.emplace(interceptor_id, task_node);
  }
  runtime_graph_->SetInterceptorIdToRank(task_id_to_rank);
  runtime_graph_->SetInterceptorIdToNode(interceptor_id_to_task);
  for (auto& unique_op : ops) {
    unique_op.release();
  }
  VLOG(5) << runtime_graph_->DebugString();
  Carrier* carrier =
      GlobalMap<std::string, Carrier>::Create(carrier_id, carrier_id);
  carrier_ids_.insert(carrier_id);
  // Set current running carrier
  GlobalVal<std::string>::Set(new std::string(carrier_id));
  InitCarrier(carrier, scope, place, num_micro_batches, program_desc,
              inference_root_scope_vars);
  GlobalVal<MessageBus>::Get()->Barrier();
}

void FleetExecutor::InitCarrier(
    Carrier* carrier, framework::Scope* scope, const platform::Place& place,
    int64_t num_micro_batches, const framework::ProgramDesc& program_desc,
    const std::vector<std::string>& inference_root_scope_vars) {
  carrier->Init(exe_desc_.cur_rank(), runtime_graph_->interceptor_id_to_rank(),
                runtime_graph_->interceptor_id_to_node(), program_desc, scope,
                num_micro_batches, place, inference_root_scope_vars);
}

void FleetExecutor::InitMessageBus() {
  std::stringstream ss;
  ss << "\nThe DNS table of the message bus is: \n";
  int64_t cur_rank = exe_desc_.cur_rank();
  std::unordered_map<int64_t, std::string> rank_to_addr;
  std::string addr;
  for (const auto& rank_info : exe_desc_.cluster_info()) {
    // init the dns map
    int64_t rank = rank_info.rank();
    std::string ip_port = rank_info.ip_port();
    ss << rank << "\t->\t" << ip_port << "\n";
    rank_to_addr.insert(std::make_pair(rank, ip_port));
    if (rank == cur_rank) {
      addr = ip_port;
    }
  }
  if (addr == "") {
    PADDLE_ENFORCE_EQ(
        rank_to_addr.size(), 1,
        platform::errors::NotFound("Empty address is not valid for "
                                   "paddle.distributed.launch method."));
    PADDLE_ENFORCE_EQ(
        cur_rank, 0,
        platform::errors::NotFound("Address is empty but cur rank is not 0."));
  }
  VLOG(3) << "Current rank is " << cur_rank << " and the ip_port is "
          << (addr == "" ? "empty" : addr) << ".";
  VLOG(3) << "The number of ranks are "
          << (rank_to_addr.size() == 0 ? 1 : rank_to_addr.size()) << ".";
  VLOG(5) << ss.str();
  GlobalVal<MessageBus>::Get()->Init(cur_rank, rank_to_addr, addr);
}

void FleetExecutor::Run(const std::string& carrier_id) {
  Carrier* carrier = GlobalMap<std::string, Carrier>::Get(carrier_id);
  // Set current running carrier
  if (*GlobalVal<std::string>::Get() != carrier_id) {
    GlobalVal<std::string>::Set(new std::string(carrier_id));
    GlobalVal<MessageBus>::Get()->Barrier();
  }
  carrier->Start();
}

}  // namespace distributed
}  // namespace paddle
