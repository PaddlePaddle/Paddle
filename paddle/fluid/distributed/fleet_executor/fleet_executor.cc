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
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"

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

FleetExecutor::~FleetExecutor() {
  root_scope_->DropKids();
  for (const auto& carrier_id : carrier_ids_) {
    GlobalMap<std::string, Carrier>::Get(carrier_id)->Release();
  }
}

void FleetExecutor::Init(
    const std::string& carrier_id, const framework::ProgramDesc& program_desc,
    framework::Scope* scope, const platform::Place& place,
    const std::vector<TaskNode*>& task_nodes,
    const std::unordered_map<int64_t, int64_t>& task_id_to_rank) {
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
  root_scope_ = scope;
  place_ = place;
  PADDLE_ENFORCE_NOT_NULL(root_scope_, platform::errors::InvalidArgument(
                                           "root_scope_ can not be nullptr"));
  minibatch_scope_ = &root_scope_->NewScope();
  int64_t num_micro_batches = exe_desc_.num_micro_batches();
  microbatch_scopes_.resize(num_micro_batches);
  for (int i = 0; i < num_micro_batches; ++i) {
    microbatch_scopes_[i] = &minibatch_scope_->NewScope();
    CopyParameters(i, program_desc);
  }
  VLOG(5) << runtime_graph_->DebugString();
  Carrier* carrier =
      GlobalMap<std::string, Carrier>::Create(carrier_id, carrier_id);
  carrier_ids_.insert(carrier_id);
  // Set current running carrier
  GlobalVal<std::string>::Set(new std::string(carrier_id));
  InitCarrier(carrier);
  GlobalVal<MessageBus>::Get()->Barrier();
}

void FleetExecutor::InitCarrier(Carrier* carrier) {
  carrier->Init(exe_desc_.cur_rank(), runtime_graph_->interceptor_id_to_rank(),
                runtime_graph_->interceptor_id_to_node(), root_scope_,
                minibatch_scope_, microbatch_scopes_, place_);
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
    // TODO(liyurui): Move barrier to service
    GlobalVal<MessageBus>::Get()->Barrier();
  }
  carrier->Start();
  for (auto* micro_scop : microbatch_scopes_) {
    // By default, we should delete all kid scopes after run executor because
    // some operators may create local scope when running, such as while_op.
    // But when while_op also create a local executor to run it's sub block,
    // the sub scopes it created should not be dropped immediately, because
    // while_grad_op will use some variables created during while_op run, so
    // we need to keep the kids and wait for the outer executor to drop them.
    micro_scop->DropKids();
  }
}

void FleetExecutor::CopyParameters(int microbatch_id,
                                   const framework::ProgramDesc& program) {
  auto& global_block = program.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Persistable() && microbatch_id == 0) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(5) << "Create persistable var: " << var->Name()
              << ", which pointer is " << ptr;
    } else if (!var->Persistable()) {
      auto* ptr = microbatch_scopes_[microbatch_id]->Var(var->Name());
      VLOG(5) << "Create variable " << var->Name() << " for microbatch "
              << microbatch_id << ", which pointer is " << ptr << ".";
      InitializeVariable(ptr, var->GetType());
    }
  }
}

}  // namespace distributed
}  // namespace paddle
