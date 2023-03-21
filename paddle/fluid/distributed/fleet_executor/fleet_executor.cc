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

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

namespace paddle {
namespace distributed {

FleetExecutor::FleetExecutor(const std::string& exe_desc_str) {
  bool parse_flag = exe_desc_.ParseFromString(exe_desc_str);
  PADDLE_ENFORCE(parse_flag,
                 platform::errors::PreconditionNotMet(
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
  for (const auto& carrier : carriers_) {
    carrier.get()->Release();
  }
}

namespace {
void GetSubBlockTask(const std::vector<TaskNode*>& tasks,
                     TaskNode* cur_task,
                     std::set<TaskNode*>* sub_block_task) {
  auto& downstream = cur_task->downstream();
  auto& id_to_dep_type = cur_task->id_to_dep_type();
  for (auto& down : downstream) {
    int64_t task_id = down.first;
    if (id_to_dep_type.at(task_id) == DependType::NORMAL) {
      for (const auto& task : tasks) {
        if (task->task_id() == task_id) {
          sub_block_task->emplace(task);
          GetSubBlockTask(tasks, task, sub_block_task);
        }
      }
    }
  }
}

void PreventVarsDelete(
    std::unordered_map<const framework::OperatorBase*,
                       std::vector<std::string>>* unused_vars,
    const std::vector<std::string>& vars_not_gc) {
  std::vector<const framework::OperatorBase*> changed_ops;

  for (const auto& pair : *unused_vars) {
    const framework::OperatorBase* op = pair.first;
    std::vector<std::string> cur_unused = pair.second;
    for (auto name : vars_not_gc) {
      auto iter = std::find(cur_unused.begin(), cur_unused.end(), name);
      if (iter != cur_unused.end()) {
        VLOG(3) << "Removing var: [" << name
                << "] from the unused vars list of op: [" << op->Type() << "]";
        cur_unused.erase(iter);
        if (std::find(changed_ops.begin(), changed_ops.end(), op) ==
            changed_ops.end()) {
          // record the op whose unused vars have been updated
          changed_ops.emplace_back(op);
        }
      }
    }
    // update the unused vars list in the map
    unused_vars->at(op) = cur_unused;
  }
  for (auto op : changed_ops) {
    const auto& iter = unused_vars->find(op);
    if (iter->second.empty()) {
      // remove those ops in the map that have empty unused vars list
      VLOG(3) << "Removing op: [" << op->Type() << "] from unused_vars map.";
      unused_vars->erase(iter);
    }
  }
}

std::vector<std::string> GetUnusedVarsAfterWhile(
    const framework::ProgramDesc& program_desc,
    const std::vector<std::string>& vars_not_gc) {
  // NOTE: Since while op won't appear in task node, in order to analyze
  // the vars which should be free after calling while op, we rebuild the
  // whole program and get the unused vars after calling while op.
  // vars in parent block should not be free until the while op is finished.
  // The local vars will be free while running op in sub block.
  // The unused vars above will be free in cond interceptor.
  std::vector<std::string> while_block_vars;
  std::vector<std::unique_ptr<framework::OperatorBase>> ops;
  for (const auto& desc : program_desc.Block(0).AllOps()) {
    ops.emplace_back(framework::OpRegistry::CreateOp(*desc));
  }
  auto unused_vars = framework::GetUnusedVars(program_desc.Block(0), ops, {});
  PreventVarsDelete(&unused_vars, vars_not_gc);
  for (const auto& pair : unused_vars) {
    if (pair.first->Type() == "while") {
      for (const auto& var_name : pair.second) {
        while_block_vars.emplace_back(var_name);
      }
    }
  }
  return while_block_vars;
}

std::unordered_map<const framework::OperatorBase*, std::vector<std::string>>
GetSubUnusedVars(const framework::ProgramDesc& program_desc,
                 const std::set<TaskNode*>& sub_block_tasks,
                 const std::vector<std::string>& vars_not_gc) {
  std::vector<std::unique_ptr<framework::OperatorBase>> ops;
  for (auto* task_node : sub_block_tasks) {
    for (const auto& op : task_node->ops()) {
      ops.emplace_back(std::unique_ptr<framework::OperatorBase>(op));
    }
  }
  auto unused_vars = framework::GetUnusedVars(program_desc.Block(1), ops, {});
  for (auto& unique_op : ops) {
    unique_op.release();
  }
  PreventVarsDelete(&unused_vars, vars_not_gc);
  return unused_vars;
}

}  // namespace

void FleetExecutor::CopyParametersFromRoot(
    const framework::ProgramDesc& program,
    const std::vector<std::string>& inference_root_scope_vars) {
  std::map<std::string, int> inference_root_scope_var_map;
  for (auto var_name : inference_root_scope_vars) {
    inference_root_scope_var_map.insert({var_name, 1});
  }
  // Create persistable variables in root scope
  for (size_t i = 0; i < program.Size(); ++i) {
    for (auto& var : program.Block(i).AllVars()) {
      std::string var_name = var->Name();
      bool force_root = inference_root_scope_var_map.find(var_name) !=
                        inference_root_scope_var_map.end();
      if (force_root) {
        VLOG(4) << var_name
                << " will be forced to be created in the root scope.";
      }
      if (var->Persistable() || force_root) {
        auto* ptr = root_scope_->Var(var_name);
        InitializeVariable(ptr, var->GetType());
        VLOG(5) << "Create persistable var: " << var_name
                << ", which pointer is " << ptr;
      }
    }
  }
  // Create other normal parameters in each micro-scope
  for (auto& micro_scope : microbatch_scopes_) {
    for (size_t i = 0; i < program.Size(); ++i) {
      for (auto& var : program.Block(i).AllVars()) {
        if (!var->Persistable()) {
          auto* ptr = micro_scope->Var(var->Name());
          VLOG(5) << "Create variable " << var->Name() << " for microbatch ptr "
                  << micro_scope << ", which pointer is " << ptr << ".";
          InitializeVariable(ptr, var->GetType());
        }
      }
    }
  }
}

void FleetExecutor::Init(
    int32_t num_of_carriers,
    const framework::ProgramDesc& program_desc,
    framework::Scope* scope,
    const platform::Place& place,
    int64_t num_micro_batches,
    const std::vector<TaskNode*>& task_nodes,
    const std::unordered_map<int64_t, int64_t>& task_id_to_rank,
    const std::vector<std::string>& inference_root_scope_vars,
    const std::vector<framework::Scope*>& micro_scope_list) {
  PADDLE_ENFORCE_GT(task_nodes.size(),
                    0,
                    platform::errors::InvalidArgument(
                        "Fleet executor is inited with empty task node"));
  root_scope_ = scope;
  // Analyze the variables in program_desc, consider the program has while op
  // Set the unused var after running while op
  std::set<TaskNode*> sub_block_tasks;
  std::vector<std::string> while_block_vars;
  for (const auto& task_node : task_nodes) {
    if (task_node->type() == "Cond") {
      GetSubBlockTask(task_nodes, task_node, &sub_block_tasks);
      while_block_vars =
          GetUnusedVarsAfterWhile(program_desc, inference_root_scope_vars);
      for (auto* task_node : sub_block_tasks) {
        for (auto iter : task_node->vars_to_dtype()) {
          while_block_vars.emplace_back(iter.first);
        }
      }
      VLOG(3) << "Vars will be gced after while op";
      for (auto var : while_block_vars) {
        VLOG(3) << var;
      }
      task_node->SetWhileBlockVars(while_block_vars);
    }
  }
  std::vector<framework::OperatorBase*> sub_block_ops;
  for (const auto& task_node : sub_block_tasks) {
    for (const auto& op : task_node->ops()) {
      sub_block_ops.emplace_back(op);
    }
  }
  // Analyse the unused vars in block 0. The operators in block 1
  // should be passed in first for prevent vars been released but removed soon.
  // Since the unused vars in block 1 need to analyse separately.
  std::vector<std::unique_ptr<framework::OperatorBase>> ops;
  for (const auto& task_node : task_nodes) {
    for (const auto& op : task_node->ops()) {
      ops.emplace_back(std::unique_ptr<framework::OperatorBase>(op));
    }
  }
  auto global_unused_vars =
      framework::GetUnusedVars(program_desc.Block(0), ops, {});

  for (auto& unique_op : ops) {
    unique_op.release();
  }

  auto sub_unused_vars =
      GetSubUnusedVars(program_desc, sub_block_tasks, while_block_vars);

  // NOTE: For inference, the vars in inference_root_scope_vars
  // shouldn't be deleted during inf, for that they may be the result of the
  // inf. If they are GCed, it will cause error during ZeroCopy the result.
  PreventVarsDelete(&global_unused_vars, inference_root_scope_vars);

  // Create num_micro_batches micro_scope if micro_scope_list is none
  // Create persistable parameters in root scope
  // Create other parameters in each micro_scope
  bool need_create_scope = micro_scope_list.empty();
  if (need_create_scope) {
    minibatch_scope_ = &scope->NewScope();
    microbatch_scopes_.resize(num_micro_batches);
    for (int i = 0; i < num_micro_batches; ++i) {
      microbatch_scopes_[i] = &minibatch_scope_->NewScope();
    }
  } else {
    microbatch_scopes_ = micro_scope_list;
  }
  CopyParametersFromRoot(program_desc, inference_root_scope_vars);

  // Here we set the thread num to num_of_carriers + 1, because we use one
  // thread for one carrier for now, and one thread for interceptor which not
  // belong to any carrier.
  thread_pool_ = std::make_unique<TaskLoopThreadPool>();
  thread_pool_->SetThreadNum(num_of_carriers + 1);
  thread_pool_->Start();

  CreateSourceAndSink(microbatch_scopes_.size(), task_nodes, place);

  runtime_graph_ = std::make_shared<RuntimeGraph>();
  std::unordered_map<int64_t, TaskNode*> interceptor_id_to_task;
  for (auto task_node : task_nodes) {
    if (sub_block_tasks.find(task_node) == sub_block_tasks.end()) {
      task_node->SetUnusedVars(global_unused_vars);
    } else {
      task_node->SetUnusedVars(sub_unused_vars);
    }
    int64_t interceptor_id = task_node->task_id();
    interceptor_id_to_task.emplace(interceptor_id, task_node);
  }

  runtime_graph_->SetInterceptorIdToRank(task_id_to_rank);
  runtime_graph_->SetInterceptorIdToNode(interceptor_id_to_task);

  VLOG(5) << runtime_graph_->DebugString();

  std::vector<std::vector<framework::Scope*>> sub_micro_scope_list(
      num_of_carriers);
  for (size_t i = 0; i < microbatch_scopes_.size(); ++i) {
    int carrier_id = (i % num_of_carriers);
    sub_micro_scope_list[carrier_id].emplace_back(microbatch_scopes_[i]);
  }
  for (auto id = 0; id < num_of_carriers; ++id) {
    carriers_.emplace_back(std::make_unique<Carrier>(id));
    InitCarrier(carriers_[id].get(),
                scope,
                minibatch_scope_,
                place,
                program_desc,
                sub_micro_scope_list[id]);
  }

  // Configure the source and sink interceptor
  source_interceptor_->SetPlace(place);
  source_interceptor_->SetMiniBatchScope(minibatch_scope_);
  source_interceptor_->SetMicroBatchScope(microbatch_scopes_);
  source_interceptor_->SetRootScope(root_scope_);
  std::vector<Carrier*> multi_carriers;
  for (const auto& carrier : carriers_) {
    multi_carriers.emplace_back(carrier.get());
  }
  source_interceptor_->RegisterMultiCarrier(multi_carriers);
  auto* loop = thread_pool_->GetLoop(carriers_.size());
  PADDLE_ENFORCE_NOT_NULL(
      loop,
      platform::errors::Fatal(
          "thread task loop for source and sink must not null"));
  source_interceptor_->RegisterTaskLoop(loop);
  sink_interceptor_->RegisterTaskLoop(loop);
  sink_interceptor_->SetConditionVariable(&cond_var_);

  for (auto& carrier : carriers_) {
    carrier->SetSourceInterceptor(source_interceptor_.get());
    carrier->SetSinkInterceptor(sink_interceptor_.get());
  }

  GlobalVal<MessageBus>::Get()->Barrier();
}

void FleetExecutor::InitCarrier(
    Carrier* carrier,
    framework::Scope* scope,
    framework::Scope* minibatch_scope,
    const platform::Place& place,
    const framework::ProgramDesc& program_desc,
    const std::vector<framework::Scope*>& micro_scope_list) {
  carrier->Init(exe_desc_.cur_rank(),
                runtime_graph_->interceptor_id_to_rank(),
                runtime_graph_->interceptor_id_to_node(),
                program_desc,
                scope,
                minibatch_scope,
                place,
                micro_scope_list,
                thread_pool_.get());
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
        rank_to_addr.size(),
        1,
        platform::errors::NotFound("Empty address is not valid for "
                                   "paddle.distributed.launch method."));
    PADDLE_ENFORCE_EQ(
        cur_rank,
        0,
        platform::errors::NotFound("Address is empty but cur rank is not 0."));
  }
  VLOG(3) << "Current rank is " << cur_rank << " and the ip_port is "
          << (addr == "" ? "empty" : addr) << ".";
  VLOG(3) << "The number of ranks are "
          << (rank_to_addr.size() == 0 ? 1 : rank_to_addr.size()) << ".";
  VLOG(5) << ss.str();
  GlobalVal<MessageBus>::Get()->Init(cur_rank, rank_to_addr, addr);
}

void FleetExecutor::WaitCondVarToExit() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.wait(lock);
}

void FleetExecutor::CreateSourceAndSink(
    int64_t max_run_times,
    const std::vector<TaskNode*>& task_nodes,
    const platform::Place& place) {
  auto cur_rank = exe_desc_.cur_rank();
  TaskNode* source = new TaskNode(cur_rank, SOURCE_ID, max_run_times);
  TaskNode* sink = new TaskNode(cur_rank, SINK_ID, max_run_times);

  // find nodes without upstreams or without downstreams
  std::vector<TaskNode*> origin_sources, origin_sinks;
  for (const auto& task_node : task_nodes) {
    if (task_node->upstream().empty()) {
      origin_sources.emplace_back(task_node);
    }
    if (task_node->downstream().empty()) {
      origin_sinks.emplace_back(task_node);
    }
  }
  // link source node with origin source
  for (const auto& node : origin_sources) {
    source->AddDownstreamTask(node->task_id(),
                              std::numeric_limits<int64_t>::max());
    node->AddUpstreamTask(SOURCE_ID, std::numeric_limits<int64_t>::max());
  }
  for (const auto& node : origin_sinks) {
    sink->AddUpstreamTask(node->task_id(), std::numeric_limits<int64_t>::max());
    node->AddDownstreamTask(SINK_ID, std::numeric_limits<int64_t>::max());
  }
  // create source and sink interceptor
  source_interceptor_ = InterceptorFactory::Create("Source", SOURCE_ID, source);
  sink_interceptor_ = InterceptorFactory::Create("Sink", SINK_ID, sink);
}

void FleetExecutor::Run() {
  for (const auto& carrier : carriers_) {
    carrier->Start();
  }
  // Send start message to source interceptor
  InterceptorMessage start_msg;
  start_msg.set_message_type(START);
  source_interceptor_->EnqueueRemoteInterceptorMessage(start_msg);

  WaitCondVarToExit();
  for (const auto& carrier : carriers_) {
    carrier.get()->ClearMicroScopes();
  }
}

}  // namespace distributed
}  // namespace paddle
