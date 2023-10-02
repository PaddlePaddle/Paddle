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

#include "paddle/fluid/distributed/fleet_executor/carrier.h"

#include <algorithm>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/utils/flags.h"
PADDLE_DEFINE_EXPORTED_bool(
    fleet_executor_with_standalone,
    false,
    "Use standalone executor to run ops. Temporary FLAGS, will be removed "
    "after all fleet executor cases are modified to run ops with standalone "
    "executor.");
PHI_DECLARE_bool(cache_inference_while_scope);

namespace paddle {
namespace distributed {

USE_INTERCEPTOR(Source);
USE_INTERCEPTOR(Compute);
USE_INTERCEPTOR(Amplifier);
USE_INTERCEPTOR(Sink);
USE_INTERCEPTOR(Cond);
USE_INTERCEPTOR(Start);

void Carrier::Init(
    int64_t rank,
    const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank) {
  rank_ = rank;
  interceptor_id_to_rank_ = interceptor_id_to_rank;

  // TODO(fleet_exe dev): thread pool
  thread_num_ = 1;
  thread_pool_.SetThreadNum(thread_num_);
  thread_pool_.Start();
}

void Carrier::Init(
    int64_t rank,
    const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
    const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node,
    const framework::ProgramDesc& program,
    framework::Scope* scope,
    int64_t num_micro_batches,
    const platform::Place& place,
    const std::vector<std::string>& inference_root_scope_vars,
    const std::vector<framework::Scope*>& micro_scope_list) {
  rank_ = rank;
  interceptor_id_to_rank_ = interceptor_id_to_rank;
  interceptor_id_to_node_ = interceptor_id_to_node;
  place_ = place;
  root_scope_ = scope;
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  bool need_create_scope = micro_scope_list.empty();

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      platform::errors::InvalidArgument("root_scope can not be nullptr"));

  if (need_create_scope) {
    minibatch_scope_ = &root_scope_->NewScope();
    microbatch_scopes_.resize(num_micro_batches);
    for (int i = 0; i < num_micro_batches; ++i) {
      microbatch_scopes_[i] = &minibatch_scope_->NewScope();
      CopyParameters(i, program, inference_root_scope_vars);
    }
  } else {
    microbatch_scopes_ = micro_scope_list;
    for (int i = 0; i < num_micro_batches; ++i) {
      CopyParameters(i, program, inference_root_scope_vars);
    }
  }

  // Add source and sink interceptor id to rank
  interceptor_id_to_rank_.emplace(SOURCE_ID, rank);
  interceptor_id_to_rank_.emplace(SINK_ID, rank);

  // TODO(fleet_exe dev): thread pool
  thread_num_ = 1;
  thread_pool_.SetThreadNum(thread_num_);
  thread_pool_.Start();

  CreateInterceptors(inference_root_scope_vars);
  is_init_ = true;
}

void Carrier::Release() {
  if (root_scope_) {
    root_scope_->DropKids();
  }
}

Carrier::~Carrier() { VLOG(3) << "Carrier's destructor."; }

void Carrier::CopyParameters(
    int microbatch_id,
    const framework::ProgramDesc& program,
    const std::vector<std::string>& inference_root_scope_vars) {
  std::map<std::string, int> inference_root_scope_var_map;
  for (auto var_name : inference_root_scope_vars) {
    inference_root_scope_var_map.insert({var_name, 1});
  }
  for (size_t i = 0; i < program.Size(); ++i) {
    for (auto& var : program.Block(i).AllVars()) {
      std::string var_name = var->Name();
      bool force_root = inference_root_scope_var_map.find(var_name) !=
                        inference_root_scope_var_map.end();
      if (force_root) {
        VLOG(4) << var_name
                << " will be forced to be created in the root scope.";
      }
      if ((var->Persistable() || force_root) && microbatch_id == 0) {
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
}

bool Carrier::EnqueueInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  PADDLE_ENFORCE_EQ(
      interceptor_message.ctrl_message(),
      false,
      platform::errors::Fatal(
          "Control message should be only send inter rank using message bus."));
  int64_t dst_id = interceptor_message.dst_id();
  Interceptor* dst_interceptor = GetInterceptor(dst_id);
  dst_interceptor->EnqueueRemoteInterceptorMessage(interceptor_message);
  return true;
}

Interceptor* Carrier::GetInterceptor(int64_t interceptor_id) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_NE(iter,
                    interceptor_idx_to_interceptor_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find interceptor instance for interceptor "
                        "id %lld. Wrong dst? Call before init?",
                        interceptor_id));
  return iter->second.get();
}

void Carrier::Wait() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  cond_var_.wait(lock);
}

void Carrier::WakeUp() {
  // probably double notify, but ok for ut
  cond_var_.notify_all();
}

void Carrier::Start() {
  PADDLE_ENFORCE_EQ(is_init_,
                    true,
                    platform::errors::PreconditionNotMet(
                        "Using carrier before initialized."));
  InterceptorMessage start_msg;
  start_msg.set_dst_id(SOURCE_ID);
  start_msg.set_src_id(SOURCE_ID);
  start_msg.set_message_type(START);
  Send(start_msg);
  // TODO(wangxi): async step
  Wait();
  dev_ctx_->Wait();
  if (!FLAGS_cache_inference_while_scope) {
    // don't drop_kids when cache_inference_while_scope
    for (auto* micro_scope : microbatch_scopes_) {
      // By default, we should delete all kid scopes after run executor because
      // some operators may create local scope when running, such as while_op.
      // But when while_op also create a local executor to run it's sub block,
      // the sub scopes it created should not be dropped immediately, because
      // while_grad_op will use some variables created during while_op run, so
      // we need to keep the kids and wait for the outer executor to drop them.
      micro_scope->DropKids();
    }
  }
}

bool Carrier::IsInit() const { return is_init_; }

int64_t Carrier::GetRank(int64_t interceptor_id) const {
  PADDLE_ENFORCE_NE(
      interceptor_id_to_rank_.find(interceptor_id),
      interceptor_id_to_rank_.end(),
      platform::errors::NotFound("Cannot find rank for interceptor id %lld.",
                                 interceptor_id));
  return interceptor_id_to_rank_.at(interceptor_id);
}

bool Carrier::Send(const InterceptorMessage& msg) {
  int64_t src_id = msg.src_id();
  // TODO(liyurui): compatible solution, will be removed completely in the
  // future
  if (interceptor_id_to_rank_.find(src_id) == interceptor_id_to_rank_.end() &&
      src_id == SOURCE_ID) {
    src_id = msg.dst_id();
  }
  int64_t dst_id = msg.dst_id();
  int64_t src_rank = GetRank(src_id);
  int64_t dst_rank = GetRank(dst_id);
  PADDLE_ENFORCE_EQ(
      src_rank,
      rank_,
      platform::errors::Fatal("The source rank id %lld, which is not equal to "
                              "the carrier rank id %lld.",
                              src_rank,
                              rank_));
  if (src_rank == dst_rank) {
    VLOG(3) << "Send a message from interceptor " << src_id
            << " to interceptor " << dst_id << ", which are in the same ranks.";
    return EnqueueInterceptorMessage(msg);
  }
  VLOG(3) << "Send a message from interceptor " << src_id << " to interceptor "
          << dst_id << ", which are in different ranks.";
  return GlobalVal<MessageBus>::Get()->Send(dst_rank, msg);
}

Interceptor* Carrier::SetInterceptor(int64_t interceptor_id,
                                     std::unique_ptr<Interceptor> interceptor) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_EQ(iter,
                    interceptor_idx_to_interceptor_.end(),
                    platform::errors::AlreadyExists(
                        "The interceptor id %lld has already been created! "
                        "The interceptor id should be unique.",
                        interceptor_id));
  interceptor->RegisterCarrier(this);

  // TODO(fleet_exe dev): get loop
  auto* loop =
      thread_pool_.GetLoop(static_cast<int>(interceptor_id % thread_num_));
  PADDLE_ENFORCE_NOT_NULL(
      loop, platform::errors::Fatal("thread task loop must not null"));
  interceptor->RegisterTaskLoop(loop);

  auto* ptr = interceptor.get();
  interceptor_idx_to_interceptor_.insert(
      std::make_pair(interceptor_id, std::move(interceptor)));
  return ptr;
}

static std::shared_ptr<framework::GarbageCollector> GetGC(
    const platform::Place& place) {
  int64_t max_memory_size = framework::GetEagerDeletionThreshold();
  std::shared_ptr<framework::GarbageCollector> gc;
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(place)) {
      if (framework::IsFastEagerDeletionModeEnabled()) {
        gc.reset(new framework::UnsafeFastGPUGarbageCollector(place,
                                                              max_memory_size));
      }
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (platform::is_custom_place(place)) {
      if (framework::IsFastEagerDeletionModeEnabled()) {
        gc.reset(new framework::CustomDeviceUnsafeFastGarbageCollector(
            place, max_memory_size));
      }
    }
#endif
  }  // max_memory_size >= 0

  return gc;
}

void Carrier::CreateInterceptors(
    const std::vector<std::string>& inference_root_scope_vars) {
  if (interceptor_id_to_node_.empty()) return;

  auto gc = GetGC(place_);

  // create source and sink task node
  int64_t max_run_times = static_cast<int64_t>(microbatch_scopes_.size());
  TaskNode* source = new TaskNode(
      rank_, SOURCE_ID, max_run_times);  // rank, task_id, max_run_times
  TaskNode* sink = new TaskNode(rank_, SINK_ID, max_run_times);
  // find nodes without upstreams or without downstreams
  std::vector<TaskNode*> origin_sources, origin_sinks;
  for (const auto& item : interceptor_id_to_node_) {
    TaskNode* task_node = item.second;
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
  // link sink node with origin sink
  for (const auto& node : origin_sinks) {
    sink->AddUpstreamTask(node->task_id(), std::numeric_limits<int64_t>::max());
    node->AddDownstreamTask(SINK_ID, std::numeric_limits<int64_t>::max());
  }
  // create source and sink interceptor
  SetInterceptor(SOURCE_ID,
                 InterceptorFactory::Create("Source", SOURCE_ID, source));
  SetInterceptor(SINK_ID, InterceptorFactory::Create("Sink", SINK_ID, sink));

  // create each Interceptor
  // no auto init since there is no config
  for (const auto& item : interceptor_id_to_node_) {
    int64_t interceptor_id = item.first;
    TaskNode* task_node = item.second;

    PADDLE_ENFORCE_LT(
        task_node->run_at_offset(),
        task_node->run_per_steps(),
        platform::errors::InvalidArgument(
            "Interceptor's run_at_offset must < run_per_steps, must now "
            "run_at_offset=%ld run_per_steps=%ld",
            task_node->run_at_offset(),
            task_node->run_per_steps()));

    std::unique_ptr<Interceptor> interceptor;
    PADDLE_ENFORCE_NE(task_node->type().empty(),
                      true,
                      platform::errors::NotFound(
                          "Cannot found type for task node with id %lld",
                          task_node->task_id()));
    interceptor = InterceptorFactory::Create(
        task_node->type(), interceptor_id, task_node);
    interceptor->SetPlace(place_);
    interceptor->SetMiniBatchScope(minibatch_scope_);
    interceptor->SetMicroBatchScope(microbatch_scopes_);
    interceptor->SetRootScope(root_scope_);

    if (FLAGS_fleet_executor_with_standalone &&
        (task_node->type() == "Amplifier" || task_node->type() == "Compute")) {
      std::vector<std::shared_ptr<InterpreterCore>> cores;
      framework::interpreter::ExecutionConfig execution_config;
      execution_config.create_local_scope = false;
      execution_config.force_root_scope_vars = std::set<std::string>(
          inference_root_scope_vars.begin(), inference_root_scope_vars.end());

      const framework::ProgramDesc* program = task_node->program();
      PADDLE_ENFORCE_NOT_NULL(
          program,
          phi::errors::InvalidArgument("TaskNode %d's program is not set.",
                                       interceptor_id));
      std::vector<framework::VarDesc*> all_vars = program->Block(0).AllVars();
      for (framework::VarDesc* var : all_vars) {
        execution_config.skip_gc_vars.insert(var->Name());
      }

      // ONLY unused vars can be GCed.
      const std::unordered_map<const framework::OperatorBase*,
                               std::vector<std::string>>& unused_vars =
          task_node->unused_vars();
      for (auto& item : unused_vars) {
        for (const std::string& unused_var : item.second) {
          execution_config.skip_gc_vars.erase(unused_var);
        }
      }

      for (framework::Scope* scope : microbatch_scopes_) {
        cores.push_back(std::make_shared<InterpreterCore>(
            place_, task_node->program()->Block(0), scope, execution_config));
      }

      for (size_t i = 1; i < cores.size(); ++i) {
        cores[i]->ShareWorkQueueFrom(cores[i - 1]);
      }

      interceptor->SetInterpreterCore(cores);
    } else {
      interceptor->SetGC(gc);
    }

    SetInterceptor(interceptor_id, std::move(interceptor));
    VLOG(3) << "Create Interceptor with interceptor id: " << interceptor_id
            << " with type: " << task_node->type() << ".";

    PADDLE_ENFORCE_EQ(
        task_node->upstream().empty(),
        false,
        platform::errors::PreconditionNotMet(
            "There should not have normal nodes as source nodes"));
    PADDLE_ENFORCE_EQ(task_node->downstream().empty(),
                      false,
                      platform::errors::PreconditionNotMet(
                          "There should not have normal nodes as sink nodes"));
  }
}

}  // namespace distributed
}  // namespace paddle
