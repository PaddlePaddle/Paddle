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
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace distributed {

USE_INTERCEPTOR(Compute);
USE_INTERCEPTOR(Amplifier);

void Carrier::Init(std::shared_ptr<RuntimeGraph> runtime_graph,
                   framework::Scope* root_scope,
                   framework::Scope* minibatch_scope,
                   const std::vector<framework::Scope*>& microbatch_scopes,
                   const platform::Place& place) {
  PADDLE_ENFORCE_EQ(is_init_, false, platform::errors::AlreadyExists(
                                         "Carrier is already init."));
  runtime_graph_ = runtime_graph;
  minibatch_scope_ = minibatch_scope;
  microbatch_scopes_ = microbatch_scopes;
  place_ = place;
  root_scope_ = root_scope;
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  CreateInterceptors();
  is_init_ = true;
}

void Carrier::Release() {
  // NOTE(wangxi): must join before `Derived Interceptor` destruct,
  // otherwise Derived object will be destructed before thread complete.

  // Sending STOP msg to the source interceptor
  MessageBus& msg_bus = MessageBus::Instance();
  PADDLE_ENFORCE_EQ(msg_bus.IsInit(), true,
                    platform::errors::PreconditionNotMet(
                        "Message bus has not been initialized."));
  for (int64_t id : source_interceptor_ids_) {
    VLOG(3) << "Carrier Release is sending stop to source interceptor " << id
            << ".";
    InterceptorMessage stop_msg;
    // source node STOP is send by carrier, so set src_id=-1
    stop_msg.set_src_id(-1);
    stop_msg.set_dst_id(id);
    stop_msg.set_message_type(STOP);
    msg_bus.Send(stop_msg);
  }

  // TODO(wangxi): Maybe need a better to use thread.
  for (auto& interceptor : interceptor_idx_to_interceptor_) {
    interceptor.second->Join();
  }
}

Carrier::~Carrier() { VLOG(3) << "Carrier's destructor."; }

bool Carrier::EnqueueInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // enqueue message to interceptor
  if (interceptor_message.ctrl_message()) {
    // handle control message
    return true;
  } else {
    {
      std::unique_lock<std::mutex> lock_creating(creating_flag_mutex_);
      if (creating_interceptors_) {
        std::unique_lock<std::mutex> lock_message(tmp_message_mutex_);
        // Cannot handle the message to interceptor since interceptors
        // are still under creating. Will enqueue into a tmp stack.
        VLOG(3) << "Receiving message while creating interceptors.";
        message_tmp_.emplace_back(interceptor_message);
        return true;
      }
    }
    int64_t dst_id = interceptor_message.dst_id();
    Interceptor* dst_interceptor = GetInterceptor(dst_id);
    bool rst =
        dst_interceptor->EnqueueRemoteInterceptorMessage(interceptor_message);
    if (rst) {
      std::condition_variable& interceptor_cond_var =
          dst_interceptor->GetCondVar();
      interceptor_cond_var.notify_all();
    }
    return rst;
  }
}

Interceptor* Carrier::GetInterceptor(int64_t interceptor_id) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_NE(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find interceptor instance for interceptor "
                        "id %lld. Wrong dst? Call before init?",
                        interceptor_id));
  return iter->second.get();
}

void Carrier::Start() {
  MessageBus& msg_bus = MessageBus::Instance();
  PADDLE_ENFORCE_EQ(msg_bus.IsInit(), true,
                    platform::errors::PreconditionNotMet(
                        "Message bus has not been initialized."));

  for (int64_t id : source_interceptor_ids_) {
    VLOG(3) << "Carrier Start is sending start to source interceptor " << id
            << ".";
    InterceptorMessage start_msg;
    // source node data_is_ready is send by carrier, so set src_id=-1
    start_msg.set_src_id(-1);
    start_msg.set_dst_id(id);
    start_msg.set_message_type(DATA_IS_READY);
    msg_bus.Send(start_msg);
  }

  std::unique_lock<std::mutex> lock(running_mutex_);
  cond_var_.wait(lock);
  dev_ctx_->Wait();
}

std::condition_variable& Carrier::GetCondVar() { return cond_var_; }

bool Carrier::IsInit() const { return is_init_; }

Interceptor* Carrier::SetInterceptor(int64_t interceptor_id,
                                     std::unique_ptr<Interceptor> interceptor) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_EQ(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::AlreadyExists(
                        "The interceptor id %lld has already been created! "
                        "The interceptor id should be unique.",
                        interceptor_id));
  auto* ptr = interceptor.get();
  interceptor_idx_to_interceptor_.insert(
      std::make_pair(interceptor_id, std::move(interceptor)));
  return ptr;
}

void Carrier::SetCreatingFlag(bool flag) {
  // set the creating flag
  creating_flag_mutex_.lock();
  VLOG(3) << "Carrier is set the creating flag from " << creating_interceptors_
          << " to " << flag << ".";
  creating_interceptors_ = flag;
  creating_flag_mutex_.unlock();
  if (!flag) {
    for (auto& pair : interceptor_idx_to_interceptor_) {
      // update the source interceptor id
      if (std::find(source_interceptor_ids_.begin(),
                    source_interceptor_ids_.end(),
                    pair.first) == source_interceptor_ids_.end()) {
        auto task = pair.second->GetTaskNode();
        if (task != nullptr && task->upstream().empty()) {
          source_interceptor_ids_.emplace_back(pair.first);
        }
      }
    }
    // finish create interceptors outside, handle tmp messsages
    HandleTmpMessages();
  }
}

void Carrier::HandleTmpMessages() {
  // NOTE: It's ok lock on the tmp_message_mutex_ here, when enter this
  // `HandleTmpMessages` method, the creating_interceptors_ flag
  // must be false, therefore, there won't have conflict with the
  // lock on the tmp_message_mutex_ inside `EnqueueInterceptorMessage`
  // on the same thread.
  std::unique_lock<std::mutex> lock(tmp_message_mutex_);
  VLOG(3) << "Carrier has received " << message_tmp_.size()
          << " messages during creating interceptors.";
  for (const auto& msg : message_tmp_) {
    EnqueueInterceptorMessage(msg);
  }
  message_tmp_.clear();
}

static std::shared_ptr<framework::GarbageCollector> GetGC(
    const platform::Place& place) {
  int64_t max_memory_size = framework::GetEagerDeletionThreshold();
  std::shared_ptr<framework::GarbageCollector> gc;
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(place)) {
      if (framework::IsFastEagerDeletionModeEnabled()) {
        gc.reset(new framework::UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place), max_memory_size));
      }
    }
#endif
  }  // max_memory_size >= 0

  return gc;
}

void Carrier::CreateInterceptors() {
  if (runtime_graph_->intercepter_id_to_node().empty()) return;

  auto gc = GetGC(place_);

  // create each Interceptor
  // no auto init since there is no config
  for (const auto& item : runtime_graph_->intercepter_id_to_node()) {
    int64_t interceptor_id = item.first;
    TaskNode* task_node = item.second;

    PADDLE_ENFORCE_LT(
        task_node->run_at_offset(), task_node->run_per_steps(),
        platform::errors::InvalidArgument(
            "Interceptor's run_at_offset must < run_per_steps, must now "
            "run_at_offset=%ld run_per_steps=%ld",
            task_node->run_at_offset(), task_node->run_per_steps()));

    std::unique_ptr<Interceptor> interceptor;
    if (task_node->type().empty()) {
      // TODO(wangxi): delete this in future
      interceptor.reset(new Interceptor(interceptor_id, task_node));
    } else {
      interceptor = InterceptorFactory::Create(task_node->type(),
                                               interceptor_id, task_node);
    }
    interceptor->SetPlace(place_);
    interceptor->SetMiniBatchScope(minibatch_scope_);
    interceptor->SetMicroBatchScope(microbatch_scopes_);
    interceptor->SetRootScope(root_scope_);
    interceptor->SetGC(gc);

    SetInterceptor(interceptor_id, std::move(interceptor));
    VLOG(3) << "Create Interceptor with interceptor id: " << interceptor_id
            << " with type: " << task_node->type() << ".";

    if (task_node->upstream().empty()) {
      source_interceptor_ids_.emplace_back(interceptor_id);
    }
  }
  // The carrier will be always waiting for outside initializer
  // since there is no interceptor has been created during auto init
  creating_flag_mutex_.lock();
  creating_interceptors_ = false;
  creating_flag_mutex_.unlock();
  HandleTmpMessages();
}

}  // namespace distributed
}  // namespace paddle
