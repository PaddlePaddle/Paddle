// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_int32(rpc_server_profile_period, 0,
             "the period of listen_and_serv to do profile");
DEFINE_string(rpc_server_profile_path, "/dev/null",
              "the profile log file path");

namespace paddle {
namespace operators {
namespace distributed {

RPCServerProfiler::RPCServerProfiler(int profile_period,
                                     const std::string& profile_log_path)
    : profile_period_(profile_period), profile_log_path_(profile_log_path) {
  step_ = 0;
}

void RPCServerProfiler::OneStep() {
  PADDLE_ENFORCE_LE(step_, profile_period_,
                    "step_ should not be larger then "
                    "profile_period_");
  if (profile_period_ <= 0) {
    return;
  }

  if (step_ == 0) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
  }
  if (step_ == profile_period_) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      profile_log_path_);
    step_ = 0;
  } else {
    step_++;
  }
}

void RPCServer::ShutDown() {
  LOG(INFO) << "RPCServer ShutDown ";
  ShutDownImpl();

  exit_flag_ = true;
  barrier_cond_.notify_all();
  rpc_cond_.notify_all();
}

void RPCServer::SavePort() const {
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  std::ofstream port_file;
  port_file.open(file_path);
  port_file << selected_port_;
  port_file.close();
  VLOG(4) << "selected port written to " << file_path;
}

void RPCServer::WaitBarrier(const std::string& rpc_name) {
  std::unique_lock<std::mutex> lock(this->mutex_);
  barrier_cond_.wait(lock, [this, &rpc_name] {
    return ((barrier_counter_[rpc_name] == client_num_ && client_num_ != 0) ||
            exit_flag_.load());
  });

  VLOG(3) << "batch_barrier_: " << rpc_name << " "
          << barrier_counter_[rpc_name];
}

void RPCServer::IncreaseBatchBarrier(const std::string rpc_name) {
  VLOG(4) << "RPCServer begin IncreaseBatchBarrier " << rpc_name;
  int b = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  b = ++barrier_counter_[rpc_name];
  if (b >= client_num_) {
    lock.unlock();
    barrier_cond_.notify_all();
    lock.lock();
  }
}

void RPCServer::Complete() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    client_num_--;
    need_reset_all_vars_ = true;

    VLOG(4) << "decrease client_num to: " << client_num_;
    if (cur_cond_.load() == rpc_cond_map_[kRequestGet]) {
      barrier_counter_[kRequestGet]--;
    }
  }
  barrier_cond_.notify_all();
}

bool RPCServer::NeedResetAllVars() {
  std::unique_lock<std::mutex> lock(mutex_);
  return need_reset_all_vars_;
}

int RPCServer::GetClientNum() {
  std::unique_lock<std::mutex> lock(mutex_);
  return client_num_;
}

void RPCServer::ResetBarrierCounter() {
  VLOG(3) << "RPCServer ResetBarrierCounter ";
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& t : barrier_counter_) {
    t.second = 0;
  }
  need_reset_all_vars_ = false;
}

void RPCServer::RegisterRPC(const std::string& rpc_name,
                            RequestHandler* handler, int thread_num) {
  rpc_call_map_[rpc_name] = handler;
  rpc_thread_num_[rpc_name] = thread_num;

  static int cond = -1;
  rpc_cond_map_[rpc_name] = ++cond;
  VLOG(4) << "RegisterRPC rpc_name:" << rpc_name << ", handler:" << handler
          << ", cond:" << rpc_cond_map_[rpc_name];
}

void RPCServer::SetCond(const std::string& rpc_name) {
  VLOG(3) << "RPCServer SetCond " << rpc_name;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cur_cond_ = rpc_cond_map_[rpc_name];
  }

  rpc_cond_.notify_all();
}

void RPCServer::WaitCond(const std::string& rpc_name) {
  VLOG(4) << "RPCServer WaitCond " << rpc_name;
  int cond = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond = rpc_cond_map_[rpc_name];
  }

  std::unique_lock<std::mutex> lock(mutex_);
  rpc_cond_.wait(
      lock, [=] { return (cur_cond_.load() == cond || exit_flag_.load()); });
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
