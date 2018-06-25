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

namespace paddle {
namespace operators {
namespace distributed {

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
    return (barrier_counter_[rpc_name] >= client_num_ || exit_flag_.load());
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

void RPCServer::DecreaseClientNum() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    client_num_--;
  }
  barrier_cond_.notify_all();
}

void RPCServer::ResetBarrierCounter() {
  VLOG(3) << "RPCServer ResetBarrierCounter ";
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& t : barrier_counter_) {
    t.second = 0;
  }
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
