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

#include "paddle/fluid/operators/distributed/barrier_monitor.h"
#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {
bool BarrierMonitor::IncreaseBarrier(const int worker_id,
                                     const std::string &barrier) {
  std::shared_ptr<BarrierBlock> block;

  if (barrier == BATCH_BARRIER_MESSAGE) {
    VLOG(4) << "BarrierMonitor send queue recv trainer: " << worker_id;
    block =
        std::make_shared<BarrierBlock>(worker_id, BarrierType::kSendBarrier);
    send_barrier_queue->Push(block);
  } else if (barrier == FETCH_BARRIER_MESSAGE) {
    VLOG(4) << "BarrierMonitor recv queue recv trainer: " << worker_id;
    block =
        std::make_shared<BarrierBlock>(worker_id, BarrierType::kRecvBarrier);
    recv_barrier_queue->Push(block);
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "unknown Message status %s, only "
        "BATCH_BARRIER_MESSAGE/FETCH_BARRIER_MESSAGE",
        barrier));
  }
  return block->Wait();
}

void BarrierMonitor::DecreaseWorker() {
  workers_--;
  VLOG(1) << "decrement worker num to " << workers_;
}

void BarrierMonitor::Reset(int workers, BarrierType type) {
  std::unique_lock<std::mutex> lk(server_mutex_);

  workers_.exchange(workers);
  barrier_type = type;

  send_barrier_queue->Clear();
  recv_barrier_queue->Clear();

  send_barrier_queue->ReCapacity(workers);
  recv_barrier_queue->ReCapacity(workers);

  VLOG(2) << "reset monitor workers: " << workers_ << " type: " << barrier_type;

  if (monitor_thread_ == nullptr) {
    running_ = true;
    monitor_thread_.reset(
        new std::thread(std::bind(&BarrierMonitor::Monitor, this)));
  }
}

void BarrierMonitor::Monitor() {
  while (!IsReady() && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    VLOG(3) << "sync at first time, wait all trainer ready";
  }

  while (running_) {
    int timer = 0;

    if (IsReady()) {
      Exchange(true);
    } else {
      VLOG(4) << "running timer: " << timer << " barrier: " << barrier_type
              << " sendQ:" << send_barrier_queue->Size()
              << " recvQ: " << recv_barrier_queue->Size();

      timer++;
      if (max_wait_ms == -1 || timer < max_wait_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        VLOG(1) << "time out of " << max_wait_ms
                << ", need barreir: " << barrier_type << " retry";
        Exchange(false);
      }
    }
  }
}

bool BarrierMonitor::IsReady() {
  if (barrier_type == BarrierType::kSendBarrier) {
    return static_cast<int>(send_barrier_queue->Size()) == workers_;
  } else {
    return static_cast<int>(recv_barrier_queue->Size()) == workers_;
  }
}

void BarrierMonitor::Exchange(bool available) {
  if (barrier_type == BarrierType::kSendBarrier) {
    barrier_type = BarrierType::kRecvBarrier;
    NotifyWorker(BarrierType::kSendBarrier, available);
    ServerWeakup();
    VLOG(4) << "barrier monitor server weak up sync to do";
    WaitServerWeakup();
    VLOG(4) << "barrier monitor server weak up sync done";
  } else {
    barrier_type = BarrierType::kSendBarrier;
    NotifyWorker(BarrierType::kRecvBarrier, available);
    VLOG(4) << "barrier monitor server switch to send barrier";
  }
}

void BarrierMonitor::NotifyWorker(BarrierType type, bool available) {
  if (type == BarrierType::kSendBarrier) {
    while (send_barrier_queue->Size() != 0) {
      auto block = send_barrier_queue->Pop();
      block->Done(available);
    }
  } else {
    while (recv_barrier_queue->Size() != 0) {
      auto block = recv_barrier_queue->Pop();
      block->Done(available);
    }
  }
}

void BarrierMonitor::Stop() {
  std::unique_lock<std::mutex> lk(server_mutex_);
  running_ = false;

  barrier_type = BarrierType::kRecvBarrier;
  NotifyWorker(BarrierType::kSendBarrier, true);
  NotifyWorker(BarrierType::kRecvBarrier, true);

  server_cv_.notify_all();

  if (monitor_thread_) monitor_thread_->join();
  monitor_thread_ = nullptr;
}

void BarrierMonitor::WaitServerWeakup() {
  std::unique_lock<std::mutex> lk(server_mutex_);
  server_cv_.wait(lk);
}

void BarrierMonitor::ServerWeakup() {
  std::unique_lock<std::mutex> lk(server_mutex_);
  server_cv_.notify_all();
}

std::once_flag BarrierMonitor::init_flag_;
std::unique_ptr<BarrierMonitor> BarrierMonitor::monitor_(nullptr);

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
