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

#pragma once

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
  working_ = true;

  if (barrier == BATCH_BARRIER_MESSAGE) {
    send_barrier_queue->Push(worker_id);
  } else if (barrier == FETCH_BARRIER_MESSAGE) {
    recv_barrier_queue->Push(worker_id);
  } else {
    PADDLE_THROW("unknown status");
  }
  Wait();
}

BarrierMonitor::Monitor() {
  while (!working_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    VLOG(4) << "Barrier not working currently";
  }

  while (running_) {
    int timer = 0;
    while (timer < kMaxWaitMS) {
      if (IsReady()) {
        Swap();
        break;
      } else {
        timer++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    if (timer >= kMaxWaitMS) {
      Invalid();
    }
  }
}

bool BarrierMonitor::IsReady() {
  if (barrier_type == kSendBarrier) {
    return send_barrier_queue.size() == workers_;
  } else {
    return recv_barrier_queue.size() == workers_;
  }
}

BarrireMonitor::Invalid() {
  std::unique_lock<std::mutex> lck(mutex_);
  valid = false;
  send_barrier_queue.Clear();
  recv_barrier_queue.Clear();
  cv_.notify_all();
}

BarrierMonitor::Swap() {
  std::unique_lock<std::mutex> lck(mutex_);

  if (barrier_type == kSendBarrier) {
    barrier_type == kRecvBarrier;
    valid = true;
    send_barrier_queue.Clear();
  } else {
    barrier_type == kSendBarrier;
    valid = true;
    recv_barrier_queue.Clear();
  }
  cv_.notify_all();
}

std::once_flag BarrierMonitor::init_flag_;
std::unique_ptr<BarrierMonitor> BarrierMonitor::monitor_(nullptr);

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
