/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace platform {

class WaitGroup {
 public:
  inline void Add(int n) {
    std::unique_lock<std::mutex> lk(mu_);
    PADDLE_ENFORCE(n >= 0, "add wait must >=0.");
    counter_ += n;
  }

  inline void Done(int n) {
    std::unique_lock<std::mutex> lk(mu_);
    PADDLE_ENFORCE(n <= counter_, " wait group done unmatch to add.");
    counter_ -= n;
    if (counter_ == 0) {
      cv_.notify_all();
    }
  }

  inline void Add() { Add(1); }

  inline void Done() { Done(1); }

  inline void Wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return counter_ == 0; });
  }

  inline int GetCount() {
    std::unique_lock<std::mutex> lk(mu_);
    return counter_;
  }

 private:
  int counter_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
};

struct Communicator {
  std::vector<ncclComm_t> comms_;
  std::unordered_map<int, int> comm_id_map_;

  int GetCommId(int device_id) const { return comm_id_map_.at(device_id); }

  void InitAll(const std::vector<int>& gpus) {
    comms_.resize(gpus.size());
    for (size_t i = 0; i < gpus.size(); ++i) {
      comm_id_map_[gpus[i]] = i;
    }
    PADDLE_ENFORCE(ncclCommInitAll(comms_.data(), gpus.size(), gpus.data()));
  }

  ~Communicator() {
    for (size_t i = 0; i < comms_.size(); ++i) {
      PADDLE_ENFORCE(ncclCommDestroy(comms_[i]));
    }
  }

  // DISABLE_COPY_AND_ASSIGN(Communicator);
};

Communicator* NewCommunicator(const std::vector<int>& gpus);

}  // namespace platform
}  // namespace paddle
