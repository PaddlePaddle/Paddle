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

#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace distributed {

class Barrier {
 public:
  explicit Barrier(int worker_size) : counter_(0), worker_size_(worker_size) {}

  void Wait();
  void Increase();
  void Decrease();
  void Reset();
  void Notify();
  void SetWorkerSize(int worker_size);

 private:
  std::mutex mutex_;
  int counter_;
  int worker_size_;
  std::condition_variable barrier_cond_;

  DISABLE_COPY_AND_ASSIGN(Barrier);
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
