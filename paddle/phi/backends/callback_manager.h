// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT

class ThreadPool;

namespace phi {

namespace stream {
class Stream;
}  // namespace stream

// NOTE(zjl): clean CallbackManager to make compilation faster
// Make CallbackManager thread-safe
class CallbackManager {
 public:
  explicit CallbackManager(stream::Stream* stream);

  ~CallbackManager() = default;

  void AddCallback(std::function<void()> callback) const;

  void Wait() const;

 private:
  stream::Stream* stream_;
  mutable std::shared_ptr<::ThreadPool> thread_pool_;
  mutable std::mutex mtx_;
  mutable std::future<void> last_future_;
};

}  // namespace phi
