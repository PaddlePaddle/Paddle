// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <ThreadPool.h>

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

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {

// NOTE(zjl): clean StreamCallbackManager to make compilation faster
// Make StreamCallbackManager thread-safe
template <typename Stream>
class StreamCallbackManager {
 public:
  explicit StreamCallbackManager(const Stream stream);

  ~StreamCallbackManager() = default;

  void AddCallback(std::function<void()> callback) const;

  void Wait() const;

 private:
  const Stream stream_;
  mutable ::ThreadPool thread_pool_;
  mutable std::mutex mtx_;
  mutable std::future<void> last_future_;
};

}  // namespace platform
}  // namespace paddle
