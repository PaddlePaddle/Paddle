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

#include <ThreadPool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace paddle {
namespace platform {

// NOTE(zjl): clean StreamCallback to make compilation faster
class StreamCallbackManager {
 public:
  explicit StreamCallbackManager(const cudaStream_t stream);

  void AddCallback(std::function<void()> callback) const;

  void Wait() const;

 private:
  const cudaStream_t stream_;
  mutable std::unique_ptr<::ThreadPool> thread_pool_;

#if CUDA_VERSION >= 10000
  static void CUDART_CB StreamCallbackFunc(void *user_data);
#else
  static void CUDART_CB StreamCallbackFunc(cudaStream_t stream,
                                           cudaError_t status, void *user_data);
#endif
};

}  // namespace platform
}  // namespace paddle
