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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif PADDLE_WITH_HIP
#include <hip/hip_runtime_api.h>
#endif
#include <functional>
#include <memory>
#include "ThreadPool.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
using StreamCallback = std::function<void(cudaStream_t, cudaError_t)>;
#elif PADDLE_WITH_HIP
using StreamCallback = std::function<void(hipStream_t, hipError_t)>;
#endif

class StreamCallbackManager;

struct StreamCallbackContext {
  template <typename Callback>
  inline StreamCallbackContext(const StreamCallbackManager *manager,
                               Callback &&callback)
      : manager_(manager), callback_(callback) {}

  const StreamCallbackManager *manager_;  // do not own
  StreamCallback callback_;
};

#ifdef PADDLE_WITH_CUDA
class StreamCallbackManager {
 public:
  explicit inline StreamCallbackManager(cudaStream_t stream = nullptr)
      : stream_(stream), thread_pool_(new ThreadPool(1)) {}

  template <typename Callback>
  inline void AddCallback(Callback &&callback) const {
    AddCallbackWithStreamAndErrorInfo(
        [=](cudaStream_t, cudaError_t) { callback(); });
  }

  template <typename Callback>
  inline void AddCallbackWithStreamAndErrorInfo(Callback &&callback) const {
    auto *stream_callback_context = new StreamCallbackContext(this, callback);
    PADDLE_ENFORCE(cudaStreamAddCallback(
        stream_, StreamCallbackManager::StreamCallbackFunc,
        stream_callback_context, 0));
  }

  void Wait() const { thread_pool_.reset(new ThreadPool(1)); }

 private:
  const cudaStream_t stream_;
  mutable std::unique_ptr<ThreadPool> thread_pool_;

  // cudaStreamCallback cannot call CUDA API inside, so we have to use
  // thread_pool here
  static void CUDART_CB StreamCallbackFunc(cudaStream_t stream,
                                           cudaError_t status,
                                           void *user_data) {
    auto *callback_context_ptr =
        reinterpret_cast<StreamCallbackContext *>(user_data);
    callback_context_ptr->manager_->thread_pool_->enqueue([=]() {
      std::unique_ptr<StreamCallbackContext> callback_context(
          callback_context_ptr);
      callback_context->callback_(stream, status);
    });
  }
};
#endif

#ifdef PADDLE_WITH_HIP
class StreamCallbackManager {
 public:
  explicit inline StreamCallbackManager(hipStream_t stream = nullptr)
      : stream_(stream), thread_pool_(new ThreadPool(1)) {}

  template <typename Callback>
  inline void AddCallback(Callback &&callback) const {
    AddCallbackWithStreamAndErrorInfo(
        [=](hipStream_t, hipError_t) { callback(); });
  }

  template <typename Callback>
  inline void AddCallbackWithStreamAndErrorInfo(Callback &&callback) const {
    auto *stream_callback_context = new StreamCallbackContext(this, callback);
    PADDLE_ENFORCE(hipStreamAddCallback(
        stream_, StreamCallbackManager::StreamCallbackFunc,
        stream_callback_context, 0));
  }

  void Wait() const { thread_pool_.reset(new ThreadPool(1)); }

 private:
  const hipStream_t stream_;
  mutable std::unique_ptr<ThreadPool> thread_pool_;

  // hipStreamCallback cannot call HIP API inside?
  // so we have to use thread_pool here
  static void StreamCallbackFunc(hipStream_t stream,
                                           hipError_t status,
                                           void *user_data) {
    auto *callback_context_ptr =
        reinterpret_cast<StreamCallbackContext *>(user_data);
    callback_context_ptr->manager_->thread_pool_->enqueue([=]() {
      std::unique_ptr<StreamCallbackContext> callback_context(
          callback_context_ptr);
      callback_context->callback_(stream, status);
    });
  }
};
#endif


}  // namespace platform
}  // namespace paddle
