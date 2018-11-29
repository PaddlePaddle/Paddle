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

#include "paddle/fluid/platform/stream_callback_manager.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

struct StreamCallbackContext {
  inline StreamCallbackContext(const StreamCallbackManager *manager,
                               std::function<void()> callback)
      : manager_(manager), callback_(std::move(callback)) {}

  const StreamCallbackManager *manager_;  // do not own
  std::function<void()> callback_;
};

StreamCallbackManager::StreamCallbackManager(const cudaStream_t stream)
    : stream_(stream), thread_pool_(new ::ThreadPool(1)) {}

void StreamCallbackManager::AddCallback(std::function<void()> callback) const {
  auto *stream_callback_context =
      new StreamCallbackContext(this, std::move(callback));
#if CUDA_VERSION >= 10000
  PADDLE_ENFORCE(cudaLaunchHostFunc(stream_,
                                    StreamCallbackManager::StreamCallbackFunc,
                                    stream_callback_context));
#else
  PADDLE_ENFORCE(
      cudaStreamAddCallback(stream_, StreamCallbackManager::StreamCallbackFunc,
                            stream_callback_context, 0));
#endif
}

void StreamCallbackManager::Wait() const {
  thread_pool_.reset(new ::ThreadPool(1));
}

#if CUDA_VERSION >= 10000
void CUDART_CB StreamCallbackManager::StreamCallbackFunc(void *user_data)
#else
void CUDART_CB StreamCallbackManager::StreamCallbackFunc(cudaStream_t stream,
                                                         cudaError_t status,
                                                         void *user_data)
#endif
{
  auto *callback_context_ptr =
      reinterpret_cast<StreamCallbackContext *>(user_data);
  callback_context_ptr->manager_->thread_pool_->enqueue(
      [callback_context_ptr]() {
        std::unique_ptr<StreamCallbackContext> callback_context(
            callback_context_ptr);
        callback_context->callback_();
      });
}

}  // namespace platform
}  // namespace paddle
