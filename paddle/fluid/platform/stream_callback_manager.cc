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
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_HIP
static void StreamCallbackFunc(gpuStream_t stream, gpuError_t status,
                               void *user_data)
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
    static void CUDART_CB StreamCallbackFunc(void *user_data)
#else
    static void CUDART_CB
    StreamCallbackFunc(cudaStream_t stream, cudaError_t status, void *user_data)
#endif
#endif

#if PADDLE_WITH_ASCEND_CL
        static void StreamCallbackFunc(void *user_data)
#endif
{
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()> *>(user_data));
  (*func)();
}

template <typename Stream>
StreamCallbackManager<Stream>::StreamCallbackManager(const Stream stream)
    : stream_(stream), thread_pool_(1) {}

template <typename Stream>
void StreamCallbackManager<Stream>::AddCallback(
    std::function<void()> callback) const {
  auto *callback_func = new std::function<void()>(std::move(callback));
  auto *func = new std::function<void()>([this, callback_func] {
    std::lock_guard<std::mutex> lock(mtx_);
    last_future_ = thread_pool_.enqueue([callback_func] {
      std::unique_ptr<std::function<void()>> releaser(callback_func);
      (*callback_func)();
    });
  });

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_CUDA_SUCCESS(
      hipStreamAddCallback(stream_, StreamCallbackFunc, func, 0));
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaLaunchHostFunc(stream_, StreamCallbackFunc, func));
#else
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamAddCallback(stream_, StreamCallbackFunc, func, 0));
#endif
#endif

#if PADDLE_WITH_ASCEND_CL
  VLOG(3) << "aclrtLaunchCallback at stream: " << stream_;
  // TODO(zhiqiu): failed to call aclrtLaunchCallback
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtLaunchCallback(StreamCallbackFunc, func,
                                                 ACL_CALLBACK_BLOCK, stream_));
#endif
}

template <typename Stream>
void StreamCallbackManager<Stream>::Wait() const {
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream_));
#endif
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  platform::Timer t;
  t.Start();
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream_));
  t.Pause();
  VLOG(5) << "sync stream elapsed " << t.ElapsedMS();
#endif
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (last_future_.valid()) {
      last_future_.wait();
    }
  }
}

#ifdef PADDLE_WITH_CUDA
template struct StreamCallbackManager<gpuStream_t>;
#endif
#ifdef PADDLE_WITH_HIP
template struct StreamCallbackManager<hipStream_t>;
#endif
#ifdef PADDLE_WITH_ASCEND_CL
template struct StreamCallbackManager<aclrtStream>;
#endif

}  // namespace platform
}  // namespace paddle
