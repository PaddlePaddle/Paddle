/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include "paddle/pten/core/backends/gpu/cuda_context.h"
#include <memory>
#include "paddle/pten/core/allocator.h"

#include "paddle/fluid/platform/enforce.h"

namespace pten {

dim3 CUDAContext::GetCUDAMaxGridDimSize() const {
  dim3 ret;
  ret.x = max_grid_dim_x_;
  ret.y = max_grid_dim_y_;
  ret.z = max_grid_dim_z_;
  return ret;
}

template <typename Callback>
void CUDAContext::AddStreamCallback(Callback&& callback) {
  if (callback_manager_ == nullptr) {
    callback_manager_.reset(new StreamCallbackManager(stream_));
  }
  callback_manager_->AddCallback(callback);
}

void CUDAContext::WaitStreamCallback() {
  if (callback_manager_ == nullptr) {
    callback_manager_.reset(new StreamCallbackManager(stream_));
  }
  callback_manager_->Wait();
}

void CUDAContext::Wait(cudaStream_t stream) const {
  cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
  e_sync = cudaStreamSynchronize(stream);
#else
  while (e_sync = cudaStreamQuery(stream)) {
    if (e_sync == cudaErrorNotReady) continue;
    break;
  }
#endif

  PADDLE_ENFORCE_CUDA_SUCCESS(e_sync);
}

void CUDAContext::Wait() { Wait(this->stream()); }

cublasHandle_t CUDAContext::cublas_handle() const { return cublas_handle_; }

CudnnWorkspaceHandle::CudnnWorkspaceHandle(pten::Allocator* allocator,
                                           std::mutex* mtx)
    : allocator_(allocator), mtx_(mtx) {}

template <typename Callback>
inline void CudnnWorkspaceHandle::RunFunc(Callback&& cudnn_func,
                                          size_t required_workspace_bytes) {
  if (required_workspace_bytes > WorkspaceSize()) {
    ReallocWorkspace(required_workspace_bytes);
  }
  VLOG(2) << "Cudnn workspace size at RunFunc: "
          << static_cast<double>(WorkspaceSize()) / (1 << 20) << " MB";
  {
    std::lock_guard<std::mutex> guard(*mtx_);
    cudnn_func(allocation_.operator->());
  }
}

/*! \brief Thread which call RunFuncSync() would release gpu memory after
  *  running the function. Currently this function is only used when cudnn
  *  exhaustive searching and callers have to guarantee that the input function
  *  is host blocking */
template <typename Callback>
inline void CudnnWorkspaceHandle::RunFuncSync(Callback&& cudnn_func,
                                              size_t required_workspace_bytes) {
  RunFunc(cudnn_func, required_workspace_bytes);
  ResetWorkspace();
}

void CudnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
  if (required_workspace_bytes <= WorkspaceSize()) {
    return;
  }
  // reset allocation first before re-allocate to save memory
  allocation_.Clear();
  // allocation_ = memory::Alloc(device_context_, required_workspace_bytes);
  allocation_ = allocator_->Allocate(required_workspace_bytes);
  num_bytes_ = required_workspace_bytes;
}

inline void CudnnWorkspaceHandle::ResetWorkspace() {
  allocation_.Clear();
  num_bytes_ = 0;
}

inline size_t CudnnWorkspaceHandle::WorkspaceSize() { return num_bytes_; }

#if CUDA_VERSION >= 10000
static void CUDART_CB StreamCallbackFunc(void* user_data)
#else
static void CUDART_CB StreamCallbackFunc(cudaStream_t stream,
                                         cudaError_t status,
                                         void* user_data)
#endif
{
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()>*>(user_data));
  (*func)();
}

StreamCallbackManager::StreamCallbackManager(const cudaStream_t stream)
    : stream_(stream), thread_pool_(1) {}

void StreamCallbackManager::AddCallback(std::function<void()> callback) const {
  auto* callback_func = new std::function<void()>(std::move(callback));
  auto* func = new std::function<void()>([this, callback_func] {
    std::lock_guard<std::mutex> lock(mtx_);
    last_future_ = thread_pool_.enqueue([callback_func] {
      std::unique_ptr<std::function<void()>> releaser(callback_func);
      (*callback_func)();
    });
  });

#if CUDA_VERSION >= 10000
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaLaunchHostFunc(stream_, StreamCallbackFunc, func));
#else
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamAddCallback(stream_, StreamCallbackFunc, func, 0));
#endif
}

void StreamCallbackManager::Wait() const {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
#endif
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (last_future_.valid()) {
      last_future_.wait();
    }
  }
}

#ifdef PADDLE_WITH_CUDNN
CudnnWorkspaceHandle* CUDAContext::cudnn_workspace_handle() {
  if (cudnn_workspace_handle_ == nullptr) {
    cudnn_workspace_handle_.reset(
        new CudnnWorkspaceHandle(allocator_, &cudnn_handle_mtx_));
  }
  return cudnn_workspace_handle_.get();
}

cudnnHandle_t CUDAContext::cudnn_handle() const { return cudnn_handle_; }
#endif

}  // namespace pten
