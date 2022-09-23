// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device/gpu/cuda/cuda_graph.h"
#endif

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
#define PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL(__cond,                            \
                                           __kernel_func,                     \
                                           __grid,                            \
                                           __block,                           \
                                           __sm_size,                         \
                                           __stream,                          \
                                           __seed_inc,                        \
                                           __seed_expr,                       \
                                           __offset_expr,                     \
                                           ...)                               \
  do {                                                                        \
    if (::paddle::platform::CUDAGraph::IsThisThreadCapturing() && (__cond)) { \
      using __Helper =                                                        \
          ::paddle::platform::IsSameKernelHelper<decltype(&__kernel_func),    \
                                                 &__kernel_func>;             \
      auto *dev_ctx =                                                         \
          ::paddle::platform::DeviceContextPool::Instance().GetByPlace(       \
              ::paddle::platform::CUDAGraph::CapturingPlace());               \
      auto __set_seed_func =                                                  \
          [=](::paddle::platform::CUDAKernelParams *__params,                 \
              bool __check_only) -> bool {                                    \
        if (__check_only) {                                                   \
          return __params->func() == &__kernel_func &&                        \
                 __Helper::Compare(*__params, __VA_ARGS__);                   \
        }                                                                     \
        auto &KERNEL_PARAMS = *__params;                                      \
        uint64_t __seed, __offset;                                            \
        ::paddle::operators::GetSeedDataAndIncrement(                         \
            *dev_ctx, nullptr, false, 0, __seed_inc, &__seed, &__offset);     \
        __seed_expr = static_cast<decltype(__seed_expr)>(__seed);             \
        __offset_expr = static_cast<decltype(__offset_expr)>(__offset);       \
        return true;                                                          \
      };                                                                      \
      ::paddle::platform::CUDAGraph::RecordRandomKernelInfo(__set_seed_func); \
    }                                                                         \
    __kernel_func<<<__grid, __block, __sm_size, __stream>>>(__VA_ARGS__);     \
  } while (0)
#else
#define PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL(__cond,                        \
                                           __kernel_func,                 \
                                           __grid,                        \
                                           __block,                       \
                                           __sm_size,                     \
                                           __stream,                      \
                                           __seed_inc,                    \
                                           __seed_expr,                   \
                                           __offset_expr,                 \
                                           ...)                           \
  do {                                                                    \
    __kernel_func<<<__grid, __block, __sm_size, __stream>>>(__VA_ARGS__); \
  } while (0)
#endif

// NOTE: These APIs are not thread-safe.
#ifdef PADDLE_WITH_CUDA
void BeginCUDAGraphCapture(platform::CUDAPlace place,
                           cudaStreamCaptureMode mode,
                           int64_t pool_id = CUDAGraph::kInvalidPoolID);
std::unique_ptr<CUDAGraph> EndCUDAGraphCapture();
#endif

inline bool IsCUDAGraphCapturing() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::IsCapturing();
#else
  return false;
#endif
}

inline platform::CUDAPlace CUDAGraphCapturingPlace() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::CapturingPlace();
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

// Add reset callback if CUDA Graph is capturing.
// Otherwise, invoke callback directly.
template <typename Callback>
inline void AddResetCallbackIfCapturingCUDAGraph(Callback &&callback) {
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    return CUDAGraph::AddResetCallbackDuringCapturing(
        std::forward<Callback>(callback));
  }
#endif
  callback();
}

template <typename T>
inline T *RestoreHostMemIfCapturingCUDAGraph(T *host_mem, size_t size) {
  static_assert(std::is_trivial<T>::value, "T must be trivial type");
  static_assert(!std::is_same<T, void>::value, "T cannot be void");
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    size_t nbytes = size * sizeof(T);
    void *new_host_mem = new uint8_t[nbytes];
    std::memcpy(new_host_mem, host_mem, nbytes);
    AddResetCallbackIfCapturingCUDAGraph(
        [new_host_mem] { delete[] reinterpret_cast<uint8_t *>(new_host_mem); });
    return reinterpret_cast<T *>(new_host_mem);
  }
#endif
  return host_mem;
}

class SkipCUDAGraphCaptureGuard {
  DISABLE_COPY_AND_ASSIGN(SkipCUDAGraphCaptureGuard);

 public:
  SkipCUDAGraphCaptureGuard() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      CUDAGraph::EndSegmentCapture();
    }
#endif
#endif
  }

  ~SkipCUDAGraphCaptureGuard() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (UNLIKELY(CUDAGraph::IsCapturing())) {
      CUDAGraph::BeginSegmentCapture();
    }
#endif
#endif
  }
};

}  // namespace platform
}  // namespace paddle
