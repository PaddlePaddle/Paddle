/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/memcpy.h"

#include <cstring>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
static constexpr size_t kMaxGpuAsyncCopyBytes = 64 * 1024;  // 64K

inline void SyncGPUStream() {
#if !defined(_WIN32)
  GpuMemcpySync(0);
#else
  gpuError_t e_sync = gpuSuccess;
  while (e_sync = gpuStreamQuery(0)) {
    if (e_sync == gpuErrorNotReady) continue;
    break;
  }
#endif
}
#endif

// Note: Just to make the function body not too complicated, the Memcpy function
// implementation is split, but MemcpyToXXX function is not recommended,
// please use Memcpy directly

static void MemcpyToCPU(const Place& dst_place,
                        void* dst,
                        const Place& src_place,
                        const void* src,
                        size_t num,
                        void* stream) {
  if (src_place.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst, src, num);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_CUDA)
  } else if (src_place.GetType() == phi::AllocationType::GPU) {
    backends::gpu::SetDeviceId(src_place.GetDeviceId());
    if (stream) {
      backends::gpu::GpuMemcpyAsync(dst,
                                    src,
                                    num,
                                    gpuMemcpyDeviceToHost,
                                    reinterpret_cast<gpuStream_t>(stream));
    } else {
      backends::gpu::GpuMemcpySync(dst, src, num, gpuMemcpyDeviceToHost);
      // FIXME(zjl): do we really need it?
      if (num <= kMaxGpuAsyncCopyBytes) {
        SyncGPUStream();
      }
    }
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
#endif
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Unsupported copy memory to CPU from `%s` device.",
        src_place.GetType()));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_CUDA)
static void MemcpyToGPU(const Place& dst_place,
                        void* dst,
                        const Place& src_place,
                        const void* src,
                        size_t num,
                        void* stream) {
  if (src_place.GetType() == phi::AllocationType::CPU) {
    backends::gpu::SetDeviceId(dst_place.GetDeviceId());
    if (stream) {
      backends::gpu::GpuMemcpyAsync(dst,
                                    src,
                                    num,
                                    gpuMemcpyHostToDevice,
                                    reinterpret_cast<gpuStream_t>(stream));
    } else {
      backends::gpu::GpuMemcpySync(dst, src, num, gpuMemcpyHostToDevice);
    }
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncGPUStream();
    }
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED) {
    backends::gpu::SetDeviceId(dst_place.GetDeviceId());
    if (stream) {
      backends::gpu::GpuMemcpyAsync(dst,
                                    src,
                                    num,
                                    gpuMemcpyHostToDevice,
                                    reinterpret_cast<gpuStream_t>(stream));
    } else {
      backends::gpu::GpuMemcpySync(dst, src, num, gpuMemcpyHostToDevice);
    }
  } else if (src_place.GetType() == phi::AllocationType::GPU) {
    if (dst_place == src_place) {
      backends::gpu::SetDeviceId(dst_place.GetDeviceId());
      if (stream) {
        backends::gpu::GpuMemcpyAsync(dst,
                                      src,
                                      num,
                                      gpuMemcpyDeviceToDevice,
                                      reinterpret_cast<gpuStream_t>(stream));
      } else {
        backends::gpu::GpuMemcpySync(dst, src, num, gpuMemcpyDeviceToDevice);
      }
    } else {
      if (stream) {
        backends::gpu::GpuMemcpyPeerAsync(
            dst,
            dst_place.device,
            src,
            src_place.device,
            num,
            reinterpret_cast<gpuStream_t>(stream));
      } else {
        backends::gpu::GpuMemcpyPeerSync(
            dst, dst_place.device, src, src_place.device, num);
      }
    }
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Unsupported copy memory to GPU from `%s` device.",
        src_place.GetType()));
  }
}

static void MemcpyToGPUPinned(const Place& dst_place,
                              void* dst,
                              const Place& src_place,
                              const void* src,
                              size_t num,
                              void* stream) {
  if (src_place.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPU) {
    backends::gpu::SetDeviceId(src_place.device);
    if (stream) {
      backends::gpu::GpuMemcpyAsync(dst,
                                    src,
                                    num,
                                    gpuMemcpyDeviceToHost,
                                    reinterpret_cast<gpuStream_t>(stream));
    } else {
      backends::gpu::GpuMemcpySync(dst, src, num, gpuMemcpyDeviceToHost);
    }
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Unsupported copy memory to GPUPinned from `%s` device.",
        src_place.GetType()));
  }
}
#endif

// TODO(chenweihang): support xpu and custom device copy later
// NOTE(chenweihang): xpu copy needs to pass in DeviceContext, which causes
// the previous internal implementation of memory::Copy to couple a singleton,
// which is not allowed under phi!!! We need to modify the MemcpySyncD2H and
// MemcpySyncD2D methods of xpu, and change their last parameter to pass
// din XContext* or stream*
void Memcpy(const Place& dst_place,
            void* dst,
            const Place& src_place,
            const void* src,
            size_t num,
            void* stream) {
  if (UNLIKELY(num == 0)) {
    return;
  }

  VLOG(4) << "Memcpy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream " << stream;

  if (dst_place.GetType() == phi::AllocationType::CPU) {
    MemcpyToCPU(dst_place, dst, src_place, src, num, stream);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_CUDA)
  } else if (dst_place.GetType() == phi::AllocationType::GPU) {
    MemcpyToGPU(dst_place, dst, src_place, src, num, stream);
  } else if (dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    MemcpyToGPUPinned(dst_place, dst, src_place, src, num, stream);
#endif
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Unsupported copy memory to `%s` device.", dst_place.GetType()));
  }
}

}  // namespace phi
