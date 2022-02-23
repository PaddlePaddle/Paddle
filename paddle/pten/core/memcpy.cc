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

#include "paddle/pten/core/memcpy.h"

#include <cstring>

#include "paddle/pten/backends/gpu/gpu_info.h"
#include "paddle/pten/backends/xpu/xpu_info.h"
#include "paddle/pten/core/enforce.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/platform/device/device_manager.h"
#endif

namespace pten {

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
  if (src_place.GetType() == pten::AllocationType::CPU) {
    std::memcpy(dst, src, num);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_CUDA)
  } else if (src_place.GetType() == pten::AllocationType::GPU) {
    backends::gpu::SetDeviceId(dst_place.GetDeviceId());
    if (stream) {
      backends::gpu:: ::GpuMemcpyAsync(dst,
                                       src,
                                       num,
                                       gpuMemcpyHostToDevice,
                                       reinterpret_cast<gpuStream_t>(stream));
    } else {
      backends::gpu:: ::GpuMemcpySync(dst, src, num, gpuMemcpyHostToDevice);
    }
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncGPUStream();
    }
  } else if (src_place.GetType() == pten::AllocationType::GPUPINNED) {
#endif
#ifdef PADDLE_WITh_XPU
  } else if (src_place.GetType() == pten::AllocationType::XPU) {
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (src_place.GetType() == pten::AllocationType::CUSTOM) {
#endif
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
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
  if (src_place.GetType() == pten::AllocationType::CPU) {
  } else if (src_place.GetType() == pten::AllocationType::GPUPINNED) {
  } else if (src_place.GetType() == pten::AllocationType::GPU) {
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
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
  if (src_place.GetType() == pten::AllocationType::CPU) {
  } else if (src_place.GetType() == pten::AllocationType::GPUPINNED) {
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
        "Unsupported copy memory to GPUPinned from `%s` device.",
        src_place.GetType()));
  }
}
#endif

#ifdef PADDLE_WITh_XPU
static void MemcpyToXPU(const Place& dst_place,
                        void* dst,
                        const Place& src_place,
                        const void* src,
                        size_t num,
                        void* stream) {
  if (src_place.GetType() == pten::AllocationType::CPU) {
  } else if (src_place.GetType() == pten::AllocationType::XPU) {
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
        "Unsupported copy memory to XPU from `%s` device.",
        src_place.GetType()));
  }
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
static void MemcpyToCustomDevice(const Place& dst_place,
                                 void* dst,
                                 const Place& src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  if (src_place.GetType() == pten::AllocationType::CPU) {
  } else if (src_place.GetType() == pten::AllocationType::CUSTOM) {
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
        "Unsupported copy memory to CUSTOM device from `%s` device.",
        src_place.GetType()));
  }
}
#endif

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

  if (dst_place.GetType() == pten::AllocationType::CPU) {
    MemcpyToCPU(dst_place, dst, src_place, src, num, stream);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_CUDA)
  } else if (dst_place.GetType() == pten::AllocationType::GPU) {
    MemcpyToGPU(dst_place, dst, src_place, src, num, stream);
  } else if (dst_place.GetType() == pten::AllocationType::GPUPINNED) {
    MemcpyToGPUPinned(dst_place, dst, src_place, src, num, stream);
#endif
#ifdef PADDLE_WITh_XPU
  } else if (dst_place.GetType() == pten::AllocationType::XPU) {
    MemcpyToXPU(dst_place, dst, src_place, src, num, stream);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (dst_place.GetType() == pten::AllocationType::CUSTOM) {
    MemcpyToCustomDevice(dst_place, dst, src_place, src, num, stream);
#endif
  } else {
    PADDLE_THROW(pten::errors::Unavailable(
        "Unsupported copy memory to `%s` device.", dst_place.GetType()));
  }
}

}  // namespace pten
