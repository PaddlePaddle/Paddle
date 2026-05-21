/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/memory/memcpy.h"
#include "glog/logging.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/test_macros.h"

#ifdef PADDLE_WITH_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#endif

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_default_stream);

namespace paddle::memory {

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <>
void Copy<CPUPlace, CustomPlace>(CPUPlace dst_place,
                                 void* dst,
                                 CustomPlace src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  if (UNLIKELY(num == 0)) return;

  auto src_type = phi::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = phi::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  phi::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  phi::DeviceManager::SetDevice(src_place);
  phi::stream::Stream stream_wrapper(src_place, stream);
  phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyD2H(
      dst, src, num, &stream_wrapper);
}

template <>
void Copy<CustomPlace, CPUPlace>(CustomPlace dst_place,
                                 void* dst,
                                 CPUPlace src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  if (UNLIKELY(num == 0)) return;
  auto src_type = phi::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = phi::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  phi::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  phi::DeviceManager::SetDevice(dst_place);
  phi::stream::Stream stream_wrapper(dst_place, stream);
  phi::DeviceManager::GetDeviceWithPlace(dst_place)->MemoryCopyH2D(
      dst, src, num, &stream_wrapper);
}

template <>
void Copy<CustomPlace, CustomPlace>(CustomPlace dst_place,
                                    void* dst,
                                    CustomPlace src_place,
                                    const void* src,
                                    size_t num,
                                    void* stream) {
  if (UNLIKELY(num == 0)) return;

  auto src_type = phi::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = phi::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  phi::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  if (src_type == dst_type) {
    phi::DeviceManager::SetDevice(src_place);
    phi::stream::Stream stream_wrapper(src_place, stream);

    auto src_id = phi::PlaceHelper::GetDeviceId(src_place);
    auto dst_id = phi::PlaceHelper::GetDeviceId(dst_place);
    if (src_id == dst_id) {
      phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyD2D(
          dst, src, num, &stream_wrapper);
    } else {
      phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyP2P(
          dst_place, dst, src, num, &stream_wrapper);
    }
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Copy between %s and %s is not supported.", src_type, dst_type));
  }
}
#endif  // PADDLE_WITH_CUSTOM_DEVICE

template <>
PADDLE_API void Copy<CPUPlace, CPUPlace>(
    CPUPlace, void* dst, CPUPlace, const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "src: " << src << ", dst: " << dst << ", num: " << num;
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_IPU
template <>
void Copy<IPUPlace, CPUPlace>(IPUPlace dst_place,
                              void* dst,
                              CPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<CPUPlace, IPUPlace>(CPUPlace dst_place,
                              void* dst,
                              IPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<IPUPlace, IPUPlace>(IPUPlace dst_place,
                              void* dst,
                              IPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

// NOTE: only for (CPUPlace and IPUPlace) -> (IPUPlace).
template <>
void Copy<IPUPlace, Place>(IPUPlace dst_place,
                           void* dst,
                           Place src_place,
                           const void* src,
                           size_t num) {
  if (src_place.GetType() == AllocationType::CPU) {
    CPUPlace place_src;
    return Copy(dst_place, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::IPU) {
    IPUPlace place_src(src_place.GetDeviceId());
    return Copy(dst_place, dst, place_src, src, num);
  }
}

// NOTE: only for (IPUPlace) -> (CPUPlace and IPUPlace).
template <>
void Copy<Place, IPUPlace>(Place dst_place,
                           void* dst,
                           IPUPlace src_place,
                           const void* src,
                           size_t num) {
  if (dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst;
    return Copy(place_dst, dst, src_place, src, num);
  } else if (dst_place.GetType() == AllocationType::IPU) {
    IPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, src_place, src, num);
  }
}
#endif

#ifdef PADDLE_WITH_XPU
template <>
void Copy<XPUPlace, CPUPlace>(XPUPlace dst_place,
                              void* dst,
                              CPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_HOST_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncH2D(dst, src, num, dst_place);
}

template <>
void Copy<CPUPlace, XPUPlace>(CPUPlace dst_place,
                              void* dst,
                              XPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_HOST size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2H(dst, src, num, src_place);
}

template <>
void Copy<XPUPlace, XPUPlace>(XPUPlace dst_place,
                              void* dst,
                              XPUPlace src_place,
                              const void* src,
                              size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2D(dst, dst_place, src, src_place, num);
}

// NOTE: only for (CPUPlace and XPUPlace) -> (XPUPlace).
template <>
void Copy<XPUPlace, Place>(XPUPlace dst_place,
                           void* dst,
                           Place src_place,
                           const void* src,
                           size_t num) {
  if (src_place.GetType() == AllocationType::CPU) {
    CPUPlace place_src;
    return Copy(dst_place, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPU) {
    XPUPlace place_src(src_place.GetDeviceId());
    return Copy(dst_place, dst, place_src, src, num);
  }
}

// NOTE: only for (XPUPlace) -> (CPUPlace and XPUPlace).
template <>
void Copy<Place, XPUPlace>(Place dst_place,
                           void* dst,
                           XPUPlace src_place,
                           const void* src,
                           size_t num) {
  if (dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst;
    return Copy(place_dst, dst, src_place, src, num);
  } else if (dst_place.GetType() == AllocationType::XPU) {
    XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, src_place, src, num);
  }
}

template <>
void Copy<XPUPlace, CPUPlace>(XPUPlace dst_place,
                              void* dst,
                              CPUPlace src_place,
                              const void* src,
                              size_t num,
                              void* stream) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_HOST_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  // platform::MemcpySyncH2D(dst, src, num, dst_place);
  xpu_memcpy_async(dst, src, num, XPU_HOST_TO_DEVICE, stream);
}

template <>
void Copy<CPUPlace, XPUPlace>(CPUPlace dst_place,
                              void* dst,
                              XPUPlace src_place,
                              const void* src,
                              size_t num,
                              void* stream) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_HOST size <= 0 (" << num << ")";
    return;
  }
  // platform::MemcpySyncD2H(dst, src, num, src_place);
  xpu_memcpy_async(dst, src, num, XPU_DEVICE_TO_HOST, stream);
}

template <>
void Copy<XPUPlace, XPUPlace>(XPUPlace dst_place,
                              void* dst,
                              XPUPlace src_place,
                              const void* src,
                              size_t num,
                              void* stream) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2D(dst, dst_place, src, src_place, num);
}

// NOTE: only for (CPUPlace and XPUPlace) -> (XPUPlace).
template <>
void Copy<XPUPlace, Place>(XPUPlace dst_place,
                           void* dst,
                           Place src_place,
                           const void* src,
                           size_t num,
                           void* stream) {
  if (src_place.GetType() == AllocationType::CPU) {
    CPUPlace place_src;
    return Copy(dst_place, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPU) {
    XPUPlace place_src(src_place.GetDeviceId());
    return Copy(dst_place, dst, place_src, src, num);
  }
}

// NOTE: only for (XPUPlace) -> (CPUPlace and XPUPlace).
template <>
void Copy<Place, XPUPlace>(Place dst_place,
                           void* dst,
                           XPUPlace src_place,
                           const void* src,
                           size_t num,
                           void* stream) {
  if (dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst;
    return Copy(place_dst, dst, src_place, src, num);
  } else if (dst_place.GetType() == AllocationType::XPU) {
    XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, src_place, src, num);
  }
}

template <>
void Copy<CPUPlace, XPUPinnedPlace>(CPUPlace dst_place,
                                    void* dst,
                                    XPUPinnedPlace src_place,
                                    const void* src,
                                    size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
PADDLE_API void Copy<XPUPinnedPlace, CPUPlace>(XPUPinnedPlace dst_place,
                                               void* dst,
                                               CPUPlace src_place,
                                               const void* src,
                                               size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<XPUPinnedPlace, XPUPinnedPlace>(XPUPinnedPlace dst_place,
                                          void* dst,
                                          XPUPinnedPlace src_place,
                                          const void* src,
                                          size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<XPUPinnedPlace, XPUPlace>(XPUPinnedPlace dst_place,
                                    void* dst,
                                    XPUPlace src_place,
                                    const void* src,
                                    size_t num,
                                    void* stream) {
  if (UNLIKELY(num == 0)) return;
  platform::SetXPUDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";

  // Record start time using std::chrono
  auto start = std::chrono::high_resolution_clock::now();

  if (stream) {
    phi::RecordEvent record_event(
        "cudaMemcpyAsync:XPU->XPUPinned", phi::TracerEventType::UserDefined, 1);
    cudaMemcpyAsync(dst,
                    src,
                    num,
                    cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(stream));

  } else {
    cudaDeviceSynchronize();
    phi::RecordEvent record_event(
        "cudaMemcpy:XPU->XPUPinned", phi::TracerEventType::UserDefined, 1);
    cudaMemcpy(dst, src, num, cudaMemcpyDeviceToHost);
  }

  // Record end time and calculate elapsed time in milliseconds
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  VLOG(4) << "cudaMemcpy time: " << elapsed.count() << " ms";
}

template <>
void Copy<XPUPlace, XPUPinnedPlace>(XPUPlace dst_place,
                                    void* dst,
                                    XPUPinnedPlace src_place,
                                    const void* src,
                                    size_t num,
                                    void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetXPUDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";

  // Record start time using std::chrono
  auto start = std::chrono::high_resolution_clock::now();

  if (stream) {
    phi::RecordEvent record_event(
        "cudaMemcpyAsync:XPUPinned->XPU", phi::TracerEventType::UserDefined, 1);
    cudaMemcpyAsync(dst,
                    src,
                    num,
                    cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream));
  } else {
    cudaDeviceSynchronize();
    phi::RecordEvent record_event(
        "cudaMemcpy:XPUPinned->XPU", phi::TracerEventType::UserDefined, 1);
    cudaMemcpy(dst, src, num, cudaMemcpyHostToDevice);
  }

  // Synchronize to ensure the memcpy operation is finished
  if (stream) {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  } else {
    cudaDeviceSynchronize();
  }

  // Record end time and calculate elapsed time in milliseconds
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  VLOG(4) << "cudaMemcpy time: " << elapsed.count() << " ms";
}

// NOTE: for XPU and XPUPINNED.
template <>
PADDLE_API void Copy<Place, Place>(Place dst_place,
                                   void* dst,
                                   Place src_place,
                                   const void* src,
                                   size_t num,
                                   void* stream) {
  if (src_place.GetType() == AllocationType::XPUPINNED &&
      dst_place.GetType() == AllocationType::XPU) {
    XPUPinnedPlace place_src;
    XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::XPU &&
             dst_place.GetType() == AllocationType::XPUPINNED) {
    XPUPinnedPlace place_dst;
    XPUPlace place_src(src_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else {
    PADDLE_THROW(::common::errors::Unimplemented(
        "Asynchronous Copy from %s to %s is not supported.",
        src_place,
        dst_place));
  }
}

template <>
void Copy<XPUPinnedPlace, Place>(XPUPinnedPlace dst_place,
                                 void* dst,
                                 Place src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

template <>
void Copy<Place, XPUPinnedPlace>(Place dst_place,
                                 void* dst,
                                 XPUPinnedPlace src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, stream);
}

template <>
void Copy<XPUPinnedPlace, Place>(XPUPinnedPlace dst_place,
                                 void* dst,
                                 Place src_place,
                                 const void* src,
                                 size_t num) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, nullptr);
}

template <>
void Copy<Place, XPUPinnedPlace>(Place dst_place,
                                 void* dst,
                                 XPUPinnedPlace src_place,
                                 const void* src,
                                 size_t num) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, nullptr);
}

#endif

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(PADDLE_WITH_CUSTOM_DEVICE)
static constexpr size_t kMaxGpuAsyncCopyBytes = 64 * 1024;  // 64K

#ifdef PADDLE_WITH_HIP
inline void SyncCUDAStream() {
#if !defined(_WIN32)
  hipStreamSynchronize(0);
#else
  hipError_t e_sync = hipSuccess;
  while (e_sync = hipStreamQuery(0)) {
    if (e_sync == hipErrorNotReady) continue;
    break;
  }
#endif
}
#else
inline void SyncCUDAStream() {
#if !defined(_WIN32)
  cudaStreamSynchronize(nullptr);
#else
  cudaError_t e_sync = cudaSuccess;
  while (e_sync = cudaStreamQuery(0)) {
    if (e_sync == cudaErrorNotReady) continue;
    break;
  }
#endif
}
#endif

// NOTE(zcd): Do not use GpuMemcpySync as much as possible.
// because GpuMemcpySync issues the copying command to the default stream,
// which will make two commands from different streams cannot run concurrently.
// Reference:
// https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

template <>
PADDLE_API void Copy<CPUPlace, GPUPlace>(CPUPlace dst_place,
                                         void* dst,
                                         GPUPlace src_place,
                                         const void* src,
                                         size_t num,
                                         void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream || FLAGS_use_default_stream) {
    phi::RecordEvent record_event(
        "GpuMemcpyAsync:GPU->CPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             hipMemcpyDeviceToHost,
                             reinterpret_cast<gpuStream_t>(stream));
#else
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             cudaMemcpyDeviceToHost,
                             reinterpret_cast<gpuStream_t>(stream));
#endif
  } else {
    phi::RecordEvent record_event(
        "GpuMemcpySync:GPU->CPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncCUDAStream();
    }
  }
}

template <>
PADDLE_API void Copy<GPUPlace, CPUPlace>(GPUPlace dst_place,
                                         void* dst,
                                         CPUPlace src_place,
                                         const void* src,
                                         size_t num,
                                         void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream || FLAGS_use_default_stream) {
    phi::RecordEvent record_event(
        "GpuMemcpyAsync:CPU->GPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             hipMemcpyHostToDevice,
                             reinterpret_cast<gpuStream_t>(stream));
#else
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             cudaMemcpyHostToDevice,
                             reinterpret_cast<gpuStream_t>(stream));
#endif
  } else {
    phi::RecordEvent record_event(
        "GpuMemcpySync:CPU->GPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      SyncCUDAStream();
    }
  }
}

template <>
PADDLE_API void Copy<GPUPlace, GPUPlace>(GPUPlace dst_place,
                                         void* dst,
                                         GPUPlace src_place,
                                         const void* src,
                                         size_t num,
                                         void* stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(7) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream || FLAGS_use_default_stream) {
      phi::RecordEvent record_event("GpuMemcpyAsync(same_gpu):GPU->GPU",
                                    phi::TracerEventType::UserDefined,
                                    1);
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpyAsync(dst,
                               src,
                               num,
                               hipMemcpyDeviceToDevice,
                               reinterpret_cast<gpuStream_t>(stream));
#else
      platform::GpuMemcpyAsync(dst,
                               src,
                               num,
                               cudaMemcpyDeviceToDevice,
                               reinterpret_cast<gpuStream_t>(stream));
#endif
    } else {
      phi::RecordEvent record_event("GpuMemcpySync(same_gpu):GPU->GPU",
                                    phi::TracerEventType::UserDefined,
                                    1);
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToDevice);
#else
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
#endif
    }
  } else {
    if (stream) {
      phi::RecordEvent record_event(
          "GpuMemcpyPeerAsync:GPU->GPU", phi::TracerEventType::UserDefined, 1);
      platform::GpuMemcpyPeerAsync(dst,
                                   dst_place.device,
                                   src,
                                   src_place.device,
                                   num,
                                   reinterpret_cast<gpuStream_t>(stream));
    } else {
      phi::RecordEvent record_event(
          "GpuMemcpyPeerSync:GPU->GPU", phi::TracerEventType::UserDefined, 1);
      platform::GpuMemcpyPeerSync(
          dst, dst_place.device, src, src_place.device, num);
    }
  }
}

template <>
void Copy<CPUPlace, GPUPinnedPlace>(CPUPlace dst_place,
                                    void* dst,
                                    GPUPinnedPlace src_place,
                                    const void* src,
                                    size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
PADDLE_API void Copy<GPUPinnedPlace, CPUPlace>(GPUPinnedPlace dst_place,
                                               void* dst,
                                               CPUPlace src_place,
                                               const void* src,
                                               size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<GPUPinnedPlace, GPUPinnedPlace>(GPUPinnedPlace dst_place,
                                          void* dst,
                                          GPUPinnedPlace src_place,
                                          const void* src,
                                          size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<GPUPinnedPlace, GPUPlace>(GPUPinnedPlace dst_place,
                                    void* dst,
                                    GPUPlace src_place,
                                    const void* src,
                                    size_t num,
                                    void* stream) {
  if (UNLIKELY(num == 0)) return;
  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream || FLAGS_use_default_stream) {
    phi::RecordEvent record_event(
        "GpuMemcpyAsync:GPU->CUDAPinned", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             hipMemcpyDeviceToHost,
                             reinterpret_cast<gpuStream_t>(stream));
#else
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             cudaMemcpyDeviceToHost,
                             reinterpret_cast<gpuStream_t>(stream));
#endif
  } else {
    phi::RecordEvent record_event(
        "GpuMemcpySync:GPU->CUDAPinned", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
  }
}

template <>
void Copy<GPUPlace, GPUPinnedPlace>(GPUPlace dst_place,
                                    void* dst,
                                    GPUPinnedPlace src_place,
                                    const void* src,
                                    size_t num,
                                    void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream || FLAGS_use_default_stream) {
    phi::RecordEvent record_event(
        "GpuMemcpyAsync:CUDAPinned->GPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             hipMemcpyHostToDevice,
                             reinterpret_cast<gpuStream_t>(stream));
#else
    platform::GpuMemcpyAsync(dst,
                             src,
                             num,
                             cudaMemcpyHostToDevice,
                             reinterpret_cast<gpuStream_t>(stream));
#endif
  } else {
    phi::RecordEvent record_event(
        "GpuMemcpySync:CUDAPinned->GPU", phi::TracerEventType::UserDefined, 1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
  }
}

// NOTE: only for CPUPlace、CUDAPlace and CUDAPinnedPlace.
template <>
PADDLE_API void Copy<Place, Place>(Place dst_place,
                                   void* dst,
                                   Place src_place,
                                   const void* src,
                                   size_t num,
                                   void* stream) {
  if (src_place.GetType() == AllocationType::CPU &&
      dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst, place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::CPU &&
             dst_place.GetType() == AllocationType::GPU) {
    GPUPlace place_dst(dst_place.GetDeviceId());
    CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::GPU &&
             dst_place.GetType() == AllocationType::CPU) {
    GPUPlace place_src(src_place.GetDeviceId());
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::GPU &&
             dst_place.GetType() == AllocationType::GPU) {
    GPUPlace place_src(src_place.GetDeviceId());
    GPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::CPU &&
             dst_place.GetType() == AllocationType::GPUPINNED) {
    CPUPlace place_src;
    GPUPinnedPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::GPUPINNED &&
             dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst;
    GPUPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::GPUPINNED &&
             dst_place.GetType() == AllocationType::GPUPINNED) {
    GPUPinnedPlace place_dst;
    GPUPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::GPUPINNED &&
             dst_place.GetType() == AllocationType::GPU) {
    GPUPinnedPlace place_src;
    GPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::GPU &&
             dst_place.GetType() == AllocationType::GPUPINNED) {
    GPUPinnedPlace place_dst;
    GPUPlace place_src(src_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  }
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CPUPlace).
template <>
PADDLE_API void Copy<CPUPlace, Place>(CPUPlace dst_place,
                                      void* dst,
                                      Place src_place,
                                      const void* src,
                                      size_t num,
                                      void* stream) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace).
template <>
PADDLE_API void Copy<Place, CPUPlace>(Place dst_place,
                                      void* dst,
                                      CPUPlace src_place,
                                      const void* src,
                                      size_t num,
                                      void* stream) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, stream);
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CUDAPlace)
template <>
void Copy<GPUPlace, Place>(GPUPlace dst_place,
                           void* dst,
                           Place src_place,
                           const void* src,
                           size_t num,
                           void* stream) {
  Copy(Place(dst_place.GetType(), dst_place.GetDeviceId()),
       dst,
       src_place,
       src,
       num,
       stream);
}

// NOTE: only for (CUDAPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace)
template <>
PADDLE_API void Copy<Place, GPUPlace>(Place dst_place,
                                      void* dst,
                                      GPUPlace src_place,
                                      const void* src,
                                      size_t num,
                                      void* stream) {
  Copy(dst_place,
       dst,
       Place(src_place.GetType(), src_place.GetDeviceId()),
       src,
       num,
       stream);
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CUDAPinnedPlace)
template <>
void Copy<GPUPinnedPlace, Place>(GPUPinnedPlace dst_place,
                                 void* dst,
                                 Place src_place,
                                 const void* src,
                                 size_t num,
                                 void* stream) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CUDAPinnedPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace)
template <>
PADDLE_API void Copy<Place, GPUPinnedPlace>(Place dst_place,
                                            void* dst,
                                            GPUPinnedPlace src_place,
                                            const void* src,
                                            size_t num,
                                            void* stream) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CUDAPinnedPlace)
template <>
void Copy<GPUPinnedPlace, Place>(GPUPinnedPlace dst_place,
                                 void* dst,
                                 Place src_place,
                                 const void* src,
                                 size_t num) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, nullptr);
}

// NOTE: only for (CUDAPinnedPlace) -> (CPUPlace)
template <>
void Copy<Place, GPUPinnedPlace>(Place dst_place,
                                 void* dst,
                                 GPUPinnedPlace src_place,
                                 const void* src,
                                 size_t num) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, nullptr);
}
#endif

// NOTE: Synchronous Copy for All Place.
template <>
PADDLE_API void Copy<Place, Place>(
    Place dst_place, void* dst, Place src_place, const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (src_place.GetType() == AllocationType::CPU &&
      dst_place.GetType() == AllocationType::CPU) {  // NOLINT
    std::memcpy(dst, src, num);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (src_place.GetType() == AllocationType::GPU &&  // NOLINT
           dst_place.GetType() == AllocationType::CPU) {
    GPUPlace place_src(src_place.GetDeviceId());
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == AllocationType::CPU &&
             dst_place.GetType() == AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == AllocationType::GPUPINNED &&
             dst_place.GetType() == AllocationType::CPU) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == AllocationType::GPUPINNED &&
             dst_place.GetType() == AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (src_place.GetType() == AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == AllocationType::XPU) {
    XPUPlace place_dst(dst_place.GetDeviceId());
    CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPU &&
             dst_place.GetType() == AllocationType::CPU) {
    XPUPlace place_src(src_place.GetDeviceId());
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPU &&
             dst_place.GetType() == AllocationType::XPU) {
    XPUPlace place_src(src_place.GetDeviceId());
    XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::CPU &&
             dst_place.GetType() == AllocationType::XPUPINNED) {
    CPUPlace place_src;
    XPUPinnedPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPUPINNED &&
             dst_place.GetType() == AllocationType::CPU) {
    CPUPlace place_dst;
    XPUPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPUPINNED &&
             dst_place.GetType() == AllocationType::XPUPINNED) {
    XPUPinnedPlace place_dst;
    XPUPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::XPUPINNED &&
             dst_place.GetType() == AllocationType::XPU) {
    XPUPinnedPlace place_src;
    XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == AllocationType::XPU &&
             dst_place.GetType() == AllocationType::XPUPINNED) {
    XPUPinnedPlace place_dst;
    XPUPlace place_src(src_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  }
#endif
#ifdef PADDLE_WITH_IPU
  else if (src_place.GetType() == AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == AllocationType::IPU) {
    IPUPlace place_dst(dst_place.GetDeviceId());
    CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::IPU &&
             dst_place.GetType() == AllocationType::CPU) {
    IPUPlace place_src(src_place.GetDeviceId());
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == AllocationType::IPU &&
             dst_place.GetType() == AllocationType::IPU) {
    IPUPlace place_src(src_place.GetDeviceId());
    IPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (src_place.GetType() == AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == AllocationType::CUSTOM) {
    CustomPlace place_dst(dst_place.GetDeviceType(), dst_place.GetDeviceId());
    CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == AllocationType::CUSTOM &&
             dst_place.GetType() == AllocationType::CPU) {
    CustomPlace place_src(src_place.GetDeviceType(), src_place.GetDeviceId());
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == AllocationType::CUSTOM &&
             dst_place.GetType() == AllocationType::CUSTOM) {
    CustomPlace place_src(src_place.GetDeviceType(), src_place.GetDeviceId());
    CustomPlace place_dst(dst_place.GetDeviceType(), dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(::common::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
}

// NOTE: Only for (CPUPlace) -> (CPUPlace and PinnedPlace).
template <>
PADDLE_API void Copy<Place, CPUPlace>(Place dst_place,
                                      void* dst,
                                      CPUPlace src_place,
                                      const void* src,
                                      size_t num) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num);
}

// NOTE: Only for (CPUPlace and PinnedPlace) -> (CPUPlace).
template <>
PADDLE_API void Copy<CPUPlace, Place>(CPUPlace dst_place,
                                      void* dst,
                                      Place src_place,
                                      const void* src,
                                      size_t num) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num);
}

#if defined(PADDLE_WITH_CUSTOM_DEVICE) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)

template <>
PADDLE_API void Copy<Place, Place>(Place dst_place,
                                   void* dst,
                                   Place src_place,
                                   const void* src,
                                   size_t num,
                                   void* stream) {
  if (src_place.GetType() == AllocationType::CPU &&  // NOLINT
      dst_place.GetType() == AllocationType::CUSTOM) {
    CPUPlace place_src;
    CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == AllocationType::CPU) {
    CustomPlace place_src(src_place);
    CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == AllocationType::CUSTOM) {
    CustomPlace place_src(src_place);
    CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
  }
}

template <>
PADDLE_API void Copy<CPUPlace, Place>(CPUPlace dst_place,
                                      void* dst,
                                      Place src_place,
                                      const void* src,
                                      size_t num,
                                      void* stream) {
  Copy(Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace).
template <>
PADDLE_API void Copy<Place, CPUPlace>(Place dst_place,
                                      void* dst,
                                      CPUPlace src_place,
                                      const void* src,
                                      size_t num,
                                      void* stream) {
  Copy(dst_place, dst, Place(src_place.GetType()), src, num, stream);
}
#endif

}  // namespace paddle::memory
