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

#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/test_macros.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#endif

namespace paddle {
namespace memory {

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <>
void Copy<platform::CPUPlace, platform::CustomPlace>(
    platform::CPUPlace dst_place,
    void* dst,
    platform::CustomPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  auto src_type = platform::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = platform::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  platform::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  phi::DeviceManager::SetDevice(src_place);
  phi::stream::Stream stream_wrapper(src_place, stream);
  phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyD2H(
      dst, src, num, &stream_wrapper);
}

template <>
void Copy<platform::CustomPlace, platform::CPUPlace>(
    platform::CustomPlace dst_place,
    void* dst,
    platform::CPUPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;
  auto src_type = platform::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = platform::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  platform::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  phi::DeviceManager::SetDevice(dst_place);
  phi::stream::Stream stream_wrapper(dst_place, stream);
  phi::DeviceManager::GetDeviceWithPlace(dst_place)->MemoryCopyH2D(
      dst, src, num, &stream_wrapper);
}

template <>
void Copy<platform::CustomPlace, platform::CustomPlace>(
    platform::CustomPlace dst_place,
    void* dst,
    platform::CustomPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  auto src_type = platform::PlaceHelper::GetDeviceType(src_place);
  auto dst_type = platform::PlaceHelper::GetDeviceType(dst_place);
  std::string msg = "Memcpy:" + src_type + "->" + dst_type;
  platform::RecordEvent record_event(msg);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << ", stream=" << stream;

  if (src_type == dst_type) {
    phi::DeviceManager::SetDevice(src_place);
    phi::stream::Stream stream_wrapper(src_place, stream);

    auto src_id = platform::PlaceHelper::GetDeviceId(src_place);
    auto dst_id = platform::PlaceHelper::GetDeviceId(dst_place);
    if (src_id == dst_id) {
      phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyD2D(
          dst, src, num, &stream_wrapper);
    } else {
      phi::DeviceManager::GetDeviceWithPlace(src_place)->MemoryCopyP2P(
          dst_place, dst, src, num, &stream_wrapper);
    }
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Copy between %s and %s is not supported.", src_type, dst_type));
  }
}
#endif  // PADDLE_WITH_CUSTOM_DEVICE

template <>
TEST_API void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace,
                                                           void* dst,
                                                           platform::CPUPlace,
                                                           const void* src,
                                                           size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "src: " << src << ", dst: " << dst << ", num: " << num;
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_IPU
template <>
void Copy<platform::IPUPlace, platform::CPUPlace>(platform::IPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src,
                                                  size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<platform::CPUPlace, platform::IPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::IPUPlace src_place,
                                                  const void* src,
                                                  size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<platform::IPUPlace, platform::IPUPlace>(platform::IPUPlace dst_place,
                                                  void* dst,
                                                  platform::IPUPlace src_place,
                                                  const void* src,
                                                  size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

// NOTE: only for (CPUPlace and IPUPlace) -> (IPUPlace).
template <>
void Copy<phi::IPUPlace, phi::Place>(phi::IPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num) {
  if (src_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_src;
    return Copy(dst_place, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::IPU) {
    platform::IPUPlace place_src(src_place.GetDeviceId());
    return Copy(dst_place, dst, place_src, src, num);
  }
}

// NOTE: only for (IPUPlace) -> (CPUPlace and IPUPlace).
template <>
void Copy<phi::Place, phi::IPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::IPUPlace src_place,
                                     const void* src,
                                     size_t num) {
  if (dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, src_place, src, num);
  } else if (dst_place.GetType() == phi::AllocationType::IPU) {
    platform::IPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, src_place, src, num);
  }
}
#endif

#ifdef PADDLE_WITH_XPU
template <>
void Copy<platform::XPUPlace, platform::CPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src,
                                                  size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_HOST_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncH2D(dst, src, num, dst_place);
}

template <>
void Copy<platform::CPUPlace, platform::XPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
                                                  const void* src,
                                                  size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_HOST size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2H(dst, src, num, src_place);
}

template <>
void Copy<platform::XPUPlace, platform::XPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
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
void Copy<phi::XPUPlace, phi::Place>(phi::XPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num) {
  if (src_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_src;
    return Copy(dst_place, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    return Copy(dst_place, dst, place_src, src, num);
  }
}

// NOTE: only for (XPUPlace) -> (CPUPlace and XPUPlace).
template <>
void Copy<phi::Place, phi::XPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::XPUPlace src_place,
                                     const void* src,
                                     size_t num) {
  if (dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, src_place, src, num);
  } else if (dst_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, src_place, src, num);
  }
}

#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
  cudaStreamSynchronize(0);
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
TEST_API void Copy<platform::CPUPlace, platform::CUDAPlace>(
    platform::CPUPlace dst_place,
    void* dst,
    platform::CUDAPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event(
        "GpuMemcpyAsync:GPU->CPU", platform::TracerEventType::UserDefined, 1);
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
    platform::RecordEvent record_event(
        "GpuMemcpySync:GPU->CPU", platform::TracerEventType::UserDefined, 1);
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
TEST_API void Copy<platform::CUDAPlace, platform::CPUPlace>(
    platform::CUDAPlace dst_place,
    void* dst,
    platform::CPUPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event(
        "GpuMemcpyAsync:CPU->GPU", platform::TracerEventType::UserDefined, 1);
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
    platform::RecordEvent record_event(
        "GpuMemcpySync:CPU->GPU", platform::TracerEventType::UserDefined, 1);
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
void Copy<platform::CUDAPlace, platform::CUDAPlace>(
    platform::CUDAPlace dst_place,
    void* dst,
    platform::CUDAPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyAsync(same_gpu):GPU->GPU",
                                         platform::TracerEventType::UserDefined,
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
      platform::RecordEvent record_event("GpuMemcpySync(same_gpu):GPU->GPU",
                                         platform::TracerEventType::UserDefined,
                                         1);
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToDevice);
#else
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
#endif
    }
  } else {
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyPeerAsync:GPU->GPU",
                                         platform::TracerEventType::UserDefined,
                                         1);
      platform::GpuMemcpyPeerAsync(dst,
                                   dst_place.device,
                                   src,
                                   src_place.device,
                                   num,
                                   reinterpret_cast<gpuStream_t>(stream));
    } else {
      platform::RecordEvent record_event("GpuMemcpyPeerSync:GPU->GPU",
                                         platform::TracerEventType::UserDefined,
                                         1);
      platform::GpuMemcpyPeerSync(
          dst, dst_place.device, src, src_place.device, num);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::CUDAPinnedPlace>(
    platform::CPUPlace dst_place,
    void* dst,
    platform::CUDAPinnedPlace src_place,
    const void* src,
    size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
TEST_API void Copy<platform::CUDAPinnedPlace, platform::CPUPlace>(
    platform::CUDAPinnedPlace dst_place,
    void* dst,
    platform::CPUPlace src_place,
    const void* src,
    size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPinnedPlace dst_place,
    void* dst,
    platform::CUDAPinnedPlace src_place,
    const void* src,
    size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPlace>(
    platform::CUDAPinnedPlace dst_place,
    void* dst,
    platform::CUDAPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;
  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CUDAPinned",
                                       platform::TracerEventType::UserDefined,
                                       1);
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
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CUDAPinned",
                                       platform::TracerEventType::UserDefined,
                                       1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPlace dst_place,
    void* dst,
    platform::CUDAPinnedPlace src_place,
    const void* src,
    size_t num,
    void* stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CUDAPinned->GPU",
                                       platform::TracerEventType::UserDefined,
                                       1);
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
    platform::RecordEvent record_event("GpuMemcpySync:CUDAPinned->GPU",
                                       platform::TracerEventType::UserDefined,
                                       1);
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
  }
}

// NOTE: only for CPUPlace、CUDAPlace and CUDAPinnedPlace.
template <>
void Copy<phi::Place, phi::Place>(phi::Place dst_place,
                                  void* dst,
                                  phi::Place src_place,
                                  const void* src,
                                  size_t num,
                                  void* stream) {
  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_dst, place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place.GetType() == phi::AllocationType::GPU) {
    platform::CUDAPlace place_dst(dst_place.GetDeviceId());
    platform::CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::GPU &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CUDAPlace place_src(src_place.GetDeviceId());
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::GPU &&
             dst_place.GetType() == phi::AllocationType::GPU) {
    platform::CUDAPlace place_src(src_place.GetDeviceId());
    platform::CUDAPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    platform::CPUPlace place_src;
    platform::CUDAPinnedPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_dst;
    platform::CUDAPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED &&
             dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    platform::CUDAPinnedPlace place_dst;
    platform::CUDAPinnedPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED &&
             dst_place.GetType() == phi::AllocationType::GPU) {
    platform::CUDAPinnedPlace place_src;
    platform::CUDAPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::GPU &&
             dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    platform::CUDAPinnedPlace place_dst;
    platform::CUDAPlace place_src(src_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, stream);
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
             dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CPUPlace place_src;
    platform::CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CustomPlace place_src(src_place);
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CustomPlace place_src(src_place);
    platform::CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
#endif
  }
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CPUPlace).
template <>
void Copy<phi::CPUPlace, phi::Place>(phi::CPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(phi::Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace).
template <>
void Copy<phi::Place, phi::CPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::CPUPlace src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(dst_place, dst, phi::Place(src_place.GetType()), src, num, stream);
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CUDAPlace)
template <>
void Copy<phi::GPUPlace, phi::Place>(phi::GPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(phi::Place(dst_place.GetType(), dst_place.GetDeviceId()),
       dst,
       src_place,
       src,
       num,
       stream);
}

// NOTE: only for (CUDAPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace)
template <>
void Copy<phi::Place, phi::GPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::GPUPlace src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(dst_place,
       dst,
       phi::Place(src_place.GetType(), src_place.GetDeviceId()),
       src,
       num,
       stream);
}

// NOTE: only for (CPUPlace, CUDAPlace and CUDAPinnedPlace) -> (CUDAPinnedPlace)
template <>
void Copy<phi::GPUPinnedPlace, phi::Place>(phi::GPUPinnedPlace dst_place,
                                           void* dst,
                                           phi::Place src_place,
                                           const void* src,
                                           size_t num,
                                           void* stream) {
  Copy(phi::Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CUDAPinnedPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace)
template <>
void Copy<phi::Place, phi::GPUPinnedPlace>(phi::Place dst_place,
                                           void* dst,
                                           phi::GPUPinnedPlace src_place,
                                           const void* src,
                                           size_t num,
                                           void* stream) {
  Copy(dst_place, dst, phi::Place(src_place.GetType()), src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CUDAPinnedPlace)
template <>
void Copy<phi::GPUPinnedPlace, phi::Place>(phi::GPUPinnedPlace dst_place,
                                           void* dst,
                                           phi::Place src_place,
                                           const void* src,
                                           size_t num) {
  Copy(phi::Place(dst_place.GetType()), dst, src_place, src, num, nullptr);
}

// NOTE: only for (CUDAPinnedPlace) -> (CPUPlace)
template <>
void Copy<phi::Place, phi::GPUPinnedPlace>(phi::Place dst_place,
                                           void* dst,
                                           phi::GPUPinnedPlace src_place,
                                           const void* src,
                                           size_t num) {
  Copy(dst_place, dst, phi::Place(src_place.GetType()), src, num, nullptr);
}
#endif

// NOTE: Only for CPUPlace, XPUPlace and PinnedPlace.
template <>
void Copy<phi::Place, phi::Place>(phi::Place dst_place,
                                  void* dst,
                                  phi::Place src_place,
                                  const void* src,
                                  size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place.GetType() == phi::AllocationType::CPU) {  // NOLINT
    std::memcpy(dst, src, num);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst, src, num);
  } else if (src_place.GetType() == phi::AllocationType::GPUPINNED &&
             dst_place.GetType() == phi::AllocationType::GPUPINNED) {
    std::memcpy(dst, src, num);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CPUPlace place_dst, place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    platform::CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::XPU &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::XPU &&
             dst_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num);
  }
#endif
#ifdef PADDLE_WITH_IPU
  else if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == phi::AllocationType::IPU) {
    platform::IPUPlace place_dst(dst_place.GetDeviceId());
    platform::CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::IPU &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::IPUPlace place_src(src_place.GetDeviceId());
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num);
  } else if (src_place.GetType() == phi::AllocationType::IPU &&
             dst_place.GetType() == phi::AllocationType::IPU) {
    platform::IPUPlace place_src(src_place.GetDeviceId());
    platform::IPUPlace place_dst(dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
           dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CustomPlace place_dst(dst_place.GetDeviceType(),
                                    dst_place.GetDeviceId());
    platform::CPUPlace place_src;
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CustomPlace place_src(src_place.GetDeviceType(),
                                    src_place.GetDeviceId());
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CustomPlace place_src(src_place.GetDeviceType(),
                                    src_place.GetDeviceId());
    platform::CustomPlace place_dst(dst_place.GetDeviceType(),
                                    dst_place.GetDeviceId());
    return Copy(place_dst, dst, place_src, src, num, nullptr);
  }
#endif
}

// NOTE: Only for (CPUPlace) -> (CPUPlace and PinnedPlace).
template <>
void Copy<phi::Place, phi::CPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::CPUPlace src_place,
                                     const void* src,
                                     size_t num) {
  Copy(dst_place, dst, phi::Place(src_place.GetType()), src, num);
}

// NOTE: Only for (CPUPlace and PinnedPlace) -> (CPUPlace).
template <>
void Copy<phi::CPUPlace, phi::Place>(phi::CPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num) {
  Copy(phi::Place(dst_place.GetType()), dst, src_place, src, num);
}

#if defined(PADDLE_WITH_CUSTOM_DEVICE) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)

template <>
void Copy<phi::Place, phi::Place>(phi::Place dst_place,
                                  void* dst,
                                  phi::Place src_place,
                                  const void* src,
                                  size_t num,
                                  void* stream) {
  if (src_place.GetType() == phi::AllocationType::CPU &&  // NOLINT
      dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CPUPlace place_src;
    platform::CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == phi::AllocationType::CPU) {
    platform::CustomPlace place_src(src_place);
    platform::CPUPlace place_dst;
    return Copy(place_dst, dst, place_src, src, num, stream);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == phi::AllocationType::CUSTOM) {
    platform::CustomPlace place_src(src_place);
    platform::CustomPlace place_dst(dst_place);
    return Copy(place_dst, dst, place_src, src, num, stream);
  }
}

template <>
void Copy<phi::CPUPlace, phi::Place>(phi::CPUPlace dst_place,
                                     void* dst,
                                     phi::Place src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(phi::Place(dst_place.GetType()), dst, src_place, src, num, stream);
}

// NOTE: only for (CPUPlace) -> (CPUPlace, CUDAPlace and CUDAPinnedPlace).
template <>
void Copy<phi::Place, phi::CPUPlace>(phi::Place dst_place,
                                     void* dst,
                                     phi::CPUPlace src_place,
                                     const void* src,
                                     size_t num,
                                     void* stream) {
  Copy(dst_place, dst, phi::Place(src_place.GetType()), src, num, stream);
}
#endif

}  // namespace memory
}  // namespace paddle
