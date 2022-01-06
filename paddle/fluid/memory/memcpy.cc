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
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace memory {

template <>
void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace, void* dst,
                                                  platform::CPUPlace,
                                                  const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  VLOG(4) << "src: " << src << ", dst: " << dst << ", num: " << num;
  std::memcpy(dst, src, num);
}
#ifdef PADDLE_WITH_IPU
template <>
void Copy<platform::IPUPlace, platform::CPUPlace>(platform::IPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<platform::CPUPlace, platform::IPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::IPUPlace src_place,
                                                  const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
template <>
void Copy<platform::IPUPlace, platform::IPUPlace>(platform::IPUPlace dst_place,
                                                  void* dst,
                                                  platform::IPUPlace src_place,
                                                  const void* src, size_t num) {
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}
#endif

#ifdef PADDLE_WITH_XPU
template <>
void Copy<platform::XPUPlace, platform::CPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_HOST_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncH2D(dst, src, num, dst_place.device);
}

template <>
void Copy<platform::CPUPlace, platform::XPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_HOST size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2H(dst, src, num, src_place.device);
}

template <>
void Copy<platform::XPUPlace, platform::XPUPlace>(platform::XPUPlace dst_place,
                                                  void* dst,
                                                  platform::XPUPlace src_place,
                                                  const void* src, size_t num) {
  if (num <= 0) {
    VLOG(1) << "memcpy XPU_DEVICE_TO_DEVICE size <= 0 (" << num << ")";
    return;
  }
  platform::MemcpySyncD2D(dst, dst_place.device, src, src_place.device, num);
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
template <>
void Copy<platform::NPUPlace, platform::CPUPlace>(platform::NPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(dst_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:CPU->NPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  } else {
    // On NPU, async operation after sync operation is ok, while sync operation
    // after async is not ok, since the async operation may not done.
    // So, its needed to do wait before sync operation.
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:CPU->NPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE);
  }
}

template <>
void Copy<platform::CPUPlace, platform::NPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::NPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(src_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPU->CPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  } else {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(src_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPU->CPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

template <>
void Copy<platform::NPUPlace, platform::NPUPlace>(platform::NPUPlace dst_place,
                                                  void* dst,
                                                  platform::NPUPlace src_place,
                                                  const void* src, size_t num,
                                                  aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetNPUDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("NpuMemcpyAsync(same_npu):NPU->NPU");
      platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               stream);
    } else {
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

      platform::RecordEvent record_event("NpuMemcpySync(same_npu):NPU->NPU");
      platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  } else {
    if (!platform::NPUCanAccessPeer(dst_place.device, dst_place.device)) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Peer access between NPU places is not allowed."));
    }
    if (stream) {
      // TODO(zhiqiu): support peer access?
      platform::RecordEvent record_event("NpuMemcpyPeerAsync:NPU->NPU");
      platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               stream);
    } else {
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

      platform::RecordEvent record_event("NpuMemcpyPeerSync:NPU->NPU");
      platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::NPUPinnedPlace>(
    platform::CPUPlace dst_place, void* dst, platform::NPUPinnedPlace src_place,
    const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::CPUPlace>(
    platform::NPUPinnedPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::NPUPinnedPlace>(
    platform::NPUPinnedPlace dst_place, void* dst,
    platform::NPUPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::NPUPinnedPlace, platform::NPUPlace>(
    platform::NPUPinnedPlace dst_place, void* dst, platform::NPUPlace src_place,
    const void* src, size_t num, aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(src_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPU->NPUPinned");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  } else {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(src_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPU->NPUPinned");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

template <>
void Copy<platform::NPUPlace, platform::NPUPinnedPlace>(
    platform::NPUPlace dst_place, void* dst, platform::NPUPinnedPlace src_place,
    const void* src, size_t num, aclrtStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetNPUDeviceId(dst_place.device);

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";

  if (stream) {
    platform::RecordEvent record_event("NpuMemcpyAsync:NPUPinned->NPU");
    platform::NPUMemcpyAsync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  } else {
    // On NPU, async operation after sync operation is ok, while sync operation
    // after async is not ok, since the async operation may not done.
    // So, its needed to do wait before sync operation.
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    static_cast<platform::NPUDeviceContext*>(pool.Get(dst_place))->Wait();

    platform::RecordEvent record_event("NpuMemcpySync:NPUPinned->NPU");
    platform::NPUMemcpySync(dst, src, num, ACL_MEMCPY_HOST_TO_DEVICE);
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
void Copy<platform::CPUPlace, platform::CUDAPlace>(
    platform::CPUPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToHost, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CPU");
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
void Copy<platform::CUDAPlace, platform::CPUPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CPU->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyHostToDevice, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CPU->GPU");
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
    platform::CUDAPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by stream(" << stream << ")";
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyAsync(same_gpu):GPU->GPU");
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToDevice, stream);
#else
      platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
#endif
    } else {
      platform::RecordEvent record_event("GpuMemcpySync(same_gpu):GPU->GPU");
#ifdef PADDLE_WITH_HIP
      platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToDevice);
#else
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
#endif
    }
  } else {
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyPeerAsync:GPU->GPU");
      platform::GpuMemcpyPeerAsync(dst, dst_place.device, src, src_place.device,
                                   num, stream);
    } else {
      platform::RecordEvent record_event("GpuMemcpyPeerSync:GPU->GPU");
      platform::GpuMemcpyPeerSync(dst, dst_place.device, src, src_place.device,
                                  num);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::CUDAPinnedPlace>(
    platform::CPUPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CPUPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CPUPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place;
  if (UNLIKELY(num == 0)) return;
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPlace src_place, const void* src, size_t num,
    gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;
  platform::SetDeviceId(src_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CUDAPinned");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyDeviceToHost, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CUDAPinned");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyDeviceToHost);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
#endif
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num,
    gpuStream_t stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetDeviceId(dst_place.device);
  VLOG(4) << "memory::Copy " << num << " Bytes from " << src_place << " to "
          << dst_place << " by thream(" << stream << ")";
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CUDAPinned->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpyAsync(dst, src, num, hipMemcpyHostToDevice, stream);
#else
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
#endif
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CUDAPinned->GPU");
#ifdef PADDLE_WITH_HIP
    platform::GpuMemcpySync(dst, src, num, hipMemcpyHostToDevice);
#else
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
#endif
  }
}

#endif

#ifdef PADDLE_WITH_MLU
template <>
void Copy<platform::CPUPlace, platform::MLUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::MLUPlace src_place,
                                                  const void* src, size_t num,
                                                  mluStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetMLUDeviceId(src_place.device);
  if (stream) {
    VLOG(4) << "Async memory::Copy " << num << " Bytes from " << src_place
            << " to " << dst_place << " by mlu stream(" << stream << ")";
    platform::RecordEvent record_event("MLUMemcpyD2HAsync:MLU->CPU");
    platform::MLUMemcpyD2HAsync(dst, src, num, stream);
  } else {
    VLOG(4) << "Sync memory::Copy " << num << " Bytes from " << src_place
            << " to " << dst_place;
    platform::RecordEvent record_event("MLUMemcpyD2HSync:MLU->CPU");
    platform::MLUMemcpyD2HSync(dst, src, num);
  }
}

template <>
void Copy<platform::MLUPlace, platform::CPUPlace>(platform::MLUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num,
                                                  mluStream stream) {
  if (UNLIKELY(num == 0)) return;

  platform::SetMLUDeviceId(dst_place.device);
  if (stream) {
    VLOG(4) << "Async memory::Copy " << num << " Bytes from " << src_place
            << " to " << dst_place << " by mlu stream(" << stream << ")";
    platform::RecordEvent record_event("MLUMemcpyH2DAsync:CPU->MLU");
    platform::MLUMemcpyH2DAsync(dst, src, num, stream);
  } else {
    VLOG(4) << "Sync memory::Copy " << num << " Bytes from " << src_place
            << " to " << dst_place;
    platform::RecordEvent record_event("MLUMemcpyH2DSync:CPU->MLU");
    platform::MLUMemcpyH2DSync(dst, src, num);
  }
}

template <>
void Copy<platform::MLUPlace, platform::MLUPlace>(platform::MLUPlace dst_place,
                                                  void* dst,
                                                  platform::MLUPlace src_place,
                                                  const void* src, size_t num,
                                                  mluStream stream) {
  if (UNLIKELY(num == 0)) return;

  if (dst_place == src_place) {
    platform::SetMLUDeviceId(dst_place.device);
    if (stream) {
      VLOG(4) << "Async memory::Copy " << num << " Bytes from " << src_place
              << " to " << dst_place << " by mlu stream(" << stream << ")";
      platform::RecordEvent record_event(
          "MLUMemcpyD2DAsync(same_mlu):MLU->MLU");
      platform::MLUMemcpyD2DAsync(dst, src, num, stream);
    } else {
      VLOG(4) << "Sync memory::Copy " << num << " Bytes from " << src_place
              << " to " << dst_place;
      platform::RecordEvent record_event("MLUMemcpyD2DSync(same_mlu):MLU->MLU");
      platform::MLUMemcpyD2DSync(dst, src, num);
    }
  } else {
    if (stream) {
      VLOG(4) << "Async memory::Copy " << num << " Bytes from " << src_place
              << " to " << dst_place << " by mlu stream(" << stream << ")";
      platform::RecordEvent record_event("MLUMemcpyPeerAsync:MLU->MLU");
      platform::MLUMemcpyPeerAsync(dst, dst_place.device, src, src_place.device,
                                   num, stream);
    } else {
      VLOG(4) << "Sync memory::Copy " << num << " Bytes from " << src_place
              << " to " << dst_place;
      platform::RecordEvent record_event("MLUMemcpyPeerSync:MLU->MLU");
      platform::MLUMemcpyPeerSync(dst, dst_place.device, src, src_place.device,
                                  num);
    }
  }
}

#endif  // PADDLE_WITH_MLU

}  // namespace memory
}  // namespace paddle
