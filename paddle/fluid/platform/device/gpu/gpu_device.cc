// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"

#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace platform {

#if CUDA_VERSION >= 10000
static void CUDART_CB StreamCallbackFunc(void* user_data)
#else
static void CUDART_CB StreamCallbackFunc(cudaStream_t stream,
                                         cudaError_t status, void* user_data)
#endif
{
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()>*>(user_data));
  (*func)();
}

class GpuDevice : public DeviceInterface {
 public:
  GpuDevice(const std::string& type, int priority, bool is_custom)
      : DeviceInterface(type, priority, is_custom) {}
  ~GpuDevice() override {}
  size_t GetDeviceCount() override { return GetGPUDeviceCount(); }
};

class CudaDevice : public GpuDevice {
 public:
  CudaDevice(const std::string& type, int priority, bool is_custom)
      : GpuDevice(type, priority, is_custom) {}
  ~CudaDevice() override {}

  std::vector<size_t> GetDeviceList() override {
    std::vector<size_t> devices;
    return devices;
  }

  void SynchronizeDevice(size_t dev_id) override {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(BOOST_GET_CONST(CUDAPlace,
    // place).GetDeviceId()));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }

  void Initialize() override {
    size_t count = GetDeviceCount();
    for (size_t i = 0; i < count; ++i) {
      InitDevice(i);
    }
  }

  void Finalize() override {
    size_t count = GetDeviceCount();
    for (size_t i = 0; i < count; ++i) {
      DeInitDevice(i);
    }
  }

  void InitDevice(size_t dev_id) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(dev_id));
  }

  void DeInitDevice(size_t dev_id) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(dev_id));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceReset());
  }

  void SetDevice(size_t dev_id) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(dev_id));
  }

  int GetDevice() override {
    int device;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&device));
    return device;
  }

  void CreateStream(size_t dev_id, stream::Stream* stream,
                    const stream::Stream::Priority& priority =
                        stream::Stream::Priority::kNormal,
                    const stream::Stream::Flag& flag =
                        stream::Stream::Flag::kDefaultFlag) override {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(BOOST_GET_CONST(CUDAPlace,
    // place).GetDeviceId()));
    cudaStream_t cuda_stream;
    if (priority == stream::Stream::Priority::kHigh) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(
          &cuda_stream, static_cast<unsigned int>(flag), -1));
    } else if (priority == stream::Stream::Priority::kNormal) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(
          &cuda_stream, static_cast<unsigned int>(flag), 0));
    }
    stream->set_stream(cuda_stream);
  }

  void DestroyStream(size_t dev_id, stream::Stream* stream) override {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(BOOST_GET_CONST(CUDAPlace,
    // place).GetDeviceId()));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(
        reinterpret_cast<cudaStream_t>(stream->raw_stream())));
  }

  void SynchronizeStream(size_t dev_id, const stream::Stream* stream) override {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(BOOST_GET_CONST(CUDAPlace,
    // place).GetDeviceId()));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream->raw_stream())));
  }

  bool QueryStream(size_t dev_id, const stream::Stream* stream) override {
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(BOOST_GET_CONST(CUDAPlace,
    // place).GetDeviceId()));
    cudaError_t err =
        cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream->raw_stream()));
    if (err == cudaSuccess) {
      return true;
    }

    if (err == cudaErrorNotReady) {
      return false;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(err);
    return false;
  }

  void AddCallback(size_t dev_id, stream::Stream* stream,
                   stream::Stream::Callback* callback) override {
#if CUDA_VERSION >= 10000
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream->raw_stream()),
                           StreamCallbackFunc, callback));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamAddCallback(
        reinterpret_cast<cudaStream_t>(stream->raw_stream()),
        StreamCallbackFunc, callback, 0));
#endif
  }

  void CreateEvent(size_t dev_id, event::Event* event,
                   event::Event::Flag flags) override {
    cudaEvent_t cuda_event;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(
        &cuda_event, static_cast<unsigned int>(flags)));
    event->set_event(cuda_event);
  }

  void DestroyEvent(size_t dev_id, event::Event* event) override {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event->raw_event())));
  }

  void RecordEvent(size_t dev_id, const event::Event* event,
                   const stream::Stream* stream) override {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(event->raw_event()),
                        reinterpret_cast<cudaStream_t>(stream->raw_stream())));
  }

  void SynchronizeEvent(size_t dev_id, const event::Event* event) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(
        reinterpret_cast<cudaEvent_t>(event->raw_event())));
  }

  bool QueryEvent(size_t dev_id, const event::Event* event) override {
    return cudaEventQuery(reinterpret_cast<cudaEvent_t>(event->raw_event()));
  }

  void StreamWaitEvent(size_t dev_id, const stream::Stream* stream,
                       const event::Event* event) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(
        reinterpret_cast<cudaStream_t>(stream->raw_stream()),
        reinterpret_cast<cudaEvent_t>(event->raw_event()), 0));
  }

  void MemoryCopyH2D(size_t dev_id, void* dst, const void* src, size_t size,
                     const stream::Stream* stream = nullptr) override {
    if (stream && stream->raw_stream()) {
      platform::GpuMemcpyAsync(
          dst, src, size, cudaMemcpyHostToDevice,
          reinterpret_cast<cudaStream_t>(stream->raw_stream()));
    } else {
      platform::GpuMemcpySync(dst, src, size, cudaMemcpyHostToDevice);
    }
  }

  void MemoryCopyD2H(size_t dev_id, void* dst, const void* src, size_t size,
                     const stream::Stream* stream = nullptr) override {
    if (stream && stream->raw_stream()) {
      platform::GpuMemcpyAsync(
          dst, src, size, cudaMemcpyDeviceToHost,
          reinterpret_cast<cudaStream_t>(stream->raw_stream()));
    } else {
      platform::GpuMemcpySync(dst, src, size, cudaMemcpyDeviceToHost);
    }
  }

  void MemoryCopyD2D(size_t dev_id, void* dst, const void* src, size_t size,
                     const stream::Stream* stream = nullptr) override {
    if (stream && stream->raw_stream()) {
      platform::GpuMemcpyAsync(
          dst, src, size, cudaMemcpyDeviceToDevice,
          reinterpret_cast<cudaStream_t>(stream->raw_stream()));
    } else {
      platform::GpuMemcpySync(dst, src, size, cudaMemcpyDeviceToDevice);
    }
  }

  void MemoryCopyP2P(const Place& dst_place, void* dst, size_t src_dev_id,
                     const void* src, size_t size,
                     const stream::Stream* stream = nullptr) override {
    if (stream && stream->raw_stream()) {
      platform::GpuMemcpyPeerAsync(
          dst, BOOST_GET_CONST(CUDAPlace, dst_place).device, src, src_dev_id,
          size, reinterpret_cast<cudaStream_t>(stream->raw_stream()));
    } else {
      platform::GpuMemcpyPeerSync(dst,
                                  BOOST_GET_CONST(CUDAPlace, dst_place).device,
                                  src, src_dev_id, size);
    }
  }

  void* MemoryAllocate(size_t dev_id, size_t size) override {
    void* ptr;

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
    return ptr;
  }

  void MemoryDeallocate(size_t dev_id, void* ptr, size_t size) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(&ptr));
  }

  void* MemoryAllocateHost(size_t dev_id, size_t size) override {
    void* ptr;

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
    return ptr;
  }

  void MemoryDeallocateHost(size_t dev_id, void* ptr, size_t size) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(&ptr));
  }

  void* MemoryAllocateUnified(size_t dev_id, size_t size) override {
    void* ptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return ptr;
  }

  void MemoryDeallocateUnified(size_t dev_id, void* ptr, size_t size) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(&ptr));
  }

  void MemorySet(size_t dev_id, void* ptr, uint8_t value,
                 size_t size) override {
    cudaMemset(ptr, value, size);
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(free, total));
    size_t used = *total - *free;
    VLOG(10) << Type() + " memory usage " << (used >> 20) << "M/"
             << (*total >> 20) << "M, " << (*free >> 20)
             << "M available to allocate";
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    // Allow to allocate the minimum chunk size is 256 bytes.
    constexpr size_t min_chunk_size = 1 << 8;
    VLOG(10) << Type() + " min chunk size " << min_chunk_size;
    return min_chunk_size;
  }
};

}  // namespace platform
}  // namespace paddle

REGISTER_BUILTIN_DEVICE(gpu, paddle::platform::CudaDevice);
