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

#include "paddle/fluid/platform/device/device_manager.h"
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/device/npu/npu_info.h"

namespace paddle {
namespace platform {

static void StreamCallbackFunc(void* user_data) {
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()>*>(user_data));
  (*func)();
}

class NpuDevice : public DeviceInterface {
 public:
  NpuDevice(const std::string& type, int priority, bool is_pluggable)
      : DeviceInterface(type, priority, is_pluggable) {
    // Initialize();
  }

  ~NpuDevice() override {
    // Finalize();
  }

  size_t VisibleDevicesCount() override {
    uint32_t count = 0;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetDeviceCount(&count));
    return count;
  }

  void SynchronizeDevice(size_t dev_id) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeDevice());
  }

  void Initialize() override {
    // TODO(wangran16): make sure to initialize once
    PADDLE_ENFORCE_NPU_SUCCESS(aclInit(nullptr));
    size_t count = VisibleDevicesCount();
    for (size_t i = 0; i < count; ++i) {
      InitDevice(i);
    }
  }

  void Finalize() override {
    size_t count = VisibleDevicesCount();
    for (size_t i = 0; i < count; ++i) {
      DeInitDevice(i);
    }
    // TODO(wangran16): make sure to finalize once
    PADDLE_ENFORCE_NPU_SUCCESS(aclFinalize());
  }

  void InitDevice(size_t dev_id) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
  }

  void DeInitDevice(size_t dev_id) override {
    // TODO(wangran16): guarantee aclrtDestroyEvent/aclrtDestroyStream -->
    // aclrtDestroyContext --> aclrtResetDevice
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtResetDevice(dev_id));
  }

  void SetDevice(size_t dev_id) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
  }

  int GetDevice() override {
    int device = 0;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetDevice(&device));
    return device;
  }

  void CreateStream(size_t dev_id, stream::Stream* stream,
                    const stream::Stream::Priority& priority =
                        stream::Stream::Priority::kNormal,
                    const stream::Stream::Flag& flag =
                        stream::Stream::Flag::kDefaultFlag) override {
    if (priority != stream::Stream::Priority::kNormal ||
        flag != stream::Stream::Flag::kDefaultFlag) {
      PADDLE_THROW(platform::errors::Unavailable(
          "priority != stream::Stream::Priority::kNormal || flag != "
          "stream::Stream::Flag::kDefaultFlag is not allowed on "
          "NpuDevice."));
    }
    aclrtStream acl_stream;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateStream(&acl_stream));
    stream->set_stream(acl_stream);
  }

  void DestroyStream(size_t dev_id, stream::Stream* stream) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyStream(
        reinterpret_cast<aclrtStream>(stream->raw_stream())));
  }

  void SynchronizeStream(size_t dev_id, const stream::Stream* stream) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(stream->raw_stream())));
  }

  bool QueryStream(size_t dev_id, const stream::Stream* stream) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(stream->raw_stream())));
    return true;
  }

  void AddCallback(size_t dev_id, stream::Stream* stream,
                   stream::Stream::Callback* callback) override {
    VLOG(3) << "aclrtLaunchCallback at stream: " << stream->raw_stream();
    // TODO(zhiqiu): failed to call aclrtLaunchCallback
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtLaunchCallback(
        StreamCallbackFunc, callback, ACL_CALLBACK_BLOCK,
        reinterpret_cast<aclrtStream>(stream->raw_stream())));
  }

  void CreateEvent(size_t dev_id, event::Event* event,
                   event::Event::Flag flags) override {
    aclrtEvent acl_event;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateEvent(&acl_event));
    event->set_event(acl_event);
  }

  void DestroyEvent(size_t dev_id, event::Event* event) override {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event->raw_event())));
  }

  void RecordEvent(size_t dev_id, const event::Event* event,
                   const stream::Stream* stream) override {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclrtRecordEvent(reinterpret_cast<aclrtEvent>(event->raw_event()),
                         reinterpret_cast<aclrtStream>(stream->raw_stream())));
  }

  void SynchronizeEvent(size_t dev_id, const event::Event* event) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeEvent(
        reinterpret_cast<aclrtEvent>(event->raw_event())));
  }

  bool QueryEvent(size_t dev_id, const event::Event* event) override {
    aclrtEventStatus status;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtQueryEvent(
        reinterpret_cast<aclrtEvent>(event->raw_event()), &status));
    return status == ACL_EVENT_STATUS_COMPLETE;
  }

  void StreamWaitEvent(size_t dev_id, const stream::Stream* stream,
                       const event::Event* event) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtStreamWaitEvent(
        reinterpret_cast<aclrtStream>(stream->raw_stream()),
        reinterpret_cast<aclrtEvent>(event->raw_event())));
  }

  void MemoryCopy(size_t dev_id, void* dst, const void* src, size_t size,
                  MemoryCpyKind kind,
                  const stream::Stream* stream = nullptr) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
    if (stream && stream->raw_stream()) {
      if (kind == MemoryCpyKind::HostToDevice) {
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpyAsync(
            dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE,
            reinterpret_cast<aclrtStream>(stream->raw_stream())));
      } else if (kind == MemoryCpyKind::DeviceToHost) {
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpyAsync(
            dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST,
            reinterpret_cast<aclrtStream>(stream->raw_stream())));
      } else if (kind == MemoryCpyKind::DeviceToDevice) {
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpyAsync(
            dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE,
            reinterpret_cast<aclrtStream>(stream->raw_stream())));
      } else {
        PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryCpyKind."));
      }
    } else {
      // On NPU, async operation after sync operation is ok, while sync
      // operation
      // after async is not ok, since the async operation may not done.
      // So, its needed to do wait before sync operation.
      auto place = platform::NPUPlace(dev_id);
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      pool.Get(place)->Wait();

      if (kind == MemoryCpyKind::HostToDevice) {
        PADDLE_ENFORCE_NPU_SUCCESS(
            aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
      } else if (kind == MemoryCpyKind::DeviceToHost) {
        PADDLE_ENFORCE_NPU_SUCCESS(
            aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
      } else if (kind == MemoryCpyKind::DeviceToDevice) {
        PADDLE_ENFORCE_NPU_SUCCESS(
            aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
      } else {
        PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryCpyKind."));
      }
    }
  }

  void MemoryCopyPeer(const Place& dst_place, void* dst, size_t src_dev_id,
                      const void* src, size_t size,
                      const stream::Stream* stream = nullptr) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(src_dev_id));
    auto dst_dev_id = BOOST_GET_CONST(NPUPlace, dst_place).GetDeviceId();
    bool can_access_peer = NPUCanAccessPeer(src_dev_id, dst_dev_id);
    if (stream && stream->raw_stream()) {
      if (!can_access_peer) {
        PADDLE_THROW(platform::errors::Unavailable(
            "Peer access between NpuDevice places is not allowed."));
      } else {
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtDeviceEnablePeerAccess(dst_dev_id, 0));
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpyAsync(
            dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE,
            reinterpret_cast<aclrtStream>(stream->raw_stream())));
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtDeviceDisablePeerAccess(dst_dev_id));
      }
    } else {
      if (!can_access_peer) {
        PADDLE_THROW(platform::errors::Unavailable(
            "Peer access between NpuDevice places is not allowed."));
      } else {
        auto src_place = platform::NPUPlace(src_dev_id);
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        pool.Get(src_place)->Wait();

        PADDLE_ENFORCE_NPU_SUCCESS(aclrtDeviceEnablePeerAccess(dst_dev_id, 0));
        PADDLE_ENFORCE_NPU_SUCCESS(
            aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
        PADDLE_ENFORCE_NPU_SUCCESS(aclrtDeviceDisablePeerAccess(dst_dev_id));
      }
    }
  }

  void* MemoryAllocate(
      size_t dev_id, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    void* ptr = nullptr;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
    if (kind == MemoryAllocKind::Normal) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    } else if (kind == MemoryAllocKind::Host) {
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtMallocHost(&ptr, size));
    } else if (kind == MemoryAllocKind::Unified) {
      PADDLE_THROW(platform::errors::Unavailable(
          "MemoryAllocKind::Unified on NpuDevice places is not allowed."));
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryAllocKind."));
    }
    return ptr;
  }

  void MemoryDeallocate(
      size_t dev_id, void* ptr, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    if (kind == MemoryAllocKind::Normal) {
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(&ptr));
    } else if (kind == MemoryAllocKind::Host) {
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtFreeHost(&ptr));
    } else if (kind == MemoryAllocKind::Unified) {
      PADDLE_THROW(platform::errors::Unavailable(
          "MemoryAllocKind::Unified on NpuDevice places is not allowed."));
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryAllocKind."));
    }
  }

  void MemorySet(size_t dev_id, void* ptr, uint8_t value,
                 size_t size) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemset(ptr, size, value, size));
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
    size_t used = *total - *free;
    VLOG(10) << Type() + " memory usage " << (used >> 20) << "M/"
             << (*total >> 20) << "M, " << (*free >> 20)
             << "M available to allocate";
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    // NOTE(zhiqiu): It seems the min chunk size should be 512 on NPU,
    // though no document specify that explicitly.
    // See https://gitee.com/zhiqiuchen/Ascend/tree/master/test_reduce_sum_d for
    // details.
    constexpr size_t min_chunk_size = 1 << 9;
    VLOG(10) << Type() + " min chunk size " << min_chunk_size;
    return min_chunk_size;
  }

  size_t GetExtraPaddingSize(size_t dev_id) override {
    constexpr size_t extra_padding_size = 32;
    VLOG(10) << Type() + " extra padding size " << extra_padding_size;
    return extra_padding_size;
  }
};

}  // namespace platform
}  // namespace paddle

REGISTER_BUILTIN_DEVICE(npu, paddle::platform::NpuDevice);
