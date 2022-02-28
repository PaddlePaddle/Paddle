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

#pragma once
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"

namespace paddle {
namespace platform {

class DeviceInterface {  // Driver / Runtime
 public:
  DeviceInterface(const std::string& type, uint8_t priority, bool is_custom)
      : type_(type), priority_(priority), is_custom_(is_custom) {}
  uint8_t Priority() { return priority_; }
  std::string Type() { return type_; }
  bool IsCustom() { return is_custom_; }

  virtual ~DeviceInterface() {}

  // Info
  virtual size_t GetComputeCapability();

  virtual size_t GetRuntimeVersion();

  virtual size_t GetDriverVersion();

  // Platform
  //! Initialize
  virtual void Initialize();

  //! Finalize
  virtual void Finalize();

  // Device
  virtual size_t GetDeviceCount() = 0;
  virtual std::vector<size_t> GetDeviceList() = 0;

  //! Wait for compute device to finish.
  virtual void SynchronizeDevice(size_t dev_id);

  //! Initialize device.
  virtual void InitDevice(size_t dev_id);

  //! Deinitialize device.
  virtual void DeInitDevice(size_t dev_id);

  // ! Set device to be used.
  virtual void SetDevice(size_t dev_id);

  // ! Returns which device is currently being used.
  virtual int GetDevice();

  // Stream
  // ! Create an asynchronous stream
  virtual void CreateStream(
      size_t dev_id, stream::Stream* stream,
      const stream::Stream::Priority& priority =
          stream::Stream::Priority::kNormal,
      const stream::Stream::Flag& flag = stream::Stream::Flag::kDefaultFlag);

  // ! Destroys an asynchronous stream.
  virtual void DestroyStream(size_t dev_id, stream::Stream* stream);

  // ! Waits for stream tasks to complete.
  virtual void SynchronizeStream(size_t dev_id, const stream::Stream* stream);

  // ! Queries an asynchronous stream for completion status.
  virtual bool QueryStream(size_t dev_id, const stream::Stream* stream);

  // ! Add a callback to a compute stream.
  virtual void AddCallback(size_t dev_id, stream::Stream* stream,
                           stream::Stream::Callback* callback);

  // Event
  // ! Create an event.
  virtual void CreateEvent(size_t dev_id, event::Event* event,
                           event::Event::Flag flags);

  // ! Destroy an event.
  virtual void DestroyEvent(size_t dev_id, event::Event* event);

  // ! Records an event.
  virtual void RecordEvent(size_t dev_id, const event::Event* event,
                           const stream::Stream* stream);

  // ! Waits for event to complete.
  virtual void SynchronizeEvent(size_t dev_id, const event::Event* event);
  // ! Queries an event for completion status.
  virtual bool QueryEvent(size_t dev_id, const event::Event* event);

  // ! Make a compute stream wait on an event
  virtual void StreamWaitEvent(size_t dev_id, const stream::Stream* stream,
                               const event::Event* event);

  // Memory
  virtual void MemoryCopyH2D(size_t dev_id, void* dst, const void* src,
                             size_t size,
                             const stream::Stream* stream = nullptr);

  virtual void MemoryCopyD2H(size_t dev_id, void* dst, const void* src,
                             size_t size,
                             const stream::Stream* stream = nullptr);

  virtual void MemoryCopyD2D(size_t dev_id, void* dst, const void* src,
                             size_t size,
                             const stream::Stream* stream = nullptr);

  virtual void MemoryCopyP2P(const Place& dst_place, void* dst, size_t src_id,
                             const void* src, size_t size,
                             const stream::Stream* stream = nullptr);

  virtual void* MemoryAllocate(size_t dev_id, size_t size);

  virtual void MemoryDeallocate(size_t dev_id, void* ptr, size_t size);

  virtual void* MemoryAllocateHost(size_t dev_id, size_t size);

  virtual void MemoryDeallocateHost(size_t dev_id, void* ptr, size_t size);

  virtual void* MemoryAllocateUnified(size_t dev_id, size_t size);

  virtual void MemoryDeallocateUnified(size_t dev_id, void* ptr, size_t size);

  virtual void MemorySet(size_t dev_id, void* ptr, uint8_t value, size_t size);

  virtual void MemoryStats(size_t dev_id, size_t* total, size_t* free);

  virtual size_t GetMinChunkSize(size_t dev_id);

  virtual size_t GetInitAllocSize(size_t dev_id);

  virtual size_t GetReallocSize(size_t dev_id);

  virtual size_t GetMaxAllocSize(size_t dev_id);

  virtual size_t GetMaxChunkSize(size_t dev_id);

  virtual size_t GetExtraPaddingSize(size_t dev_id);

 private:
  const std::string type_;
  const uint8_t priority_;
  const bool is_custom_;

  size_t AllocSize(size_t dev_id, bool realloc);

  size_t AvailableAllocSize(size_t dev_id);
};

}  // namespace platform
}  // namespace paddle

#endif
