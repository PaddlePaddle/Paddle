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
#ifdef PADDLE_WITH_PLUGGABLE_DEVICE

#include "paddle/fluid/platform/device/device_base.h"
#include "paddle/fluid/platform/device/device_ext.h"
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
class Device final {
 public:
  using MemoryCpyKind = DeviceInterface::MemoryCpyKind;
  using MemoryAllocKind = DeviceInterface::MemoryAllocKind;

  Device(size_t dev_id, DeviceInterface* impl) : dev_id_(dev_id), impl_(impl) {}

  // Stream
  // ! Create an asynchronous stream
  void CreateStream(
      stream::Stream* stream, const stream::Stream::Priority& priority =
                                  stream::Stream::Priority::kNormal,
      const stream::Stream::Flag& flag = stream::Stream::Flag::kDefaultFlag);

  // ! Destroys an asynchronous stream.
  void DestroyStream(stream::Stream* stream);

  // ! Waits for stream tasks to complete.
  void SynchronizeStream(const stream::Stream* stream);

  // ! Queries an asynchronous stream for completion status.
  bool QueryStream(const stream::Stream* stream);

  // ! Add a callback to a compute stream.
  void AddCallback(stream::Stream* stream, stream::Stream::Callback* callback);

  // Event
  // ! Create an event.
  void CreateEvent(event::Event* event, event::Event::Flag flags);

  // ! Destroy an event.
  void DestroyEvent(event::Event* event);

  // ! Records an event.
  void RecordEvent(const event::Event* event, const stream::Stream* stream);

  // ! Waits for event to complete.
  void SynchronizeEvent(const event::Event* event);

  // ! Queries an event for completion status.
  bool QueryEvent(const event::Event* event);

  // ! Make a compute stream wait on an event
  void StreamWaitEvent(const stream::Stream* stream, const event::Event* event);

  // Memory
  void MemoryCopy(void* dst, const void* src, size_t size, MemoryCpyKind kind,
                  const stream::Stream* stream = nullptr);

  void MemoryCopyPeer(const Place& dst_place, void* dst, const void* src,
                      size_t size, const stream::Stream* stream = nullptr);

  void* MemoryAllocate(size_t size,
                       MemoryAllocKind kind = MemoryAllocKind::Normal);

  void MemoryDeallocate(void* ptr, size_t size,
                        MemoryAllocKind kind = MemoryAllocKind::Normal);

  void MemorySet(void* ptr, uint8_t value, size_t size);

  std::string Type();

 private:
  size_t dev_id_;
  DeviceInterface* impl_;
};

class DeviceManager {
 public:
  static bool Register(std::unique_ptr<DeviceInterface> device);
  static bool RegisterPinnedDevice(DeviceInterface* device);
  static Device* GetDeviceWithPlace(const Place& place);
  static std::vector<std::string> AllDeviceTypes();
  static std::vector<std::string> AllPluggableDeviceTypes();
  static std::vector<std::string> ListAllVisibleDevices();
  static std::vector<std::string> ListAllVisiblePluggableDevices();
  static bool HasDeviceType(const std::string& device_type);
  static bool IsPluggable(const std::string& device_type);

  // platform & device
  static void Initialize(const std::string& device_type);

  static void Finalize(const std::string& device_type);

  static void SynchronizeDevice(const Place& place);

  static void InitDevice(const Place& place);

  static void DeInitDevice(const Place& place);

  static void SetDevice(const std::string& device_type, size_t device_id);

  static void SetDevice(const Place& place);

  static int GetDevice(const std::string& device_type);

  static size_t GetMinChunkSize(const Place& place);

  static size_t GetMaxChunkSize(const Place& place);

  static size_t GetMaxAllocSize(const Place& place);

  static size_t GetInitAllocSize(const Place& place);

  static size_t GetReallocSize(const Place& place);

  static size_t GetExtraPaddingSize(const Place& place);

  static void MemoryStats(const Place& place, size_t* total, size_t* free);

  static size_t VisibleDevicesCount(const std::string& device_type);

 private:
  DISABLE_COPY_AND_ASSIGN(DeviceManager);
  DeviceManager() {}
  static DeviceManager& Instance();
  static DeviceInterface* GetDeviceInterfaceWithType(
      const std::string& device_type);

  std::unordered_map<std::string, std::unique_ptr<DeviceInterface>>
      device_impl_map_;
  std::unordered_map<std::string, std::vector<std::unique_ptr<Device>>>
      device_map_;
};

bool LoadRuntimePlugin(const std::string& plugin_path);

// just for test
bool LoadRuntimePlugin(const RuntimePluginParams& plugin_params,
                       std::unique_ptr<C_DeviceInterface> device_interface,
                       void* dso_handle);

bool LoadPluggableDevice(const std::string& library_path);

class Registrar {
 public:
  template <typename DeviceT>
  explicit Registrar(DeviceT* device_ptr) {
    DeviceManager::Register(std::unique_ptr<DeviceT>(device_ptr));
  }

  void Touch() {}
};

#define REGISTER_BUILTIN_DEVICE(device_type, DeviceT)                    \
  static ::paddle::platform::Registrar __builtin_device_##device_type__( \
      new DeviceT(#device_type, 255, false));                            \
  int TouchBuiltInDeviceRegistrar_##device_type() {                      \
    __builtin_device_##device_type__.Touch();                            \
    return 0;                                                            \
  }

#define USE_DEVICE(device_type)                            \
  extern int TouchBuiltInDeviceRegistrar_##device_type();  \
  UNUSED static int use_buiiltin_device_##device_type##_ = \
      TouchBuiltInDeviceRegistrar_##device_type()

}  // namespace platform
}  // namespace paddle

#endif
