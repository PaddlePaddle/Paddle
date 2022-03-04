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

#include "paddle/fluid/platform/device/custom/enforce_custom.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/callback_manager.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"

static bool operator==(const C_Device_st& d1, const C_Device_st& d2) {
  return d1.id == d2.id;
}

namespace phi {

class CustomDevice : public DeviceInterface {
 public:
  CustomDevice(const std::string& type,
               int priority,
               bool is_custom,
               std::unique_ptr<C_DeviceInterface> pimpl,
               void* dso_handle)
      : DeviceInterface(type, priority, is_custom),
        pimpl_(std::move(pimpl)),
        dso_handle_(dso_handle) {
    Initialize();
  }

  ~CustomDevice() override { Finalize(); }

  size_t GetDeviceCount() override {
    size_t count;
    if (pimpl_->get_device_count(&count) != C_SUCCESS) {
      count = 0;
    }
    return count;
  }

  std::vector<size_t> GetDeviceList() override {
    size_t count = GetDeviceCount();
    std::vector<size_t> devices(count);
    pimpl_->get_device_list(devices.data());
    return devices;
  }

  C_DeviceInterface* Impl() { return pimpl_.get(); }

  void SynchronizeDevice(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->synchronize_device(device));
  }

  void Initialize() override {
    if (pimpl_->initialize && pimpl_->initialize() != C_SUCCESS) {
      LOG(ERROR) << "Initialize " << Type() << " Failed\n";
      exit(-1);
    }
    auto devices = GetDeviceList();
    for (auto dev_id : devices) {
      C_Device_st device;
      device.id = dev_id;
      devices_pool[dev_id] = device;
      InitDevice(dev_id);
    }
  }

  void Finalize() override {
    auto devices = GetDeviceList();
    for (auto dev_id : devices) {
      // SetDevice(dev_id);
      // SynchronizeDevice(dev_id);
      DeInitDevice(dev_id);
    }

    bool ok = true;
    if (pimpl_->finalize && pimpl_->finalize() != C_SUCCESS) {
      LOG(ERROR) << "Finalize " << Type() << " Failed\n";
      ok = false;
    }
    if (dso_handle_) {
      dlclose(dso_handle_);
      dso_handle_ = nullptr;
    }
    if (!ok) {
      exit(1);
    }
  }

  void InitDevice(size_t dev_id) override {
    if (pimpl_->init_device) {
      // Core set logical id, and Plugin replace it with physical id
      const auto device = &devices_pool[dev_id];
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->init_device(device));
    }
  }

  void DeInitDevice(size_t dev_id) override {
    if (pimpl_->deinit_device) {
      const auto device = &devices_pool[dev_id];
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->deinit_device(device));
    }
  }

  void SetDevice(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->set_device(device));
  }

  int GetDevice() override {
    C_Device_st device;
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->get_device(&device));
    return device.id;
  }

  void CreateStream(size_t dev_id,
                    stream::Stream* stream,
                    const stream::Stream::Priority& priority =
                        stream::Stream::Priority::kNormal,
                    const stream::Stream::Flag& flag =
                        stream::Stream::Flag::kDefaultFlag) override {
    if (priority != stream::Stream::Priority::kNormal ||
        flag != stream::Stream::Flag::kDefaultFlag) {
      PADDLE_THROW(phi::errors::Unavailable(
          "priority != stream::Stream::Priority::kNormal || flag != "
          "stream::Stream::Flag::kDefaultFlag is not allowed on "
          "CustomDevice."));
    }
    const auto device = &devices_pool[dev_id];
    C_Stream c_stream;
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->create_stream(device, &c_stream));
    stream->set_stream(c_stream);
  }

  void DestroyStream(size_t dev_id, stream::Stream* stream) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->destroy_stream(
        device, reinterpret_cast<C_Stream>(stream->raw_stream())));
  }

  void SynchronizeStream(size_t dev_id, const stream::Stream* stream) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->synchronize_stream(
        device, reinterpret_cast<C_Stream>(stream->raw_stream())));
  }

  bool QueryStream(size_t dev_id, const stream::Stream* stream) override {
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->query_stream) {
      SynchronizeStream(dev_id, stream);
      return true;
    }
    if (pimpl_->query_stream(
            device, reinterpret_cast<C_Stream>(stream->raw_stream())) ==
        C_SUCCESS) {
      return true;
    }
    return false;
  }

  void AddCallback(size_t dev_id,
                   stream::Stream* stream,
                   stream::Stream::Callback* callback) override {
    if (!pimpl_->stream_add_callback) {
      PADDLE_THROW(phi::errors::Unavailable(
          "AddCallback is not supported on %s.", Type()));
    } else {
      const auto device = &devices_pool[dev_id];
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->stream_add_callback(
          device,
          reinterpret_cast<C_Stream>(stream->raw_stream()),
          [](C_Device device,
             C_Stream stream,
             void* user_data,
             C_Status* status) {
            std::unique_ptr<std::function<void()>> func(
                reinterpret_cast<std::function<void()>*>(user_data));
            (*func)();
          },
          callback));
    }
  }

  void CreateEvent(size_t dev_id,
                   event::Event* event,
                   event::Event::Flag flags) override {
    const auto device = &devices_pool[dev_id];
    C_Event c_event;

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->create_event(device, &c_event));
    event->set_event(c_event);
  }

  void DestroyEvent(size_t dev_id, event::Event* event) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->destroy_event(
        device, reinterpret_cast<C_Event>(event->raw_event())));
  }

  void RecordEvent(size_t dev_id,
                   const event::Event* event,
                   const stream::Stream* stream) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->record_event(device,
                             reinterpret_cast<C_Stream>(stream->raw_stream()),
                             reinterpret_cast<C_Event>(event->raw_event())));
  }

  void SynchronizeEvent(size_t dev_id, const event::Event* event) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->synchronize_event(
        device, reinterpret_cast<C_Event>(event->raw_event())));
  }

  bool QueryEvent(size_t dev_id, const event::Event* event) override {
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->query_event) {
      SynchronizeEvent(dev_id, event);
      return true;
    }
    if (pimpl_->query_event(device,
                            reinterpret_cast<C_Event>(event->raw_event())) ==
        C_SUCCESS) {
      return true;
    }
    return false;
  }

  void StreamWaitEvent(size_t dev_id,
                       const stream::Stream* stream,
                       const event::Event* event) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->stream_wait_event(
        device,
        reinterpret_cast<C_Stream>(stream->raw_stream()),
        reinterpret_cast<C_Event>(event->raw_event())));
  }

  void MemoryCopyH2D(size_t dev_id,
                     void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr) override {
    const auto device = &devices_pool[dev_id];
    auto place = CustomPlace(Type(), dev_id);

    if (stream && stream->raw_stream() && pimpl_->async_memory_copy_h2d) {
      C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->async_memory_copy_h2d(device, c_stream, dst, src, size));
    } else {
      paddle::platform::DeviceContextPool& pool =
          paddle::platform::DeviceContextPool::Instance();
      pool.Get(place)->Wait();
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->memory_copy_h2d(device, dst, src, size));
    }
  }

  void MemoryCopyD2H(size_t dev_id,
                     void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr) override {
    const auto device = &devices_pool[dev_id];
    auto place = CustomPlace(Type(), dev_id);

    if (stream && stream->raw_stream() && pimpl_->async_memory_copy_d2h) {
      C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->async_memory_copy_d2h(device, c_stream, dst, src, size));
    } else {
      paddle::platform::DeviceContextPool& pool =
          paddle::platform::DeviceContextPool::Instance();
      pool.Get(place)->Wait();
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->memory_copy_d2h(device, dst, src, size));
    }
  }

  void MemoryCopyD2D(size_t dev_id,
                     void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr) override {
    const auto device = &devices_pool[dev_id];
    auto place = CustomPlace(Type(), dev_id);

    if (stream && stream->raw_stream() && pimpl_->async_memory_copy_d2d) {
      C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->async_memory_copy_d2d(device, c_stream, dst, src, size));
    } else {
      paddle::platform::DeviceContextPool& pool =
          paddle::platform::DeviceContextPool::Instance();
      pool.Get(place)->Wait();
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->memory_copy_d2d(device, dst, src, size));
    }
  }

  void MemoryCopyP2P(const Place& dst_place,
                     void* dst,
                     size_t src_dev_id,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr) override {
    int dst_dev_id = PlaceToId(dst_place);
    auto dst_device = &devices_pool[dst_dev_id];
    auto src_device = &devices_pool[src_dev_id];

    if (stream && stream->raw_stream()) {
      if (!pimpl_->async_memory_copy_p2p) {
        MemoryCopyP2P(dst_place, dst, src_dev_id, src, size);
      } else {
        PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->async_memory_copy_p2p(
            dst_device,
            src_device,
            reinterpret_cast<C_Stream>(stream->raw_stream()),
            dst,
            src,
            size));
      }
    } else {
      if (!pimpl_->memory_copy_p2p) {
        std::unique_ptr<uint8_t> tmp(new uint8_t[size]);
        MemoryCopyD2H(src_dev_id, tmp.get(), src, size);
        MemoryCopyH2D(dst_dev_id, dst, tmp.get(), size);
      } else {
        auto src_place = CustomPlace(Type(), src_dev_id);
        paddle::platform::DeviceContextPool& pool =
            paddle::platform::DeviceContextPool::Instance();
        pool.Get(src_place)->Wait();
        PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
            pimpl_->memory_copy_p2p(dst_device, src_device, dst, src, size));
      }
    }
  }

  void* MemoryAllocate(size_t dev_id, size_t size) override {
    void* ptr = nullptr;
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->device_memory_allocate(device, &ptr, size));
    return ptr;
  }

  void MemoryDeallocate(size_t dev_id, void* ptr, size_t size) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->device_memory_deallocate(device, ptr, size));
  }

  void* MemoryAllocateHost(size_t dev_id, size_t size) override {
    void* ptr = nullptr;
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->unified_memory_allocate) {
      PADDLE_THROW(phi::errors::Unavailable(
          "MemoryAllocateHost is not supported on %s.", Type()));
    } else {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->host_memory_allocate(device, &ptr, size));
    }
    return ptr;
  }

  void MemoryDeallocateHost(size_t dev_id, void* ptr, size_t size) override {
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->host_memory_deallocate) {
      PADDLE_THROW(phi::errors::Unavailable(
          "MemoryDeallocateHost is not supported on %s.", Type()));
    } else {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->host_memory_deallocate(device, ptr, size));
    }
  }

  void* MemoryAllocateUnified(size_t dev_id, size_t size) override {
    void* ptr = nullptr;
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->unified_memory_allocate) {
      PADDLE_THROW(phi::errors::Unavailable(
          "MemoryAllocateUnified is not supported on %s.", Type()));
    } else {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->unified_memory_allocate(device, &ptr, size));
    }
    return ptr;
  }

  void MemoryDeallocateUnified(size_t dev_id, void* ptr, size_t size) override {
    const auto device = &devices_pool[dev_id];

    if (!pimpl_->unified_memory_deallocate) {
      PADDLE_THROW(phi::errors::Unavailable(
          "MemoryDeallocateUnified is not supported on %s.", Type()));
    } else {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->unified_memory_deallocate(device, ptr, size));
    }
  }

  void MemorySet(size_t dev_id,
                 void* ptr,
                 uint8_t value,
                 size_t size) override {
    const auto device = &devices_pool[dev_id];

    if (pimpl_->device_memory_set) {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->device_memory_set(device, ptr, value, size));
    } else {
      std::unique_ptr<uint8_t> tmp(new uint8_t[size]);
      memset(tmp.get(), value, size);
      MemoryCopyH2D(dev_id, ptr, tmp.get(), size);
    }
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    const auto device = &devices_pool[dev_id];

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->device_memory_stats(device, total, free));

    size_t used = *total - *free;
    VLOG(10) << Type() << " memory usage " << (used >> 20) << "M/"
             << (*total >> 20) << "M, " << (*free >> 20)
             << "M available to allocate";
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];

    size_t size = 0;
    pimpl_->device_min_chunk_size(device, &size);
    VLOG(10) << Type() << " min chunk size " << size << "B";
    return size;
  }

  size_t GetMaxChunkSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];

    size_t size = 0;
    if (pimpl_->device_max_chunk_size) {
      pimpl_->device_max_chunk_size(device, &size);
      VLOG(10) << Type() << " max chunk size " << size << "B";
    } else {
      return DeviceInterface::GetMaxChunkSize(dev_id);
    }
    return size;
  }

  size_t GetMaxAllocSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];

    size_t size = 0;
    if (pimpl_->device_max_alloc_size) {
      pimpl_->device_max_alloc_size(device, &size);
      VLOG(10) << Type() << " max alloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetMaxAllocSize(dev_id);
    }
    return size;
  }

  size_t GetInitAllocSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];
    size_t size = 0;
    if (pimpl_->device_init_alloc_size) {
      pimpl_->device_init_alloc_size(device, &size);
      VLOG(10) << Type() << " init alloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetInitAllocSize(dev_id);
    }
    return size;
  }

  size_t GetReallocSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];
    size_t size = 0;
    if (pimpl_->device_realloc_size) {
      pimpl_->device_realloc_size(device, &size);
      VLOG(10) << Type() << " realloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetReallocSize(dev_id);
    }
    return size;
  }

  size_t GetExtraPaddingSize(size_t dev_id) override {
    const auto device = &devices_pool[dev_id];

    size_t padding_size = 0;
    if (pimpl_->device_extra_padding_size) {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->device_extra_padding_size(device, &padding_size));
      VLOG(10) << Type() << " extra padding size " << (padding_size >> 20)
               << "M";
    } else {
      return DeviceInterface::GetExtraPaddingSize(dev_id);
    }
    return 0;
  }

  size_t GetComputeCapability() override {
    size_t compute_capability = 0;
    if (pimpl_->get_compute_capability) {
      pimpl_->get_compute_capability(&compute_capability);
    }
    VLOG(10) << Type() << " get compute capability " << compute_capability;
    return compute_capability;
  }

  size_t GetRuntimeVersion() override {
    size_t version = 0;
    if (pimpl_->get_runtime_version) {
      pimpl_->get_runtime_version(&version);
    }
    VLOG(10) << Type() << " get runtime version " << version;
    return version;
  }

  size_t GetDriverVersion() override {
    size_t version = 0;
    if (pimpl_->get_driver_version) {
      pimpl_->get_driver_version(&version);
    }
    VLOG(10) << Type() << " get driver version " << version;
    return version;
  }

 private:
  inline int PlaceToIdNoCheck(const Place& place) {
    int dev_id = place.GetDeviceId();
    return dev_id;
  }

  inline int PlaceToId(const Place& place) {
    int dev_id = PlaceToIdNoCheck(place);
    PADDLE_ENFORCE_NE(devices_pool.find(dev_id),
                      devices_pool.end(),
                      phi::errors::NotFound(
                          "Cannot found %s %d, please check visible devices",
                          Type(),
                          dev_id));
    return dev_id;
  }

  std::unique_ptr<C_DeviceInterface> pimpl_;
  void* dso_handle_;
  std::unordered_map<size_t, C_Device_st> devices_pool;
};

bool ValidCustomCustomRuntimeParams(const CustomRuntimeParams* params) {
#define CHECK_PTR(ptr, required)                                   \
  if (params->interface->ptr == nullptr && required) {             \
    LOG(WARNING) << "CustomRuntime [type: " << params->device_type \
                 << "] pointer: " << #ptr << " is not set.";       \
    return false;                                                  \
  }

  int version = params->version.major * 10000 + params->version.minor * 100 +
                params->version.patch;
  const int runtime_version = PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION * 10000 +
                              PADDLE_CUSTOM_RUNTIME_MINOR_VERSION * 100 +
                              PADDLE_CUSTOM_RUNTIME_PATCH_VERSION;

  if (version < runtime_version) {
    LOG(WARNING) << "CustomRuntime [type: " << params->device_type
                 << "] version: " << version
                 << " < PADDLE_CUSTOM_RUNTIME_VERSION " << runtime_version;
    return false;
  }

  CHECK_PTR(initialize, false);
  CHECK_PTR(finalize, false)

  CHECK_PTR(init_device, false);
  CHECK_PTR(set_device, true);
  CHECK_PTR(get_device, true);
  CHECK_PTR(deinit_device, false);

  CHECK_PTR(create_stream, true);
  CHECK_PTR(destroy_stream, true);
  CHECK_PTR(query_stream, false);
  CHECK_PTR(stream_add_callback, false);

  CHECK_PTR(create_event, true);
  CHECK_PTR(record_event, true);
  CHECK_PTR(destroy_event, true);
  CHECK_PTR(query_event, false);

  CHECK_PTR(synchronize_device, false);
  CHECK_PTR(synchronize_stream, true);
  CHECK_PTR(synchronize_event, true);
  CHECK_PTR(stream_wait_event, true);

  CHECK_PTR(device_memory_allocate, true);
  CHECK_PTR(device_memory_deallocate, true);
  CHECK_PTR(host_memory_allocate, false);
  CHECK_PTR(host_memory_deallocate, false);
  CHECK_PTR(unified_memory_allocate, false);
  CHECK_PTR(unified_memory_deallocate, false);
  CHECK_PTR(memory_copy_h2d, true);
  CHECK_PTR(memory_copy_d2h, true);
  CHECK_PTR(memory_copy_d2d, true);
  CHECK_PTR(memory_copy_p2p, false);
  CHECK_PTR(async_memory_copy_h2d, false);
  CHECK_PTR(async_memory_copy_d2h, false);
  CHECK_PTR(async_memory_copy_d2d, false);
  CHECK_PTR(async_memory_copy_p2p, false);

  CHECK_PTR(get_device_count, true);
  CHECK_PTR(get_device_list, true);
  CHECK_PTR(device_memory_stats, true);

  CHECK_PTR(device_min_chunk_size, true);
  CHECK_PTR(device_max_chunk_size, false);
  CHECK_PTR(device_max_alloc_size, false);
  CHECK_PTR(device_extra_padding_size, false);
  CHECK_PTR(get_compute_capability, false);
  CHECK_PTR(get_runtime_version, false);
  CHECK_PTR(get_driver_version, false);

  return true;
#undef CHECK_PTR
}

typedef bool (*RegisterDevicePluginFn)(CustomRuntimeParams* runtime_params);

void LoadCustomRuntimeLib(const CustomRuntimeParams& runtime_params,
                          std::unique_ptr<C_DeviceInterface> device_interface,
                          const std::string& dso_lib_path,
                          void* dso_handle) {
  if (ValidCustomCustomRuntimeParams(&runtime_params)) {
    auto device = std::make_unique<CustomDevice>(runtime_params.device_type,
                                                 255,
                                                 true,
                                                 std::move(device_interface),
                                                 dso_handle);
    if (false == DeviceManager::Register(std::move(device))) {
      LOG(WARNING) << "Skipped lib [" << dso_lib_path
                   << "]. Register failed!!! there may be a "
                      "Custom Runtime with the same name.";
    }
  } else {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path
                 << "]. Wrong parameters!!! please check the version "
                    "compatibility between PaddlePaddle and Custom Runtime.";
  }
}

void LoadCustomRuntimeLib(const std::string& dso_lib_path, void* dso_handle) {
  CustomRuntimeParams runtime_params;
  std::memset(&runtime_params, 0, sizeof(CustomRuntimeParams));
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  RegisterDevicePluginFn init_plugin_fn =
      reinterpret_cast<RegisterDevicePluginFn>(dlsym(dso_handle, "InitPlugin"));

  if (init_plugin_fn == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path << "]: fail to find "
                 << "InitPlugin symbol in this lib.";
    return;
  }

  init_plugin_fn(&runtime_params);
  if (runtime_params.device_type == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path
                 << "]: InitPlugin failed, please check the version "
                    "compatibility between PaddlePaddle and Custom Runtime.";
    return;
  }
  LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), dso_lib_path, dso_handle);
  LOG(INFO) << "Successed in loading custom runtime in lib: " << dso_lib_path;
}

}  // namespace phi
