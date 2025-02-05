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

#include "glog/logging.h"

#include "paddle/phi/api/profiler/trace_event_collector.h"
#include "paddle/phi/backends/callback_manager.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/custom/enforce_custom.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/common/data_type.h"

static bool operator==(const C_Device_st& d1, const C_Device_st& d2) {
  return d1.id == d2.id;
}

namespace phi {

#define INTERFACE_UNIMPLEMENT                 \
  PADDLE_THROW(common::errors::Unimplemented( \
      "%s is not implemented on %s device.", __func__, Type()));
#define CHECK_PTR(x)       \
  if (x == nullptr) {      \
    INTERFACE_UNIMPLEMENT; \
  }

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
    if (!device_init_flag_) {
      if (pimpl_->get_device_count(&device_count_) != C_SUCCESS) {
        device_count_ = 0;
      } else {
        device_init_flag_ = true;
      }
    }
    return device_count_;
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
    }
  }

  void Finalize() override {
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
    const auto device = &devices_pool[dev_id];
    C_Stream c_stream;
    if (pimpl_->create_stream) {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->create_stream(device, &c_stream));
    } else {
      c_stream = nullptr;
    }
    stream->set_stream(c_stream);
  }

  void DestroyStream(size_t dev_id, stream::Stream* stream) override {
    if (pimpl_->destroy_stream) {
      const auto device = &devices_pool[dev_id];
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->destroy_stream(
          device, reinterpret_cast<C_Stream>(stream->raw_stream())));
    }
  }

  void SynchronizeStream(size_t dev_id, const stream::Stream* stream) override {
    if (pimpl_->synchronize_stream) {
      const auto device = &devices_pool[dev_id];
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->synchronize_stream(
          device, reinterpret_cast<C_Stream>(stream->raw_stream())));
    }
  }

  bool QueryStream(size_t dev_id, const stream::Stream* stream) override {
    if (!pimpl_->query_stream) {
      SynchronizeStream(dev_id, stream);
      return true;
    } else {
      const auto device = &devices_pool[dev_id];
      return pimpl_->query_stream(
                 device, reinterpret_cast<C_Stream>(stream->raw_stream())) ==
             C_SUCCESS;
    }
  }

  void AddCallback(size_t dev_id,
                   stream::Stream* stream,
                   stream::Stream::Callback* callback) override {
    if (!pimpl_->stream_add_callback) {
      PADDLE_THROW(common::errors::Unavailable(
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
    if (pimpl_->stream_wait_event) {
      const auto device = &devices_pool[dev_id];

      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->stream_wait_event(
          device,
          reinterpret_cast<C_Stream>(stream->raw_stream()),
          reinterpret_cast<C_Event>(event->raw_event())));
    }
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
    } else if (pimpl_->memory_copy_h2d) {
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
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
    } else if (pimpl_->memory_copy_d2h) {
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
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
    } else if (pimpl_->memory_copy_d2d) {
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
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
        std::unique_ptr<uint8_t[]> tmp(new uint8_t[size]);  // NOLINT
        MemoryCopyD2H(src_dev_id, tmp.get(), src, size);
        MemoryCopyH2D(dst_dev_id, dst, tmp.get(), size);
      } else {
        auto src_place = CustomPlace(Type(), src_dev_id);
        phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
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

    if (!pimpl_->host_memory_allocate) {
      PADDLE_THROW(common::errors::Unavailable(
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
      PADDLE_THROW(common::errors::Unavailable(
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
      PADDLE_THROW(common::errors::Unavailable(
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
      PADDLE_THROW(common::errors::Unavailable(
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
      std::unique_ptr<uint8_t[]> tmp(new uint8_t[size]);  // NOLINT
      memset(tmp.get(), value, size);
      MemoryCopyH2D(dev_id, ptr, tmp.get(), size);
    }
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    if (pimpl_->device_memory_stats) {
      const auto device = &devices_pool[dev_id];

      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
          pimpl_->device_memory_stats(device, total, free));

      size_t used = *total - *free;
      VLOG(10) << Type() << " memory usage " << (used >> 20) << "M/"
               << (*total >> 20) << "M, " << (*free >> 20)
               << "M available to allocate";
    } else {
      *total = 0;
      *free = 0;
    }
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    if (pimpl_->device_min_chunk_size) {
      const auto device = &devices_pool[dev_id];

      size_t size = 0;
      pimpl_->device_min_chunk_size(device, &size);
      VLOG(10) << Type() << " min chunk size " << size << "B";
      return size;
    } else {
      return 1;
    }
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
      VLOG(10) << Type() << " extra padding size:" << padding_size;
      return padding_size;
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

  C_CCLReduceOp ToXCCLReduceOp(ccl::CCLReduceOp reduce_op) {
#define return_result(in, ret) \
  case ccl::CCLReduceOp::in:   \
    return C_CCLReduceOp::ret
    switch (reduce_op) {
      return_result(SUM, SUM);
      return_result(AVG, AVG);
      return_result(MAX, MAX);
      return_result(MIN, MIN);
      return_result(PRODUCT, PRODUCT);
      default: {
        PADDLE_THROW(common::errors::Unavailable(
            "ReduceOp is not supported on %s.", Type()));
      }
    }
#undef return_result
  }

  C_DataType ToCDataType(phi::DataType data_type) {
#define return_result(in, ret) \
  case in:                     \
    return C_DataType::ret
    switch (data_type) {
      return_result(phi::DataType::BOOL, BOOL);
      return_result(phi::DataType::UINT8, UINT8);
      return_result(phi::DataType::UINT16, UINT16);
      return_result(phi::DataType::UINT32, UINT32);
      return_result(phi::DataType::UINT64, UINT64);
      return_result(phi::DataType::INT8, INT8);
      return_result(phi::DataType::INT16, INT16);
      return_result(phi::DataType::INT32, INT32);
      return_result(phi::DataType::INT64, INT64);
      return_result(phi::DataType::FLOAT16, FLOAT16);
      return_result(phi::DataType::FLOAT32, FLOAT32);
      return_result(phi::DataType::FLOAT64, FLOAT64);
      return_result(phi::DataType::BFLOAT16, BFLOAT16);
      return_result(phi::DataType::COMPLEX64, COMPLEX64);
      return_result(phi::DataType::COMPLEX128, COMPLEX128);
      default: {
        PADDLE_THROW(common::errors::Unavailable(
            "DataType is not supported on %s.", Type()));
        return C_DataType::UNDEFINED;
      }
    }
#undef return_result
  }

  void CCLGetUniqueId(ccl::CCLRootId* unique_id) override {
    CHECK_PTR(pimpl_->xccl_get_unique_id_size);
    CHECK_PTR(pimpl_->xccl_get_unique_id);

    C_CCLRootId root_id;
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->xccl_get_unique_id_size(&(root_id.sz)));
    root_id.data = new uint8_t[root_id.sz];
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_get_unique_id(&root_id));

    uint8_t* ptr = reinterpret_cast<uint8_t*>(root_id.data);
    *unique_id = std::vector<uint8_t>(ptr, ptr + root_id.sz);
    delete[] ptr;
  }

  void CCLCommInitRank(size_t nranks,
                       ccl::CCLRootId* unique_id,
                       size_t rank,
                       ccl::CCLComm* comm) override {
    CHECK_PTR(pimpl_->xccl_comm_init_rank);

    C_CCLRootId root_id;
    root_id.sz = unique_id->size();
    root_id.data = unique_id->data();

    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_comm_init_rank(
        nranks, &root_id, rank, reinterpret_cast<C_CCLComm*>(comm)));
  }

  void CCLDestroyComm(ccl::CCLComm comm) override {
    CHECK_PTR(pimpl_->xccl_destroy_comm);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->xccl_destroy_comm(reinterpret_cast<C_CCLComm>(comm)));
  }

  void CCLCommName(ccl::CCLComm comm, char* comm_name) {
    CHECK_PTR(pimpl_->xccl_get_comm_name);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_get_comm_name(
        reinterpret_cast<C_CCLComm>(comm), comm_name));
  }

  void CCLAllReduce(void* send_buf,
                    void* recv_buf,
                    size_t count,
                    phi::DataType data_type,
                    ccl::CCLReduceOp op,
                    const ccl::CCLComm& comm,
                    const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_all_reduce);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_all_reduce(
        send_buf,
        recv_buf,
        count,
        ToCDataType(data_type),
        ToXCCLReduceOp(op),
        reinterpret_cast<C_CCLComm>(comm),
        reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLBroadcast(void* buf,
                    size_t count,
                    phi::DataType data_type,
                    size_t root,
                    const ccl::CCLComm& comm,
                    const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_broadcast);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_broadcast(
        buf,
        count,
        ToCDataType(data_type),
        root,
        reinterpret_cast<C_CCLComm>(comm),
        reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLReduce(void* in_data,
                 void* out_data,
                 size_t num,
                 phi::DataType data_type,
                 ccl::CCLReduceOp reduce_op,
                 size_t root_id,
                 const ccl::CCLComm& comm,
                 const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_reduce);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->xccl_reduce(in_data,
                            out_data,
                            num,
                            ToCDataType(data_type),
                            ToXCCLReduceOp(reduce_op),
                            root_id,
                            reinterpret_cast<C_CCLComm>(comm),
                            reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLAllGather(void* send_buf,
                    void* recv_buf,
                    size_t count,
                    phi::DataType data_type,
                    const ccl::CCLComm& comm,
                    const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_all_gather);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_all_gather(
        send_buf,
        recv_buf,
        count,
        ToCDataType(data_type),
        reinterpret_cast<C_CCLComm>(comm),
        reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLReduceScatter(void* send_buf,
                        void* recv_buf,
                        size_t count,
                        phi::DataType data_type,
                        ccl::CCLReduceOp reduce_op,
                        const ccl::CCLComm& comm,
                        const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_reduce_scatter);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_reduce_scatter(
        send_buf,
        recv_buf,
        count,
        ToCDataType(data_type),
        ToXCCLReduceOp(reduce_op),
        reinterpret_cast<C_CCLComm>(comm),
        reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLGroupStart() override {
    if (pimpl_->xccl_group_start) {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_group_start());
    }
  }

  void CCLGroupEnd() override {
    if (pimpl_->xccl_group_end) {
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_group_end());
    }
  }

  void CCLSend(void* send_buf,
               size_t count,
               phi::DataType data_type,
               size_t dest_rank,
               const ccl::CCLComm& comm,
               const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_send);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->xccl_send(send_buf,
                          count,
                          ToCDataType(data_type),
                          dest_rank,
                          reinterpret_cast<C_CCLComm>(comm),
                          reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLRecv(void* recv_buf,
               size_t count,
               phi::DataType data_type,
               size_t src_rank,
               const ccl::CCLComm& comm,
               const stream::Stream& stream) override {
    CHECK_PTR(pimpl_->xccl_recv);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->xccl_recv(recv_buf,
                          count,
                          ToCDataType(data_type),
                          src_rank,
                          reinterpret_cast<C_CCLComm>(comm),
                          reinterpret_cast<C_Stream>(stream.raw_stream())));
  }

  void CCLAllToAll(const void** send_buf,
                   const size_t* send_count,
                   const phi::DataType* send_dtype,
                   void** recv_buf,
                   const size_t* recv_count,
                   const phi::DataType* recv_dtype,
                   size_t rank,
                   size_t nranks,
                   const ccl::CCLComm& comm,
                   const stream::Stream& stream) override {
    if (pimpl_->xccl_all_to_all) {
      std::vector<C_DataType> c_send_dtype, c_recv_dtype;
      for (size_t i = 0; i < nranks; ++i) {
        c_send_dtype.push_back(ToCDataType(send_dtype[i]));
        c_recv_dtype.push_back(ToCDataType(recv_dtype[i]));
      }
      PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_all_to_all(
          send_buf,
          send_count,
          c_send_dtype.data(),
          recv_buf,
          recv_count,
          c_recv_dtype.data(),
          rank,
          nranks,
          reinterpret_cast<C_CCLComm>(comm),
          reinterpret_cast<C_Stream>(stream.raw_stream())));
    } else if (pimpl_->xccl_send && pimpl_->xccl_recv) {
      // NOTE(wangran16): fallback to send and recv, while avoiding some devices
      // not supporting asynchronous send and recv.
      for (size_t i = 0; i < rank; ++i) {
        PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
            pimpl_->xccl_recv(recv_buf[i],
                              recv_count[i],
                              ToCDataType(recv_dtype[i]),
                              i,
                              reinterpret_cast<C_CCLComm>(comm),
                              reinterpret_cast<C_Stream>(stream.raw_stream())));
      }
      for (size_t i = 0; i < nranks; ++i) {
        if (i != rank) {
          PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->xccl_send(
              const_cast<void*>(send_buf[i]),
              send_count[i],
              ToCDataType(send_dtype[i]),
              i,
              reinterpret_cast<C_CCLComm>(comm),
              reinterpret_cast<C_Stream>(stream.raw_stream())));
        }
      }
      MemoryCopyD2D(rank,
                    recv_buf[rank],
                    send_buf[rank],
                    send_count[rank] * phi::SizeOf(send_dtype[rank]),
                    &stream);
      for (size_t i = rank + 1; i < nranks; ++i) {
        PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
            pimpl_->xccl_recv(recv_buf[i],
                              recv_count[i],
                              ToCDataType(recv_dtype[i]),
                              i,
                              reinterpret_cast<C_CCLComm>(comm),
                              reinterpret_cast<C_Stream>(stream.raw_stream())));
      }
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "CCLAllToAll is not supported on %s.", Type()));
    }
  }

  void BlasAXPBY(size_t dev_id,
                 const stream::Stream& stream,
                 phi::DataType dtype,
                 size_t numel,
                 float alpha,
                 void* x,
                 float beta,
                 void* y) override {
    CHECK_PTR(pimpl_->blas_axpby);
    const auto device = &devices_pool[dev_id];
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(
        pimpl_->blas_axpby(device,
                           reinterpret_cast<C_Stream>(stream.raw_stream()),
                           ToCDataType(dtype),
                           numel,
                           alpha,
                           x,
                           beta,
                           y));
  }

  // Profiler
  void ProfilerInitialize(phi::TraceEventCollector* collector,
                          void** user_data) override {
    CHECK_PTR(pimpl_->profiler_initialize);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_initialize(
        reinterpret_cast<C_Profiler>(collector), user_data));
  }

  void ProfilerFinalize(phi::TraceEventCollector* collector,
                        void* user_data) override {
    CHECK_PTR(pimpl_->profiler_finalize);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_finalize(
        reinterpret_cast<C_Profiler>(collector), user_data));
  }

  void ProfilerPrepareTracing(phi::TraceEventCollector* collector,
                              void* user_data) override {
    CHECK_PTR(pimpl_->profiler_prepare_tracing);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_prepare_tracing(
        reinterpret_cast<C_Profiler>(collector), user_data));
  }

  void ProfilerStartTracing(phi::TraceEventCollector* collector,
                            void* user_data) override {
    CHECK_PTR(pimpl_->profiler_start_tracing);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_start_tracing(
        reinterpret_cast<C_Profiler>(collector), user_data));
  }

  void ProfilerStopTracing(phi::TraceEventCollector* collector,
                           void* user_data) override {
    CHECK_PTR(pimpl_->profiler_stop_tracing);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_stop_tracing(
        reinterpret_cast<C_Profiler>(collector), user_data));
  }

  void ProfilerCollectTraceData(phi::TraceEventCollector* collector,
                                uint64_t start_ns,
                                void* user_data) override {
    CHECK_PTR(pimpl_->profiler_collect_trace_data);
    PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(pimpl_->profiler_collect_trace_data(
        reinterpret_cast<C_Profiler>(collector), start_ns, user_data));
  }

 private:
  inline int PlaceToIdNoCheck(const Place& place) {
    int dev_id = place.GetDeviceId();  // NOLINT
    return dev_id;
  }

  inline int PlaceToId(const Place& place) {
    int dev_id = PlaceToIdNoCheck(place);
    PADDLE_ENFORCE_NE(devices_pool.find(dev_id),
                      devices_pool.end(),
                      common::errors::NotFound(
                          "Cannot found %s %d, please check visible devices",
                          Type(),
                          dev_id));
    return dev_id;
  }

  std::unique_ptr<C_DeviceInterface> pimpl_;
  void* dso_handle_;
  std::unordered_map<size_t, C_Device_st> devices_pool;
  bool device_init_flag_ = false;
  size_t device_count_;
};

bool ValidCustomCustomRuntimeParams(const CustomRuntimeParams* params) {
#define CHECK_INTERFACE(ptr, required)                             \
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

  CHECK_INTERFACE(initialize, false);
  CHECK_INTERFACE(finalize, false)

  CHECK_INTERFACE(init_device, false);
  CHECK_INTERFACE(set_device, true);
  CHECK_INTERFACE(get_device, true);
  CHECK_INTERFACE(deinit_device, false);

  CHECK_INTERFACE(create_stream, false);
  CHECK_INTERFACE(destroy_stream, false);
  CHECK_INTERFACE(query_stream, false);
  CHECK_INTERFACE(stream_add_callback, false);

  CHECK_INTERFACE(create_event, true);
  CHECK_INTERFACE(record_event, true);
  CHECK_INTERFACE(destroy_event, true);
  CHECK_INTERFACE(query_event, false);

  CHECK_INTERFACE(synchronize_device, false);
  CHECK_INTERFACE(synchronize_stream, false);
  CHECK_INTERFACE(synchronize_event, true);
  CHECK_INTERFACE(stream_wait_event, false);

  CHECK_INTERFACE(device_memory_allocate, true);
  CHECK_INTERFACE(device_memory_deallocate, true);
  CHECK_INTERFACE(host_memory_allocate, false);
  CHECK_INTERFACE(host_memory_deallocate, false);
  CHECK_INTERFACE(unified_memory_allocate, false);
  CHECK_INTERFACE(unified_memory_deallocate, false);
  CHECK_INTERFACE(memory_copy_h2d, false);
  CHECK_INTERFACE(memory_copy_d2h, false);
  CHECK_INTERFACE(memory_copy_d2d, false);
  CHECK_INTERFACE(memory_copy_p2p, false);
  CHECK_INTERFACE(async_memory_copy_h2d, false);
  CHECK_INTERFACE(async_memory_copy_d2h, false);
  CHECK_INTERFACE(async_memory_copy_d2d, false);
  CHECK_INTERFACE(async_memory_copy_p2p, false);

  CHECK_INTERFACE(get_device_count, true);
  CHECK_INTERFACE(get_device_list, true);
  CHECK_INTERFACE(device_memory_stats, false);

  CHECK_INTERFACE(device_min_chunk_size, false);
  CHECK_INTERFACE(device_max_chunk_size, false);
  CHECK_INTERFACE(device_max_alloc_size, false);
  CHECK_INTERFACE(device_extra_padding_size, false);
  CHECK_INTERFACE(get_compute_capability, false);
  CHECK_INTERFACE(get_runtime_version, false);
  CHECK_INTERFACE(get_driver_version, false);

  CHECK_INTERFACE(xccl_get_unique_id, false);
  CHECK_INTERFACE(xccl_get_unique_id_size, false);
  CHECK_INTERFACE(xccl_comm_init_rank, false);
  CHECK_INTERFACE(xccl_destroy_comm, false);
  CHECK_INTERFACE(xccl_all_reduce, false);
  CHECK_INTERFACE(xccl_broadcast, false);
  CHECK_INTERFACE(xccl_reduce, false);
  CHECK_INTERFACE(xccl_all_gather, false);
  CHECK_INTERFACE(xccl_reduce_scatter, false);
  CHECK_INTERFACE(xccl_group_start, false);
  CHECK_INTERFACE(xccl_group_end, false);
  CHECK_INTERFACE(xccl_send, false);
  CHECK_INTERFACE(xccl_recv, false);

  CHECK_INTERFACE(blas_axpby, false);

  CHECK_INTERFACE(profiler_initialize, false);
  CHECK_INTERFACE(profiler_finalize, false);
  CHECK_INTERFACE(profiler_prepare_tracing, false);
  CHECK_INTERFACE(profiler_start_tracing, false);
  CHECK_INTERFACE(profiler_stop_tracing, false);
  CHECK_INTERFACE(profiler_collect_trace_data, false);
  return true;
#undef CHECK_INTERFACE
}

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
                 << "]. Wrong Runtime parameters!!! please check the version "
                    "compatibility between PaddlePaddle and Custom Runtime.";
  }
}

#undef INTERFACE_UNIMPLEMENT

}  // namespace phi
