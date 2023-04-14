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

#include <unordered_map>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/utils/rw_lock.h"

#include "paddle/phi/backends/c_comm_lib.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/dynload/port.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"

namespace phi {
class Device final {
 public:
  Device(size_t dev_id, DeviceInterface* impl) : dev_id_(dev_id), impl_(impl) {}

  // Stream
  // ! Create an asynchronous stream
  void CreateStream(
      stream::Stream* stream,
      const stream::Stream::Priority& priority =
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
  void CreateEvent(event::Event* event,
                   event::Event::Flag flags = event::Event::Flag::Default);

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
  void MemoryCopyH2D(void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr);

  void MemoryCopyD2H(void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr);

  void MemoryCopyD2D(void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr);

  void MemoryCopyP2P(const Place& dst_place,
                     void* dst,
                     const void* src,
                     size_t size,
                     const stream::Stream* stream = nullptr);

  void* MemoryAllocate(size_t size);

  void MemoryDeallocate(void* ptr, size_t size);

  void* MemoryAllocateHost(size_t size);

  void MemoryDeallocateHost(void* ptr, size_t size);

  void* MemoryAllocateUnified(size_t size);

  void MemoryDeallocateUnified(void* ptr, size_t size);

  void MemorySet(void* ptr, uint8_t value, size_t size);

  // Blas
  // ! y = alpha * x + beta * y
  template <typename T>
  void BlasAXPBY(const stream::Stream& stream,
                 size_t numel,
                 float alpha,
                 const T* x,
                 float beta,
                 T* y);

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
  static std::vector<std::string> GetAllDeviceTypes();
  static std::vector<std::string> GetAllCustomDeviceTypes();
  static std::vector<std::string> GetAllDeviceList();
  static std::vector<std::string> GetAllCustomDeviceList();
  static bool HasDeviceType(const std::string& device_type);
  static bool IsCustom(const std::string& device_type);

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

  static size_t GetDeviceCount(const std::string& device_type);

  static std::vector<size_t> GetDeviceList(const std::string& device_type);

  static std::vector<size_t> GetSelectedDeviceList(
      const std::string& device_type);

  // CCL
  static void CCLDestroyComm(const std::string& device_type,
                             ccl::CCLComm ccl_comm);
  static void CCLCommInitRank(const std::string& device_type,
                              size_t num_ranks,
                              ccl::CCLRootId* root_id,
                              size_t rank_id,
                              ccl::CCLComm* ccl_comm);
  static void CCLGetUniqueId(const std::string& device_type,
                             ccl::CCLRootId* root_id);
  static void CCLBroadcast(const std::string& device_type,
                           void* data,
                           size_t num,
                           ccl::CCLDataType data_type,
                           size_t root,
                           const ccl::CCLComm& ccl_comm,
                           const stream::Stream& stream);
  static void CCLAllReduce(const std::string& device_type,
                           void* in_data,
                           void* out_data,
                           size_t num,
                           ccl::CCLDataType data_type,
                           ccl::CCLReduceOp reduce_op,
                           const ccl::CCLComm& ccl_comm,
                           const stream::Stream& stream);
  static void CCLReduce(const std::string& device_type,
                        void* in_data,
                        void* out_data,
                        size_t num,
                        ccl::CCLDataType data_type,
                        ccl::CCLReduceOp reduce_op,
                        size_t root_id,
                        const ccl::CCLComm& ccl_comm,
                        const stream::Stream& stream);
  static void CCLAllGather(const std::string& device_type,
                           void* in_data,
                           void* out_data,
                           size_t num,
                           ccl::CCLDataType data_type,
                           const ccl::CCLComm& ccl_comm,
                           const stream::Stream& stream);
  static void CCLReduceScatter(const std::string& device_type,
                               void* in_data,
                               void* out_data,
                               size_t num,
                               ccl::CCLDataType data_type,
                               ccl::CCLReduceOp op,
                               const ccl::CCLComm& ccl_comm,
                               const stream::Stream& stream);
  static void CCLGroupStart(const std::string& device_type);
  static void CCLGroupEnd(const std::string& device_type);
  static void CCLSend(const std::string& device_type,
                      void* sendbuf,
                      size_t num,
                      ccl::CCLDataType data_type,
                      size_t dst_rank,
                      const ccl::CCLComm& ccl_comm,
                      const stream::Stream& stream);
  static void CCLRecv(const std::string& device_type,
                      void* recvbuf,
                      size_t num,
                      ccl::CCLDataType data_type,
                      size_t src_rank,
                      const ccl::CCLComm& ccl_comm,
                      const stream::Stream& stream);

  // profiler
  static void ProfilerInitialize(const std::string& dev_type,
                                 phi::TraceEventCollector* collector,
                                 void** context);
  static void ProfilerFinalize(const std::string& dev_type,
                               phi::TraceEventCollector* collector,
                               void* context);
  static void ProfilerPrepareTracing(const std::string& dev_type,
                                     phi::TraceEventCollector* collector,
                                     void* context);
  static void ProfilerStartTracing(const std::string& dev_type,
                                   phi::TraceEventCollector* collector,
                                   void* context);
  static void ProfilerStopTracing(const std::string& dev_type,
                                  phi::TraceEventCollector* collector,
                                  void* context);
  static void ProfilerCollectTraceData(const std::string& dev_type,
                                       phi::TraceEventCollector* collector,
                                       uint64_t start_ns,
                                       void* context);

  static void Clear();

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

std::vector<std::string> ListAllLibraries(const std::string& library_dir);

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void LoadCustomRuntimeLib(const std::string& dso_lib_path, void* dso_handle);

void LoadCustomRuntimeLib(const CustomRuntimeParams& runtime_params,
                          std::unique_ptr<C_DeviceInterface> device_interface,
                          const std::string& dso_lib_path,
                          void* dso_handle);
#endif

class Registrar {
 public:
  template <typename DeviceT>
  explicit Registrar(DeviceT* device_ptr) {
    DeviceManager::Register(std::unique_ptr<DeviceT>(device_ptr));
  }

  void Touch() {}
};

}  // namespace phi
