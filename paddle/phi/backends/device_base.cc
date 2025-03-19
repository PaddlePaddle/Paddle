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

#include "paddle/phi/backends/device_base.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"

PD_DECLARE_double(fraction_of_gpu_memory_to_use);
PD_DECLARE_uint64(initial_gpu_memory_in_mb);
PD_DECLARE_uint64(reallocate_gpu_memory_in_mb);

constexpr static float fraction_reserve_gpu_memory = 0.05f;

namespace phi {

#define INTERFACE_UNIMPLEMENT                 \
  PADDLE_THROW(common::errors::Unimplemented( \
      "%s is not implemented on %s device.", __func__, Type()));

// info
size_t DeviceInterface::GetComputeCapability() {
  VLOG(10) << Type() << " get compute capability " << 0;
  return 0;
}

size_t DeviceInterface::GetRuntimeVersion() {
  VLOG(10) << Type() << " get runtime version " << 0;
  return 0;
}

size_t DeviceInterface::GetDriverVersion() {
  VLOG(10) << Type() << " get driver version " << 0;
  return 0;
}

// device manage
void DeviceInterface::Initialize() { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::Finalize() { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::SynchronizeDevice(size_t dev_id) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::InitDevice(size_t dev_id) { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::DeInitDevice(size_t dev_id) { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::SetDevice(size_t dev_id) { INTERFACE_UNIMPLEMENT; }

int DeviceInterface::GetDevice() { INTERFACE_UNIMPLEMENT; }

// stream manage
void DeviceInterface::CreateStream(size_t dev_id,
                                   stream::Stream* stream,
                                   const stream::Stream::Priority& priority,
                                   const stream::Stream::Flag& flag) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::DestroyStream(size_t dev_id, stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::SynchronizeStream(size_t dev_id,
                                        const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

bool DeviceInterface::QueryStream(size_t dev_id, const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
  return true;
}

void DeviceInterface::AddCallback(size_t dev_id,
                                  stream::Stream* stream,
                                  stream::Stream::Callback* callback) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::StreamWaitEvent(size_t dev_id,
                                      const stream::Stream* stream,
                                      const event::Event* event) {
  INTERFACE_UNIMPLEMENT;
}

// event manage
void DeviceInterface::CreateEvent(size_t dev_id,
                                  event::Event* event,
                                  event::Event::Flag flags) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::DestroyEvent(size_t dev_id, event::Event* event) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::RecordEvent(size_t dev_id,
                                  const event::Event* event,
                                  const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::SynchronizeEvent(size_t dev_id,
                                       const event::Event* event) {
  INTERFACE_UNIMPLEMENT;
}

bool DeviceInterface::QueryEvent(size_t dev_id, const event::Event* event) {
  INTERFACE_UNIMPLEMENT;
  return true;
}

// memory manage
void DeviceInterface::MemoryCopyH2D(size_t dev_id,
                                    void* dst,
                                    const void* src,
                                    size_t size,
                                    const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::MemoryCopyD2H(size_t dev_id,
                                    void* dst,
                                    const void* src,
                                    size_t size,
                                    const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::MemoryCopyD2D(size_t dev_id,
                                    void* dst,
                                    const void* src,
                                    size_t size,
                                    const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::MemoryCopyP2P(const Place& dst_place,
                                    void* dst,
                                    size_t src_id,
                                    const void* src,
                                    size_t size,
                                    const stream::Stream* stream) {
  INTERFACE_UNIMPLEMENT;
}

void* DeviceInterface::MemoryAllocate(size_t dev_id, size_t size) {
  INTERFACE_UNIMPLEMENT;
  return nullptr;
}

void DeviceInterface::MemoryDeallocate(size_t dev_id, void* ptr, size_t size) {
  INTERFACE_UNIMPLEMENT;
}

void* DeviceInterface::MemoryAllocateHost(size_t dev_id, size_t size) {
  INTERFACE_UNIMPLEMENT;
  return nullptr;
}

void DeviceInterface::MemoryDeallocateHost(size_t dev_id,
                                           void* ptr,
                                           size_t size) {
  INTERFACE_UNIMPLEMENT;
}

void* DeviceInterface::MemoryAllocateUnified(size_t dev_id, size_t size) {
  INTERFACE_UNIMPLEMENT;
  return nullptr;
}

void DeviceInterface::MemoryDeallocateUnified(size_t dev_id,
                                              void* ptr,
                                              size_t size) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::MemorySet(size_t dev_id,
                                void* ptr,
                                uint8_t value,
                                size_t size) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::MemoryStats(size_t dev_id, size_t* total, size_t* free) {
  INTERFACE_UNIMPLEMENT;
}

size_t DeviceInterface::GetMinChunkSize(size_t dev_id) {
  INTERFACE_UNIMPLEMENT;
}

size_t DeviceInterface::AllocSize(size_t dev_id, bool realloc) {
  size_t available_to_alloc = AvailableAllocSize(dev_id);
  PADDLE_ENFORCE_GT(available_to_alloc,
                    0,
                    common::errors::ResourceExhausted(
                        "Not enough available %s memory.", Type()));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20
                     : available_to_alloc *
                           FLAGS_fraction_of_gpu_memory_to_use);  // NOLINT
  PADDLE_ENFORCE_GE(available_to_alloc,
                    alloc_bytes,
                    common::errors::ResourceExhausted(
                        "Not enough available %s memory.", Type()));
  return alloc_bytes;
}

size_t DeviceInterface::AvailableAllocSize(size_t dev_id) {
  size_t total = 0;
  size_t available = 0;
  MemoryStats(dev_id, &total, &available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t min_chunk_size = GetMinChunkSize(dev_id);
  if (available < min_chunk_size) {
    available = 0;
  }
  return available;
}

size_t DeviceInterface::GetInitAllocSize(size_t dev_id) {
  size_t init_alloc_size = AllocSize(dev_id, false);
  VLOG(10) << Type() << " init alloc size " << (init_alloc_size >> 20) << "M";
  return init_alloc_size;
}

size_t DeviceInterface::GetReallocSize(size_t dev_id) {
  size_t realloc_size = AllocSize(dev_id, true);
  VLOG(10) << Type() << " realloc size " << (realloc_size >> 20) << "M";
  return realloc_size;
}

size_t DeviceInterface::GetMaxAllocSize(size_t dev_id) {
  size_t max_alloc_size =
      std::max(GetInitAllocSize(dev_id), GetReallocSize(dev_id));
  VLOG(10) << Type() << " max alloc size " << (max_alloc_size >> 20) << "M";
  return max_alloc_size;
}

size_t DeviceInterface::GetMaxChunkSize(size_t dev_id) {
  size_t max_chunk_size = GetMaxAllocSize(dev_id);
  VLOG(10) << Type() << " max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

size_t DeviceInterface::GetExtraPaddingSize(size_t dev_id) {
  VLOG(10) << Type() << " extra padding size " << 0;
  return 0;
}

void DeviceInterface::CCLCommName(ccl::CCLComm ccl_comm, char* comm_name) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLDestroyComm(ccl::CCLComm ccl_comm) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLCommInitRank(size_t num_ranks,
                                      ccl::CCLRootId* root_id,
                                      size_t rank_id,
                                      ccl::CCLComm* ccl_comm) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLGetUniqueId(ccl::CCLRootId* root_id) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLBroadcast(void* data,
                                   size_t num,
                                   phi::DataType data_type,
                                   size_t root,
                                   const ccl::CCLComm& ccl_comm,
                                   const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLAllReduce(void* in_data,
                                   void* out_data,
                                   size_t num,
                                   phi::DataType data_type,
                                   ccl::CCLReduceOp reduce_op,
                                   const ccl::CCLComm& ccl_comm,
                                   const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLReduce(void* in_data,
                                void* out_data,
                                size_t num,
                                phi::DataType data_type,
                                ccl::CCLReduceOp reduce_op,
                                size_t root_id,
                                const ccl::CCLComm& ccl_comm,
                                const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLAllGather(void* in_data,
                                   void* out_data,
                                   size_t num,
                                   phi::DataType data_type,
                                   const ccl::CCLComm& ccl_comm,
                                   const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLReduceScatter(void* in_data,
                                       void* out_data,
                                       size_t num,
                                       phi::DataType data_type,
                                       ccl::CCLReduceOp op,
                                       const ccl::CCLComm& ccl_comm,
                                       const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLGroupStart() { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::CCLGroupEnd() { INTERFACE_UNIMPLEMENT; }

void DeviceInterface::CCLSend(void* sendbuf,
                              size_t num,
                              phi::DataType data_type,
                              size_t dst_rank,
                              const ccl::CCLComm& ccl_comm,
                              const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLRecv(void* recvbuf,
                              size_t num,
                              phi::DataType data_type,
                              size_t src_rank,
                              const ccl::CCLComm& ccl_comm,
                              const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::CCLAllToAll(const void** send_buf,
                                  const size_t* send_count,
                                  const phi::DataType* send_dtype,
                                  void** recv_buf,
                                  const size_t* recv_count,
                                  const phi::DataType* recv_dtype,
                                  size_t rank,
                                  size_t nranks,
                                  const ccl::CCLComm& comm,
                                  const stream::Stream& stream) {
  INTERFACE_UNIMPLEMENT;
}

// blas
void DeviceInterface::BlasAXPBY(size_t dev_id,
                                const stream::Stream& stream,
                                phi::DataType dtype,
                                size_t numel,
                                float alpha,
                                void* x,
                                float beta,
                                void* y) {
  INTERFACE_UNIMPLEMENT;
}

// profiler
void DeviceInterface::ProfilerInitialize(phi::TraceEventCollector* collector,
                                         void** user_data) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::ProfilerFinalize(phi::TraceEventCollector* collector,
                                       void* user_data) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::ProfilerPrepareTracing(
    phi::TraceEventCollector* collector, void* user_data) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::ProfilerStartTracing(phi::TraceEventCollector* collector,
                                           void* user_data) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::ProfilerStopTracing(phi::TraceEventCollector* collector,
                                          void* user_data) {
  INTERFACE_UNIMPLEMENT;
}

void DeviceInterface::ProfilerCollectTraceData(
    phi::TraceEventCollector* collector, uint64_t start_ns, void* user_data) {
  INTERFACE_UNIMPLEMENT;
}

#undef INTERFACE_UNIMPLEMENT

}  // namespace phi
