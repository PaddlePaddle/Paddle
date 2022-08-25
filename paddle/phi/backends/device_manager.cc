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

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/complex.h"

#if !defined(_WIN32)
#include <dirent.h>
#else

#endif

#include <functional>
#include <regex>

#include "glog/logging.h"
#include "paddle/utils/string/split.h"

namespace phi {

void Device::CreateStream(stream::Stream* stream,
                          const stream::Stream::Priority& priority,
                          const stream::Stream::Flag& flag) {
  impl_->CreateStream(dev_id_, stream, priority, flag);
}

void Device::DestroyStream(stream::Stream* stream) {
  impl_->DestroyStream(dev_id_, stream);
}

void Device::SynchronizeStream(const stream::Stream* stream) {
  impl_->SynchronizeStream(dev_id_, stream);
}

bool Device::QueryStream(const stream::Stream* stream) {
  return impl_->QueryStream(dev_id_, stream);
}

void Device::AddCallback(stream::Stream* stream,
                         stream::Stream::Callback* callback) {
  impl_->AddCallback(dev_id_, stream, callback);
}

void Device::CreateEvent(event::Event* event, event::Event::Flag flags) {
  impl_->CreateEvent(dev_id_, event, flags);
}

void Device::DestroyEvent(event::Event* event) {
  impl_->DestroyEvent(dev_id_, event);
}

void Device::RecordEvent(const event::Event* event,
                         const stream::Stream* stream) {
  impl_->RecordEvent(dev_id_, event, stream);
}

void Device::SynchronizeEvent(const event::Event* event) {
  impl_->SynchronizeEvent(dev_id_, event);
}

bool Device::QueryEvent(const event::Event* event) {
  return impl_->QueryEvent(dev_id_, event);
}

void Device::StreamWaitEvent(const stream::Stream* stream,
                             const event::Event* event) {
  impl_->StreamWaitEvent(dev_id_, stream, event);
}

void Device::MemoryCopyH2D(void* dst,
                           const void* src,
                           size_t size,
                           const stream::Stream* stream) {
  impl_->MemoryCopyH2D(dev_id_, dst, src, size, stream);
}

void Device::MemoryCopyD2H(void* dst,
                           const void* src,
                           size_t size,
                           const stream::Stream* stream) {
  impl_->MemoryCopyD2H(dev_id_, dst, src, size, stream);
}

void Device::MemoryCopyD2D(void* dst,
                           const void* src,
                           size_t size,
                           const stream::Stream* stream) {
  impl_->MemoryCopyD2D(dev_id_, dst, src, size, stream);
}

void Device::MemoryCopyP2P(const Place& dst_place,
                           void* dst,
                           const void* src,
                           size_t size,
                           const stream::Stream* stream) {
  impl_->MemoryCopyP2P(dst_place, dst, dev_id_, src, size, stream);
}

void* Device::MemoryAllocate(size_t size) {
  return impl_->MemoryAllocate(dev_id_, size);
}

void Device::MemoryDeallocate(void* ptr, size_t size) {
  impl_->MemoryDeallocate(dev_id_, ptr, size);
}

void* Device::MemoryAllocateHost(size_t size) {
  return impl_->MemoryAllocateHost(dev_id_, size);
}

void Device::MemoryDeallocateHost(void* ptr, size_t size) {
  impl_->MemoryDeallocateHost(dev_id_, ptr, size);
}

void* Device::MemoryAllocateUnified(size_t size) {
  return impl_->MemoryAllocateUnified(dev_id_, size);
}

void Device::MemoryDeallocateUnified(void* ptr, size_t size) {
  impl_->MemoryDeallocateUnified(dev_id_, ptr, size);
}

void Device::MemorySet(void* ptr, uint8_t value, size_t size) {
  impl_->MemorySet(dev_id_, ptr, value, size);
}

template <typename T>
void Device::BlasAXPBY(const stream::Stream& stream,
                       size_t numel,
                       float alpha,
                       const T* x,
                       float beta,
                       T* y) {
  impl_->BlasAXPBY(dev_id_,
                   stream,
                   paddle::experimental::CppTypeToDataType<T>::Type(),
                   numel,
                   alpha,
                   reinterpret_cast<void*>(const_cast<T*>(x)),
                   beta,
                   reinterpret_cast<void*>(y));
}

template void Device::BlasAXPBY<paddle::float16>(const stream::Stream& stream,
                                                 size_t numel,
                                                 float alpha,
                                                 const paddle::float16* x,
                                                 float beta,
                                                 paddle::float16* y);
template void Device::BlasAXPBY<float>(const stream::Stream& stream,
                                       size_t numel,
                                       float alpha,
                                       const float* x,
                                       float beta,
                                       float* y);
template void Device::BlasAXPBY<double>(const stream::Stream& stream,
                                        size_t numel,
                                        float alpha,
                                        const double* x,
                                        float beta,
                                        double* y);
template void Device::BlasAXPBY<int8_t>(const stream::Stream& stream,
                                        size_t numel,
                                        float alpha,
                                        const int8_t* x,
                                        float beta,
                                        int8_t* y);
template void Device::BlasAXPBY<int16_t>(const stream::Stream& stream,
                                         size_t numel,
                                         float alpha,
                                         const int16_t* x,
                                         float beta,
                                         int16_t* y);
template void Device::BlasAXPBY<int32_t>(const stream::Stream& stream,
                                         size_t numel,
                                         float alpha,
                                         const int32_t* x,
                                         float beta,
                                         int32_t* y);
template void Device::BlasAXPBY<int64_t>(const stream::Stream& stream,
                                         size_t numel,
                                         float alpha,
                                         const int64_t* x,
                                         float beta,
                                         int64_t* y);
template void Device::BlasAXPBY<phi::dtype::complex<float>>(
    const stream::Stream& stream,
    size_t numel,
    float alpha,
    const phi::dtype::complex<float>* x,
    float beta,
    phi::dtype::complex<float>* y);
template void Device::BlasAXPBY<phi::dtype::complex<double>>(
    const stream::Stream& stream,
    size_t numel,
    float alpha,
    const phi::dtype::complex<double>* x,
    float beta,
    phi::dtype::complex<double>* y);

std::string Device::Type() { return impl_->Type(); }

static phi::RWLock _global_device_manager_rw_lock;

bool DeviceManager::Register(std::unique_ptr<DeviceInterface> device_impl) {
  phi::AutoWRLock lock(&_global_device_manager_rw_lock);
  VLOG(4) << "Register Device - " << device_impl->Type();
  auto device_type = device_impl->Type();
  auto& dev_impl_map = Instance().device_impl_map_;
  auto& dev_map = Instance().device_map_;

  if (dev_impl_map.find(device_type) == dev_impl_map.end()) {
    dev_impl_map.insert(
        std::pair<std::string, std::unique_ptr<DeviceInterface>>(
            device_type, std::move(device_impl)));
    auto& dev_impl = dev_impl_map[device_type];
    auto& dev_vec = dev_map[device_type];
    VLOG(4) << "GetDeviceCount is " << dev_impl->GetDeviceCount();
    for (size_t i = 0; i < dev_impl->GetDeviceCount(); ++i) {
      dev_vec.emplace_back(new Device(i, dev_impl.get()));
    }
  } else {
    auto& plat = dev_impl_map[device_type];
    if (plat->IsCustom() && plat->Priority() > device_impl->Priority()) {
      dev_impl_map[device_type] = std::move(device_impl);
      auto& dev_impl = dev_impl_map[device_type];
      auto& dev_vec = dev_map[device_type];
      dev_vec.clear();
      VLOG(4) << "GetDeviceCount is " << dev_impl->GetDeviceCount();
      for (size_t i = 0; i < dev_impl->GetDeviceCount(); ++i) {
        dev_vec.emplace_back(new Device(i, dev_impl.get()));
      }
    } else {
      return false;
    }
  }
  return true;
}

DeviceInterface* DeviceManager::GetDeviceInterfaceWithType(
    const std::string& device_type) {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);

  auto& dev_impl_map = Instance().device_impl_map_;
  PADDLE_ENFORCE_NE(
      dev_impl_map.find(device_type),
      dev_impl_map.end(),
      phi::errors::NotFound("%s interface not found.", device_type));
  return dev_impl_map.at(device_type).get();
}

Device* DeviceManager::GetDeviceWithPlace(const Place& place) {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);

  auto& dev_map = Instance().device_map_;
  auto dev_type = place.GetDeviceType();
  auto dev_id = place.GetDeviceId();
  PADDLE_ENFORCE_NE(
      dev_map.find(dev_type),
      dev_map.end(),
      phi::errors::NotFound("Unable to find Device with type %s.", dev_type));
  auto& dev_vec = dev_map[dev_type];
  PADDLE_ENFORCE_LT(
      dev_id,
      dev_vec.size(),
      phi::errors::OutOfRange(
          "The visible devices count of type %s is %d, but dev_id is %d.",
          dev_type,
          dev_vec.size(),
          dev_id));
  return dev_vec[dev_id].get();
}

std::vector<std::string> DeviceManager::GetAllDeviceTypes() {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);
  auto& dev_impl_map = Instance().device_impl_map_;
  std::vector<std::string> devices;
  for (auto iter = dev_impl_map.cbegin(); iter != dev_impl_map.cend(); ++iter) {
    devices.push_back(iter->first);
  }
  return devices;
}

std::vector<std::string> DeviceManager::GetAllCustomDeviceTypes() {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);
  auto& dev_impl_map = Instance().device_impl_map_;
  std::vector<std::string> devices;
  for (auto iter = dev_impl_map.cbegin(); iter != dev_impl_map.cend(); ++iter) {
    if (iter->second->IsCustom()) {
      devices.push_back(iter->first);
    }
  }
  return devices;
}

std::vector<std::string> DeviceManager::GetAllDeviceList() {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);
  auto& dev_impl_map = Instance().device_impl_map_;
  std::vector<std::string> devices;
  for (auto iter = dev_impl_map.cbegin(); iter != dev_impl_map.cend(); ++iter) {
    size_t device_count = iter->second->GetDeviceCount();
    std::string dev_type = iter->second->Type();
    if (device_count == 1) {
      devices.push_back(dev_type);
    } else {
      for (size_t i = 0; i < device_count; ++i) {
        devices.push_back(dev_type + ":" + std::to_string(i));
      }
    }
  }
  return devices;
}

std::vector<std::string> DeviceManager::GetAllCustomDeviceList() {
  phi::AutoRDLock lock(&_global_device_manager_rw_lock);
  auto& dev_impl_map = Instance().device_impl_map_;
  std::vector<std::string> devices;
  for (auto iter = dev_impl_map.cbegin(); iter != dev_impl_map.cend(); ++iter) {
    size_t device_count = iter->second->GetDeviceCount();
    std::string dev_type = iter->second->Type();
    if (iter->second->IsCustom()) {
      if (device_count == 1) {
        devices.push_back(dev_type);
      } else {
        for (size_t i = 0; i < device_count; ++i) {
          devices.push_back(dev_type + ":" + std::to_string(i));
        }
      }
    }
  }
  return devices;
}

bool DeviceManager::HasDeviceType(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl != nullptr;
}

bool DeviceManager::IsCustom(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->IsCustom();
}

void DeviceManager::Initialize(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->Initialize();
}

void DeviceManager::Finalize(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->Finalize();
}

void DeviceManager::SynchronizeDevice(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->SynchronizeDevice(device_id);
}

void DeviceManager::InitDevice(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->InitDevice(device_id);
}

void DeviceManager::DeInitDevice(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->DeInitDevice(device_id);
}

void DeviceManager::SetDevice(const std::string& device_type,
                              size_t device_id) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->SetDevice(device_id);
}

void DeviceManager::SetDevice(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  DeviceManager::SetDevice(device_type, device_id);
}

int DeviceManager::GetDevice(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetDevice();
}

size_t DeviceManager::GetMinChunkSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetMinChunkSize(device_id);
}

size_t DeviceManager::GetMaxChunkSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetMaxChunkSize(device_id);
}

size_t DeviceManager::GetMaxAllocSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetMaxAllocSize(device_id);
}

size_t DeviceManager::GetInitAllocSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetInitAllocSize(device_id);
}

size_t DeviceManager::GetReallocSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetReallocSize(device_id);
}

size_t DeviceManager::GetExtraPaddingSize(const Place& place) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetExtraPaddingSize(device_id);
}

void DeviceManager::MemoryStats(const Place& place,
                                size_t* total,
                                size_t* free) {
  auto device_type = place.GetDeviceType();
  auto device_id = place.GetDeviceId();
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->MemoryStats(device_id, total, free);
}

size_t DeviceManager::GetDeviceCount(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetDeviceCount();
}

std::vector<size_t> DeviceManager::GetDeviceList(
    const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  return dev_impl->GetDeviceList();
}

std::vector<size_t> DeviceManager::GetSelectedDeviceList(
    const std::string& device_type) {
  std::vector<size_t> devices;
  std::string FLAGS = "FLAGS_selected_" + device_type + "s";
  auto FLAGS_selected_devices = getenv(FLAGS.c_str());
  if (FLAGS_selected_devices) {
    auto devices_str = paddle::string::Split(FLAGS_selected_devices, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = DeviceManager::GetDeviceCount(device_type);
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void DeviceManager::CCLDestroyComm(const std::string& device_type,
                                   ccl::CCLComm ccl_comm) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLDestroyComm(ccl_comm);
}

void DeviceManager::CCLCommInitRank(const std::string& device_type,
                                    size_t num_ranks,
                                    ccl::CCLRootId* root_id,
                                    size_t rank_id,
                                    ccl::CCLComm* ccl_comm) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLCommInitRank(num_ranks, root_id, rank_id, ccl_comm);
}

void DeviceManager::CCLGetUniqueId(const std::string& device_type,
                                   ccl::CCLRootId* root_id) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLGetUniqueId(root_id);
}

void DeviceManager::CCLBroadcast(const std::string& device_type,
                                 void* data,
                                 size_t num,
                                 ccl::CCLDataType data_type,
                                 size_t root_id,
                                 const ccl::CCLComm& ccl_comm,
                                 const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLBroadcast(data, num, data_type, root_id, ccl_comm, stream);
}

void DeviceManager::CCLAllReduce(const std::string& device_type,
                                 void* in_data,
                                 void* out_data,
                                 size_t num,
                                 ccl::CCLDataType data_type,
                                 ccl::CCLReduceOp reduce_op,
                                 const ccl::CCLComm& ccl_comm,
                                 const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLAllReduce(
      in_data, out_data, num, data_type, reduce_op, ccl_comm, stream);
}

void DeviceManager::CCLReduce(const std::string& device_type,
                              void* in_data,
                              void* out_data,
                              size_t num,
                              ccl::CCLDataType data_type,
                              ccl::CCLReduceOp reduce_op,
                              size_t root_id,
                              const ccl::CCLComm& ccl_comm,
                              const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLReduce(
      in_data, out_data, num, data_type, reduce_op, root_id, ccl_comm, stream);
}

void DeviceManager::CCLAllGather(const std::string& device_type,
                                 void* in_data,
                                 void* out_data,
                                 size_t num,
                                 ccl::CCLDataType data_type,
                                 const ccl::CCLComm& ccl_comm,
                                 const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLAllGather(in_data, out_data, num, data_type, ccl_comm, stream);
}

void DeviceManager::CCLReduceScatter(const std::string& device_type,
                                     void* in_data,
                                     void* out_data,
                                     size_t num,
                                     ccl::CCLDataType data_type,
                                     ccl::CCLReduceOp op,
                                     const ccl::CCLComm& ccl_comm,
                                     const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLReduceScatter(
      in_data, out_data, num, data_type, op, ccl_comm, stream);
}

void DeviceManager::CCLGroupStart(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLGroupStart();
}

void DeviceManager::CCLGroupEnd(const std::string& device_type) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLGroupEnd();
}

void DeviceManager::CCLSend(const std::string& device_type,
                            void* sendbuf,
                            size_t num,
                            ccl::CCLDataType data_type,
                            size_t dst_rank,
                            const ccl::CCLComm& ccl_comm,
                            const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLSend(sendbuf, num, data_type, dst_rank, ccl_comm, stream);
}

void DeviceManager::CCLRecv(const std::string& device_type,
                            void* recvbuf,
                            size_t num,
                            ccl::CCLDataType data_type,
                            size_t src_rank,
                            const ccl::CCLComm& ccl_comm,
                            const stream::Stream& stream) {
  auto dev_impl = GetDeviceInterfaceWithType(device_type);
  dev_impl->CCLRecv(recvbuf, num, data_type, src_rank, ccl_comm, stream);
}

// profiler
void DeviceManager::ProfilerInitialize(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    void** context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerInitialize(collector, context);
}

void DeviceManager::ProfilerFinalize(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    void* context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerFinalize(collector, context);
}

void DeviceManager::ProfilerPrepareTracing(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    void* context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerPrepareTracing(collector, context);
}

void DeviceManager::ProfilerStartTracing(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    void* context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerStartTracing(collector, context);
}

void DeviceManager::ProfilerStopTracing(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    void* context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerStopTracing(collector, context);
}

void DeviceManager::ProfilerCollectTraceData(
    const std::string& dev_type,
    paddle::platform::TraceEventCollector* collector,
    uint64_t start_ns,
    void* context) {
  auto dev_impl = GetDeviceInterfaceWithType(dev_type);
  dev_impl->ProfilerCollectTraceData(collector, start_ns, context);
}

DeviceManager& DeviceManager::Instance() {
  static DeviceManager platform_manager;
  return platform_manager;
}

void DeviceManager::Clear() {
  // TODO(wangran16): fix coredump when using npu plugin

  // Instance().device_map_.clear();
  // Instance().device_impl_map_.clear();
}

std::vector<std::string> ListAllLibraries(const std::string& library_dir) {
  std::vector<std::string> libraries;
  std::regex express(".*\\.so");
  std::match_results<std::string::iterator> results;
  DIR* dir = nullptr;
  dirent* ptr = nullptr;

  dir = opendir(library_dir.c_str());
  if (dir == nullptr) {
    VLOG(4) << "Failed to open path: " << library_dir;
  } else {
    while ((ptr = readdir(dir)) != nullptr) {
      std::string filename(ptr->d_name);
      if (std::regex_match(
              filename.begin(), filename.end(), results, express)) {
        libraries.push_back(library_dir + '/' + filename);
        VLOG(4) << "Found lib: " << libraries.back();
      }
    }
    closedir(dir);
  }

  return libraries;
}

}  // namespace phi
#endif
