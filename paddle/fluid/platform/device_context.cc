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
#include "paddle/fluid/platform/device_context.h"
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/memory/memory.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

namespace paddle {
namespace platform {

DeviceContextPool* DeviceContextPool::pool = nullptr;

platform::DeviceContext* DeviceContextPool::Get(const platform::Place& place) {
  auto it = device_contexts_.find(place);
  if (it == device_contexts_.end()) {
    PADDLE_THROW(
        "'Place' is not supported, Please re-compile with WITH_GPU "
        "option");
  }
  return it->second.get().get();
}

template <typename DevCtx, typename PlaceType>
inline void EmplaceDeviceContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        map_ptr,
    platform::Place p) {
  using PtrType = std::unique_ptr<DeviceContext>;
  map_ptr->emplace(p, std::async(std::launch::deferred, [=] {
                     // lazy evaluation. i.e., only create device context at
                     // first `Get`
                     return PtrType(new DevCtx(boost::get<PlaceType>(p)));
                   }));
}

DeviceContextPool::DeviceContextPool(
    const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(), 0);
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }

  for (auto& p : set) {
    if (platform::is_cpu_place(p)) {
#ifdef PADDLE_WITH_MKLDNN
      EmplaceDeviceContext<MKLDNNDeviceContext, CPUPlace>(&device_contexts_, p);
#else
      EmplaceDeviceContext<CPUDeviceContext, CPUPlace>(&device_contexts_, p);
#endif
    } else if (platform::is_gpu_place(p)) {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
      EmplaceDeviceContext<CUDADeviceContext, CUDAPlace>(&device_contexts_, p);
#else
      PADDLE_THROW(
          "'CUDAPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    } else if (platform::is_cuda_pinned_place(p)) {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
      EmplaceDeviceContext<CUDAPinnedDeviceContext, CUDAPinnedPlace>(
          &device_contexts_, p);
#else
      PADDLE_THROW(
          "'CUDAPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    }
  }
}

CPUDeviceContext::CPUDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CPUDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CPUDeviceContext::GetPlace() const { return place_; }

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}

  void Reinitialize(const cudaStream_t* cuda_stream, CUDAPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const cudaStream_t& stream() const override { return *stream_; }

  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    auto buf = paddle::memory::Alloc(place_, num_bytes,
                                     memory::Allocator::kScratchpad);
    void* retv = buf->ptr();
    allocations_[buf->ptr()] = std::move(buf);
    return retv;
  }

  void deallocate(void* buffer) const override {
    allocations_.erase(allocations_.find(buffer));
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      PADDLE_ENFORCE(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  CUDAPlace place_;
  const cudaStream_t* stream_;         // not owned;
  const cudaDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::unordered_map<void*, memory::AllocationPtr> allocations_;
};

CudnnHolder::CudnnHolder(const cudaStream_t* stream, const CUDAPlace& place)
    : workspace_(nullptr), stream_(stream), place_(place) {
  PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
  PADDLE_ENFORCE(dynload::cudnnSetStream(cudnn_handle_, *stream_));
}

CudnnHolder::~CudnnHolder() {
  PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
}

void CudnnHolder::ReallocateWorkspace(size_t required_workspace_len) {
  if (required_workspace_len <= WorkspaceSize()) {
    return;
  }
  if (workspace_ != nullptr) {
    // Maybe someone is using the current workspace
    PADDLE_ENFORCE(cudaStreamSynchronize(*stream_));
    workspace_.reset();
  }
  workspace_ = paddle::memory::Alloc(place_, required_workspace_len,
                                     paddle::memory::Allocator::kScratchpad);
}

CUDADeviceContext::CUDADeviceContext(CUDAPlace place)
    : place_(place), cudnn_holder_(nullptr) {
  CUDADeviceGuard guard(place_.device);
  compute_capability_ = GetCUDAComputeCapability(place_.device);
  multi_process_ = GetCUDAMultiProcessors(place_.device);
  max_threads_per_mp_ = GetCUDAMaxThreadsPerMultiProcessor(place_.device);
  PADDLE_ENFORCE(cudaStreamCreate(&stream_));
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  PADDLE_ENFORCE(dynload::cublasCreate(&cublas_handle_));
  PADDLE_ENFORCE(dynload::cublasSetStream(cublas_handle_, stream_));
  if (dynload::HasCUDNN()) {
    cudnn_holder_.reset(new CudnnHolder(&stream_, place));
  }

  driver_version_ = GetCUDADriverVersion(place_.device);
  runtime_version_ = GetCUDARuntimeVersion(place_.device);

  LOG_FIRST_N(WARNING, 1) << "Please NOTE: device: " << place_.device
                          << ", CUDA Capability: " << compute_capability_
                          << ", Driver Version: " << driver_version_ / 1000
                          << "." << (driver_version_ % 100) / 10
                          << ", Runtime Version: " << runtime_version_ / 1000
                          << "." << (runtime_version_ % 100) / 10;
  size_t cudnn_dso_ver = dynload::cudnnGetVersion();
  LOG_FIRST_N(WARNING, 1) << "device: " << place_.device
                          << ", cuDNN Version: " << cudnn_dso_ver / 1000 << "."
                          << (cudnn_dso_ver % 100) / 10 << ".";
  callback_manager_.reset(new StreamCallbackManager(stream_));
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  WaitStreamCallback();
  PADDLE_ENFORCE(dynload::cublasDestroy(cublas_handle_));
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(cudaStreamDestroy(stream_));
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
  PADDLE_ENFORCE(cudaGetLastError());
}

int CUDADeviceContext::GetComputeCapability() const {
  return compute_capability_;
}

int CUDADeviceContext::GetMaxPhysicalThreadCount() const {
  return multi_process_ * max_threads_per_mp_;
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

cublasHandle_t CUDADeviceContext::cublas_handle() const {
  return cublas_handle_;
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() const {
  return cudnn_holder_->cudnn_handle();
}

CudnnWorkspaceHandle CUDADeviceContext::cudnn_workspace_handle() const {
  return CudnnWorkspaceHandle(cudnn_holder_.get());
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext(CUDAPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CUDAPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CUDAPinnedDeviceContext::GetPlace() const { return place_; }
#endif

#ifdef PADDLE_WITH_HIP

class EigenHipStreamDevice : public Eigen::StreamInterface {
 public:
  EigenHipStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenHipStreamDevice() override {}

  void Reinitialize(const hipStream_t* cuda_stream, CUDAPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const hipStream_t& stream() const override { return *stream_; }

  const hipDeviceProp_t& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    return paddle::memory::Alloc(place_, num_bytes);
  }

  void deallocate(void* buffer) const override {
    paddle::memory::Free(place_, buffer);
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kHipScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kHipScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      PADDLE_ENFORCE(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  CUDAPlace place_;
  const hipStream_t* stream_;         // not owned;
  const hipDeviceProp_t* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

CUDADeviceContext::CUDADeviceContext(CUDAPlace place) : place_(place) {
  SetDeviceId(place_.device);
  compute_capability = GetCUDAComputeCapability(place_.device);
  multi_process = GetCUDAMultiProcessors(place_.device);
  max_threads_per_mp = GetCUDAMaxThreadsPerMultiProcessor(place_.device);
  PADDLE_ENFORCE(hipStreamCreate(&stream_));
  eigen_stream_.reset(new EigenHipStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  PADDLE_ENFORCE(dynload::hipblasCreate(&hipblas_handle_));
  PADDLE_ENFORCE(dynload::hipblasSetStream(hipblas_handle_, stream_));
  if (dynload::HasMIOpen()) {
    PADDLE_ENFORCE(dynload::miopenCreate(&miopen_handle_));
    PADDLE_ENFORCE(dynload::miopenSetStream(miopen_handle_, stream_));
  } else {
    miopen_handle_ = nullptr;
  }
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  PADDLE_ENFORCE(dynload::hipblasDestroy(hipblas_handle_));
  if (miopen_handle_ != nullptr) {
    PADDLE_ENFORCE(dynload::miopenDestroy(miopen_handle_));
  }
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(hipStreamDestroy(stream_));
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  PADDLE_ENFORCE(hipStreamSynchronize(stream_));
  PADDLE_ENFORCE(hipGetLastError());
}

int CUDADeviceContext::GetComputeCapability() const {
  return compute_capability;
}

int CUDADeviceContext::GetMaxPhysicalThreadCount() const {
  return multi_process * max_threads_per_mp;
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

hipblasHandle_t CUDADeviceContext::hipblas_handle() const {
  return hipblas_handle_;
}

miopenHandle_t CUDADeviceContext::miopen_handle() const { return miopen_handle_; }

hipStream_t CUDADeviceContext::stream() const { return stream_; }

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext(CUDAPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CUDAPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CUDAPinnedDeviceContext::GetPlace() const { return place_; }
#endif

#ifdef PADDLE_WITH_MKLDNN
MKLDNNDeviceContext::MKLDNNDeviceContext(CPUPlace place)
    : CPUDeviceContext(place), engine_(mkldnn::engine::cpu, 0), p_blobmap_() {
  p_blobmap_.reset(new BlobMap());
  p_mutex_.reset(new std::mutex());
}

namespace {
// Current thread's id.
thread_local int cur_thread_id = 0;
}

void set_cur_thread_id(int tid) { cur_thread_id = tid; }
int get_cur_thread_id(void) { return cur_thread_id; }

void MKLDNNDeviceContext::SetBlob(const std::string& name,
                                  std::shared_ptr<void> data) const {
  BlobMap* pMap = p_blobmap_.get();
  std::shared_ptr<KeyBlob> pBlob = nullptr;

  int tid = platform::get_cur_thread_id();

  std::lock_guard<std::mutex> lock(*p_mutex_.get());

  // Find KeyBlob for current thread
  auto map_it = pMap->find(tid);

  if (map_it == pMap->end()) {
    // 1st time to set blob in current thread
    pBlob = std::shared_ptr<KeyBlob>(new KeyBlob());
    (*pMap)[tid] = pBlob;
  } else {
    pBlob = map_it->second;
  }

  // Find Key in found (or newly created) KeyBlob
  auto key_it = pBlob->find(name);

  if (key_it == pBlob->end()) {
    (*pBlob)[name] = data;  // create new blob
  } else {
    key_it->second = data;  // set data to existing blob
  }

  // lock will be automatically released when out of scope
  return;
}

std::shared_ptr<void> MKLDNNDeviceContext::GetBlob(
    const std::string& name) const {
  BlobMap* pMap = p_blobmap_.get();
  std::shared_ptr<KeyBlob> pBlob = nullptr;

  int tid = platform::get_cur_thread_id();

  std::lock_guard<std::mutex> lock(*p_mutex_.get());

  // Find KeyBlob for current thread firstly
  auto map_it = pMap->find(tid);
  if (map_it == pMap->end()) return nullptr;
  pBlob = map_it->second;

  // Find Blob via name
  auto key_it = pBlob->find(name);

  if (key_it == pBlob->end()) return nullptr;

  // lock will be automatically released when out of scope
  return key_it->second;
}

#endif

}  // namespace platform
}  // namespace paddle
