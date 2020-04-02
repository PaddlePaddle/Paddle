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
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

#include "glog/logging.h"

namespace paddle {
namespace memory {

AllocationPtr Alloc(const platform::DeviceContext& dev_ctx, size_t size) {
  auto place = dev_ctx.GetPlace();
#ifdef PADDLE_WITH_CUDA
  if (size == 0 || !platform::is_gpu_place(place)) {
    return Alloc(place, size);
  }
  auto* default_dev_ctx = static_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto& desired_dev_ctx =
      static_cast<const platform::CUDADeviceContext&>(dev_ctx);
  if (default_dev_ctx->stream() == desired_dev_ctx.stream()) {
    return Alloc(place, size);
  } else {
    return allocation::CUDADeviceContextAllocatorPool::Instance().Alloc(
        desired_dev_ctx, size);
  }
#else
  return Alloc(place, size);
#endif
}

}  // namespace memory
}  // namespace paddle

namespace paddle {
namespace platform {

DeviceContextPool* DeviceContextPool::pool = nullptr;

platform::DeviceContext* DeviceContextPool::Get(const platform::Place& place) {
  auto it = device_contexts_.find(place);
  if (it == device_contexts_.end()) {
    PADDLE_THROW(
        "Place %s is not supported, Please check that your paddle compiles "
        "with WITH_GPU "
        "option or check that your train process hold the correct gpu_id if "
        "you use Executor",
        place);
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
#ifdef PADDLE_WITH_CUDA
      EmplaceDeviceContext<CUDADeviceContext, CUDAPlace>(&device_contexts_, p);
#else
      PADDLE_THROW(
          "'CUDAPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    } else if (platform::is_cuda_pinned_place(p)) {
#ifdef PADDLE_WITH_CUDA
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
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = memory::Alloc(place_, num_bytes);
    VLOG(4) << "Eigen allocated at " << buf->ptr() << ", size" << buf->size()
            << " requested " << num_bytes;
    void* retv = buf->ptr();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.emplace(retv, std::move(buf));
    }
    return retv;
  }

  void deallocate(void* buffer) const override {
    if (LIKELY(buffer)) {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.erase(buffer);
    }
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
      PADDLE_ENFORCE_CUDA_SUCCESS(
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
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void*, memory::AllocationPtr> allocations_;
};

void CudnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
  if (required_workspace_bytes <= WorkspaceSize()) {
    return;
  }
  // reset allocation first before re-allocate to save memory
  allocation_.reset();
  allocation_ = memory::Alloc(device_context_, required_workspace_bytes);
}

CUDADeviceContext::CUDADeviceContext(CUDAPlace place) : place_(place) {
  CUDADeviceGuard guard(place_.device);
  compute_capability_ = GetCUDAComputeCapability(place_.device);
  multi_process_ = GetCUDAMultiProcessors(place_.device);
  max_threads_per_mp_ = GetCUDAMaxThreadsPerMultiProcessor(place_.device);
  max_grid_dim_size_ = GetGpuMaxGridDimSize(place_.device);
  max_threads_per_block_ = GetCUDAMaxThreadsPerBlock(place_.device);
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream_));
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  cublas_handle_.reset(new CublasHandleHolder(stream_, CUBLAS_DEFAULT_MATH));

  if (TensorCoreAvailable()) {
#if CUDA_VERSION >= 9000
    cublas_tensor_core_handle_.reset(
        new CublasHandleHolder(stream_, CUBLAS_TENSOR_OP_MATH));
#endif
  }

  driver_version_ = GetCUDADriverVersion(place_.device);
  runtime_version_ = GetCUDARuntimeVersion(place_.device);

  LOG_FIRST_N(WARNING, 1) << "Please NOTE: device: " << place_.device
                          << ", CUDA Capability: " << compute_capability_
                          << ", Driver API Version: " << driver_version_ / 1000
                          << "." << (driver_version_ % 100) / 10
                          << ", Runtime API Version: "
                          << runtime_version_ / 1000 << "."
                          << (runtime_version_ % 100) / 10;
  size_t cudnn_dso_ver = dynload::cudnnGetVersion();
  LOG_FIRST_N(WARNING, 1) << "device: " << place_.device
                          << ", cuDNN Version: " << cudnn_dso_ver / 1000 << "."
                          << (cudnn_dso_ver % 1000) / 100 << ".";

  {
    // Check CUDA/CUDNN version compatiblity
    auto local_cuda_version =
        (driver_version_ / 1000) * 10 + (driver_version_ % 100) / 10;
    auto compile_cuda_version =
        (CUDA_VERSION / 1000) * 10 + (CUDA_VERSION % 100) / 10;
    if (local_cuda_version < compile_cuda_version) {
      LOG_FIRST_N(WARNING, 1)
          << "WARNING: device: " << place_.device
          << ". The installed Paddle is compiled with CUDA "
          << compile_cuda_version / 10 << "." << compile_cuda_version % 10
          << ", but CUDA runtime version in your machine is "
          << local_cuda_version / 10 << "." << local_cuda_version % 10
          << ", which may cause serious incompatible bug. "
          << "Please recompile or reinstall Paddle with compatible CUDA "
             "version.";
    }

    if (dynload::HasCUDNN()) {
      auto local_cudnn_version = cudnn_dso_ver / 100;
      auto compile_cudnn_version = CUDNN_VERSION / 100;
      if (local_cudnn_version < static_cast<size_t>(compile_cudnn_version)) {
        LOG_FIRST_N(WARNING, 1)
            << "WARNING: device: " << place_.device
            << ". The installed Paddle is compiled with CUDNN "
            << compile_cudnn_version / 10 << "." << compile_cudnn_version % 10
            << ", but CUDNN version in your machine is "
            << local_cudnn_version / 10 << "." << local_cudnn_version % 10
            << ", which may cause serious incompatible bug. "
            << "Please recompile or reinstall Paddle with compatible CUDNN "
               "version.";
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(
          dynload::cudnnCreate(&cudnn_handle_),
          "Failed to create Cudnn handle in DeviceContext");
      PADDLE_ENFORCE_CUDA_SUCCESS(
          dynload::cudnnSetStream(cudnn_handle_, stream_),
          "Failed to set stream for Cudnn handle in DeviceContext");
    } else {
      cudnn_handle_ = nullptr;
    }
  }

  callback_manager_.reset(new StreamCallbackManager(stream_));
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  WaitStreamCallback();
  cublas_handle_.reset();
  cublas_tensor_core_handle_.reset();
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
  if (cudnn_handle_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroy(cudnn_handle_),
                                "Failed to destory Cudnn handle");
  }
#if defined(PADDLE_WITH_NCCL)
  if (nccl_comm_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclCommDestroy(nccl_comm_));
  }
#endif
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
  e_sync = cudaStreamSynchronize(stream_);
#else
  while (e_sync = cudaStreamQuery(stream_)) {
    if (e_sync == cudaErrorNotReady) continue;
    break;
  }
#endif

  PADDLE_ENFORCE_CUDA_SUCCESS(
      e_sync, platform::errors::Fatal(
                  "cudaStreamSynchronize raises error: %s, errono: %d",
                  cudaGetErrorString(e_sync), static_cast<int>(e_sync)));
}

int CUDADeviceContext::GetComputeCapability() const {
  return compute_capability_;
}

int CUDADeviceContext::GetMaxPhysicalThreadCount() const {
  return multi_process_ * max_threads_per_mp_;
}

int CUDADeviceContext::GetSMCount() const { return multi_process_; }

int CUDADeviceContext::GetMaxThreadsPerBlock() const {
  return max_threads_per_block_;
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

bool CUDADeviceContext::tensor_core_available() const {
  return cublas_tensor_core_handle_ != nullptr;
}

dim3 CUDADeviceContext::GetCUDAMaxGridDimSize() const {
  return max_grid_dim_size_;
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() const { return cudnn_handle_; }

CudnnWorkspaceHandle CUDADeviceContext::cudnn_workspace_handle() const {
  return CudnnWorkspaceHandle(*this, &cudnn_handle_mtx_);
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

#ifdef PADDLE_WITH_MKLDNN
MKLDNNDeviceContext::MKLDNNDeviceContext(CPUPlace place)
    : CPUDeviceContext(place),
      engine_(mkldnn::engine::kind::cpu, 0),
      p_blobmap_() {
  p_blobmap_.reset(new BlobMap());
  p_mutex_.reset(new std::mutex());
}

namespace {
// Current mkldnn session id.
thread_local size_t cur_mkldnn_session_id = kMKLDNNSessionID_Default;
// Current data input shape string.
// - For fixed-shape, it's a null string in default.
// - For dynamic-shape, it's user specific.
thread_local std::string cur_input_shape_str = "";
// the cache capacity of different input shapes for MKLDNN.
// Default 1 means fixed input shape, not dynamic shape.
thread_local int cur_input_shape_cache_capacity = 1;
// Recently registered data_format. This is needed to
// know for converting MKL-DNN Tensor to non MKL-DNN
thread_local paddle::framework::DataLayout cur_paddle_data_layout =
    paddle::framework::DataLayout::kNCHW;
}  // namespace

void set_cur_mkldnn_session_id(size_t sid) { cur_mkldnn_session_id = sid; }
size_t get_cur_mkldnn_session_id(void) { return cur_mkldnn_session_id; }
void set_cur_input_shape_str(std::string input_shape_str) {
  cur_input_shape_str = input_shape_str;
}
void set_cur_input_shape_cache_capacity(int input_shape_cache_capacity) {
  cur_input_shape_cache_capacity = input_shape_cache_capacity;
}

void set_cur_paddle_data_layout(framework::DataLayout dl) {
  cur_paddle_data_layout = dl;
}

framework::DataLayout get_cur_paddle_data_layout(void) {
  return cur_paddle_data_layout;
}

void MKLDNNDeviceContext::ResetBlobMap() const { p_blobmap_->clear(); }

size_t MKLDNNDeviceContext::GetShapeBlobSize() const {
  std::lock_guard<std::mutex> lock(*p_mutex_);
  BlobMap* pMap = p_blobmap_.get();
  auto map_it = pMap->find(cur_mkldnn_session_id);
  if (map_it == pMap->end()) {
    LOG(FATAL) << "MKLDNNDeviceContext don't find cur_mkldnn_session_id : "
               << cur_mkldnn_session_id;
  }
  return map_it->second->size();
}

void MKLDNNDeviceContext::SetBlob(const std::string& name,
                                  std::shared_ptr<void> data) const {
  BlobMap* pMap = p_blobmap_.get();
  std::shared_ptr<ShapeBlob> sBlob = nullptr;
  std::shared_ptr<KeyBlob> pBlob = nullptr;

  int sid = platform::get_cur_mkldnn_session_id();

  std::lock_guard<std::mutex> lock(*p_mutex_);

  // Find ShapeBlob for current mkldnn session id.
  auto map_it = pMap->find(sid);

  if (map_it == pMap->end()) {
    // 1st time to set blob in current thread
    sBlob = std::shared_ptr<ShapeBlob>(new ShapeBlob());
    (*pMap)[sid] = sBlob;
    VLOG(2) << "SetBlob: sid=" << sid << ", add new sid\n";
  } else {
    sBlob = map_it->second;
  }

  // Find KeyBlob for current input shape
  auto key_it = sBlob->find(cur_input_shape_str);

  if (key_it == sBlob->end()) {
    // In cache clearing mode, cur_input_shape_cache_capacity defines
    // max pblob capacity
    if ((static_cast<size_t>(sid) == kMKLDNNSessionID_CacheClearing) &&
        sBlob->size() &&
        (sBlob->size() >=
         static_cast<size_t>(cur_input_shape_cache_capacity))) {
      VLOG(2) << "sid=" << sid
              << ", remove all blobs of shape: " << sBlob->begin()->first;
      sBlob->erase(sBlob->begin()->first);
    }
    pBlob = std::shared_ptr<KeyBlob>(new KeyBlob());
    (*sBlob)[cur_input_shape_str] = pBlob;
  } else {
    pBlob = key_it->second;
  }

  // Find Blob via name
  auto blob_it = pBlob->find(name);
  if (blob_it == pBlob->end()) {
    (*pBlob)[name] = data;
  } else {
    blob_it->second = data;  // set data to existing blob
  }
  VLOG(2) << "SetBlob: sid=" << sid << ", add blob=" << name << "\n";
  // lock will be automatically released when out of scope
  return;
}

std::shared_ptr<void> MKLDNNDeviceContext::GetBlob(
    const std::string& name) const {
  BlobMap* pMap = p_blobmap_.get();
  std::shared_ptr<ShapeBlob> sBlob = nullptr;
  std::shared_ptr<KeyBlob> pBlob = nullptr;

  int sid = platform::get_cur_mkldnn_session_id();

  std::lock_guard<std::mutex> lock(*p_mutex_);

  // Find ShapeBlob for current mkldnn session id firstly
  auto map_it = pMap->find(sid);
  if (map_it == pMap->end()) {
    VLOG(2) << "GetBlob: sid=" << sid << ", miss sid\n";
    return nullptr;
  }
  sBlob = map_it->second;

  // Find KeyBlob for current input shape secondly
  auto sBlob_it = sBlob->find(cur_input_shape_str);
  if (sBlob_it == sBlob->end()) {
    VLOG(2) << "GetBlob: sid=" << cur_input_shape_str
            << ", miss input_shape_str\n";
    return nullptr;
  }
  pBlob = sBlob_it->second;

  // Find Blob via name
  auto key_it = pBlob->find(name);

  if (key_it == pBlob->end()) {
    VLOG(2) << "GetBlob sid=" << sid << ", miss blob=" << name << "\n";
    return nullptr;
  }

  VLOG(2) << "GetBlob sid=" << sid << ", get blob=" << name << "\n";
  // lock will be automatically released when out of scope
  return key_it->second;
}

#endif

}  // namespace platform
}  // namespace paddle
