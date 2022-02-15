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
#include <functional>
#include <memory>
#include <set>
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/allocator.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/device_context_allocator.h"
#endif
#include "glog/logging.h"
#include "paddle/fluid/framework/expect.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace memory {

AllocationPtr Alloc(const platform::DeviceContext& dev_ctx, size_t size) {
  auto place = dev_ctx.GetPlace();
  if (size == 0) {
    return Alloc(place, size);
  }

  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't use CUDA device since it's not compiled with CUDA,"
        "Please recompile or reinstall Paddle with GPU support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    // TODO(liuyuhui): Consider xpu stream later
    return Alloc(place, size);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't use XPU device since it's not compiled with XPU,"
        "Please recompile or reinstall Paddle with XPU support."));
#endif
  } else if (platform::is_mlu_place(place)) {
#ifdef PADDLE_WITH_MLU
    auto* default_dev_ctx = static_cast<platform::MLUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    auto& desired_dev_ctx =
        static_cast<const platform::MLUDeviceContext&>(dev_ctx);
    if (default_dev_ctx->stream() == desired_dev_ctx.stream()) {
      return Alloc(place, size);
    } else {
      return allocation::MLUDeviceContextAllocatorPool::Instance().Alloc(
          desired_dev_ctx, size);
    }
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't use MLU device since it's not compiled with MLU,"
        "Please recompile or reinstall Paddle with MLU support."));
#endif
  } else {
    return Alloc(place, size);
  }
}

}  // namespace memory
}  // namespace paddle

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool allow_tf32_cublas = true;
void SetAllowTF32Cublas(bool active) { allow_tf32_cublas = active; }
bool AllowTF32Cublas() { return allow_tf32_cublas; }

bool allow_tf32_cudnn = true;
void SetAllowTF32Cudnn(bool active) { allow_tf32_cudnn = active; }
bool AllowTF32Cudnn() { return allow_tf32_cudnn; }
#endif  // PADDLE_WITH_CUDA

DeviceType Place2DeviceType(const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return platform::DeviceType::CPU;
  } else if (platform::is_gpu_place(place)) {
    return platform::DeviceType::CUDA;
  } else if (platform::is_xpu_place(place)) {
    return platform::DeviceType::XPU;
  } else if (platform::is_mlu_place(place)) {
    return platform::DeviceType::MLU;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported place %s to convert into platform::DeviceType.", place));
  }
}

DeviceContextPool* DeviceContextPool::pool = nullptr;

platform::DeviceContext* DeviceContextPool::Get(const platform::Place& place) {
  VLOG(6) << "DeviceContextPool Get: " << place;
  auto it = device_contexts_.find(place);
  if (it == device_contexts_.end()) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Place %s is not supported. Please check that your paddle compiles "
        "with WITH_GPU, WITH_XPU, WITH_IPU, WITH_MLU or WITH_ASCEND_CL option "
        "or check "
        "that your train process set the correct device id if you use "
        "Executor.",
        place));
  }
  return it->second.get().get();
}

template <typename DevCtx>
inline void EmplaceDeviceContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        map_ptr,
    platform::Place p) {
  using PtrType = std::unique_ptr<DeviceContext>;
  map_ptr->emplace(
      p, std::async(std::launch::deferred, [=] {
        // lazy evaluation. i.e., only create device context at
        // first `Get`
        auto* dev_ctx = new DevCtx(p);
        if (is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
          auto* cuda_ctx = dynamic_cast<CUDADeviceContext*>(dev_ctx);
          PADDLE_ENFORCE_NOT_NULL(
              cuda_ctx,
              platform::errors::InvalidArgument(
                  "Failed to dynamic_cast dev_ctx into CUDADeviceContext."));
          // Note: A trick method to init context, why GetAllocator interface
          // needs a stream parameter?
          dev_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetAllocator(p, cuda_ctx->stream())
                                    .get());
          cuda_ctx->PartialInitWithAllocator();
          dev_ctx->SetGenerator(
              framework::GetDefaultCUDAGenerator(p.GetDeviceId()).get());
#endif
        } else {
          dev_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetAllocator(p)
                                    .get());
          dev_ctx->SetGenerator(framework::DefaultCPUGenerator().get());
        }
        dev_ctx->SetHostAllocator(
            memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(platform::CPUPlace())
                .get());
        dev_ctx->SetZeroAllocator(
            memory::allocation::AllocatorFacade::Instance()
                .GetZeroAllocator(p)
                .get());
        return PtrType(dev_ctx);
      }));
}

DeviceContextPool::DeviceContextPool(
    const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(
      places.size(), 0,
      platform::errors::InvalidArgument("The number of platform places should "
                                        "be larger than 0. But received %d.",
                                        places.size()));
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }
  for (auto& p : set) {
    if (platform::is_cpu_place(p)) {
#ifdef PADDLE_WITH_MKLDNN
      EmplaceDeviceContext<MKLDNNDeviceContext>(&device_contexts_, p);
#else
      EmplaceDeviceContext<CPUDeviceContext>(&device_contexts_, p);
#endif
    } else if (platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<CUDADeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                          "re-compile with WITH_GPU option."));
#endif
    } else if (platform::is_cuda_pinned_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<CUDAPinnedDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported. Please re-compile with WITH_GPU "
          "option."));
#endif
    } else if (platform::is_xpu_place(p)) {
#ifdef PADDLE_WITH_XPU
      EmplaceDeviceContext<XPUDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("XPUPlace is not supported. Please "
                                          "re-compile with WITH_XPU option."));
#endif
    } else if (platform::is_mlu_place(p)) {
#ifdef PADDLE_WITH_MLU
      EmplaceDeviceContext<MLUDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("MLUPlace is not supported. Please "
                                          "re-compile with WITH_MLU option."));
#endif
    } else if (platform::is_ipu_place(p)) {
#ifdef PADDLE_WITH_IPU
      EmplaceDeviceContext<IPUDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("IPUPlace is not supported. Please "
                                          "re-compile with WITH_IPU option."));
#endif
    } else if (platform::is_npu_place(p)) {
#ifdef PADDLE_WITH_ASCEND_CL
      EmplaceDeviceContext<NPUDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "NPUPlace is not supported. Please "
          "re-compile with WITH_ASCEND_CL option."));
#endif
    } else if (platform::is_npu_pinned_place(p)) {
#ifdef PADDLE_WITH_ASCEND_CL
      EmplaceDeviceContext<NPUPinnedDeviceContext>(&device_contexts_, p);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "NPUPinnedPlace is not supported. Please re-compile with "
          "WITH_ASCEND_CL "
          "option."));
#endif
    }
  }
}

CPUDeviceContext::CPUDeviceContext() : pten::CPUContext() {
  pten::CPUContext::Init();
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) : pten::CPUContext(place) {
  pten::CPUContext::Init();
}

#ifdef PADDLE_WITH_IPU
IPUDeviceContext::IPUDeviceContext(IPUPlace place) : place_(place) {}

const Place& IPUDeviceContext::GetPlace() const { return place_; }

void IPUDeviceContext::Wait() const {
  /*! \brief  Wait for all operations completion in the stream. */
}

IPUDeviceContext::~IPUDeviceContext() {}

#endif
#ifdef PADDLE_WITH_XPU
XPUDeviceContext::XPUDeviceContext() : pten::XPUContext() {
  pten::XPUContext::Init();
}

XPUDeviceContext::~XPUDeviceContext() {}

XPUDeviceContext::XPUDeviceContext(XPUPlace place) : pten::XPUContext(place) {
  pten::XPUContext::Init();
  LOG_FIRST_N(WARNING, 1) << "Please NOTE: xpu device: "
                          << static_cast<int>(place.device);
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
NPUDeviceContext::NPUDeviceContext(NPUPlace place) : place_(place) {
  NPUDeviceGuard guard(place_.device);
  // PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateContext(&context_, place_.device));
  // NOTE(zhiqiu): Usually, no need to create context explicitly,
  // ACL creates a default context which contains 1 default stream
  // and 1 sync strean after aclrtSetDevice.
  platform::GetCurrentNPUContext(&context_);
  stream_.reset(new stream::NPUStream(place));
}

NPUDeviceContext::~NPUDeviceContext() {
  // NPUDeviceGuard guard(place_.device);
  // PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyContext(context_));
}

void NPUDeviceContext::Wait() const {
  platform::RecordEvent record_event("NPUDeviceContext/wait");
  VLOG(4) << "NPU context(" << this << ")  Wait";
  stream_->Wait();
}

aclrtStream NPUDeviceContext::stream() const { return stream_->raw_stream(); }

const Place& NPUDeviceContext::GetPlace() const { return place_; }

aclrtContext NPUDeviceContext::context() const { return context_; }

NPUPinnedDeviceContext::NPUPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

NPUPinnedDeviceContext::NPUPinnedDeviceContext(NPUPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* NPUPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

const Place& NPUPinnedDeviceContext::GetPlace() const { return place_; }

#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}

  void Reinitialize(const gpuStream_t* cuda_stream, CUDAPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const gpuStream_t& stream() const override { return *stream_; }

#ifdef PADDLE_WITH_HIP
  const hipDeviceProp_t& deviceProperties() const override {
#else
  const cudaDeviceProp& deviceProperties() const override {
#endif
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
      scratch_ = allocate(Eigen::kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch = static_cast<char*>(scratchpad()) + Eigen::kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
#endif
    }
    return semaphore_;
  }

 private:
  CUDAPlace place_;
  const gpuStream_t* stream_;  // not owned;
#ifdef PADDLE_WITH_HIP
  const hipDeviceProp_t* device_prop_;
#else
  const cudaDeviceProp* device_prop_;  // not owned;
#endif
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

thread_local std::unordered_map<const CUDADeviceContext*,
                                std::shared_ptr<CUDAContext>>
    CUDADeviceContext::thread_ctx_;
thread_local std::mutex CUDADeviceContext::ctx_mtx_;

void CUDAContext::InitEigenContext() {
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&RawStream(), place_);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
}

CUDAContext::CUDAContext(const CUDAPlace& place,
                         const stream::Priority& priority,
                         const stream::StreamFlag& flag) {
  place_ = place;
  CUDADeviceGuard guard(place_.device);
  stream_.reset(new stream::CUDAStream(place, priority, flag));
  InitEigenContext();
  InitCuBlasContext();
  InitCuDNNContext();
#ifndef PADDLE_WITH_HIP
  InitCuSparseContext();
  InitCuSolverContext();
#endif
}

void CUDAContext::SetStream(gpuStream_t stream) {
  if (stream_->raw_stream() != stream) {
    CUDADeviceGuard guard(place_.device);
    DestoryCuDNNContext();
    DestoryCuBlasContext();
#ifndef PADDLE_WITH_HIP
    DestoryCuSolverContext();
#endif

    stream_->SetStream(stream);

    InitEigenContext();
    InitCuBlasContext();
    InitCuDNNContext();
#ifndef PADDLE_WITH_HIP
    InitCuSolverContext();
#endif
  }
}

CUDAContext::~CUDAContext() {
  CUDADeviceGuard guard(place_.device);
  DestoryCuDNNContext();
  DestoryCuBlasContext();
#ifndef PADDLE_WITH_HIP
  DestoryCuSparseContext();
  DestoryCuSolverContext();
#endif
}

CUDADeviceContext::CUDADeviceContext(CUDAPlace place)
    : pten::GPUContext(place) {
  pten::GPUContext::PartialInitWithoutAllocator();
  cuda_stream_.reset(new stream::CUDAStream(pten::GPUContext::stream(), place));
  workspace_.reset(new pten::DnnWorkspaceHandle(
      memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(place, pten::GPUContext::stream())
          .get()));
}

CUDADeviceContext::~CUDADeviceContext() = default;

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  if (thread_ctx_.count(this)) {
    return context()->EigenDevice().get();
  }
  return pten::GPUContext::eigen_device();
}

void CUDADeviceContext::Wait() const {
  if (thread_ctx_.count(this)) {
    context()->Stream()->Wait();
    return;
  }
  pten::GPUContext::Wait();
}

#ifdef PADDLE_WITH_HIP
miopenHandle_t CUDADeviceContext::cudnn_handle() const {
#else
cudnnHandle_t CUDADeviceContext::cudnn_handle() const {
#endif
  if (thread_ctx_.count(this)) {
    return context()->CudnnHandle();
  }
  return pten::GPUContext::cudnn_handle();
}

#ifdef PADDLE_WITH_HIP
rocblas_handle CUDADeviceContext::cublas_handle() const {
  if (thread_ctx_.count(this)) {
    return context()->CublasHandle()->GetCublasHandle();
  }
  return pten::GPUContext::cublas_handle();
}
#else
cublasHandle_t CUDADeviceContext::cublas_handle() const {
  if (thread_ctx_.count(this)) {
    return context()->CublasHandle()->GetCublasHandle();
  }
  return pten::GPUContext::cublas_handle();
}
cusparseHandle_t CUDADeviceContext::cusparse_handle() const {
  if (thread_ctx_.count(this)) {
    return context()->CusparseHandle()->GetCusparseHandle();
  }
  return pten::GPUContext::cusparse_handle();
}
cusolverDnHandle_t CUDADeviceContext::cusolver_dn_handle() const {
  if (thread_ctx_.count(this)) {
    return context()->CusolverDnHandle();
  }
  return pten::GPUContext::cusolver_dn_handle();
}
#endif

void CUDADeviceContext::RecordEvent(
    gpuEvent_t ev, const std::function<void()>& callback) const {
  if (thread_ctx_.count(this)) {
    context()->Stream()->RecordEvent(ev, callback);
    return;
  }
  pten::GPUContext::RecordEvent(ev, callback);
}

void CUDADeviceContext::AddStreamCallback(
    const std::function<void()>& callback) const {
  if (thread_ctx_.count(this)) {
    context()->Stream()->AddCallback(callback);
    return;
  }
  pten::GPUContext::AddStreamCallback(callback);
}

void CUDADeviceContext::WaitStreamCallback() const {
  if (thread_ctx_.count(this)) {
    context()->Stream()->WaitCallback();
    return;
  }
  pten::GPUContext::WaitStreamCallback();
}

pten::DnnWorkspaceHandle CUDADeviceContext::cudnn_workspace_handle() const {
  if (thread_ctx_.count(this)) {
    // return workspace_.get();
    return pten::DnnWorkspaceHandle(
        memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(GetPlace(), pten::GPUContext::stream())
            .get());
  }
  return pten::GPUContext::cudnn_workspace_handle();
}

gpuStream_t CUDADeviceContext::stream() const {
  if (thread_ctx_.count(this)) {
    return context()->RawStream();
  }
  return pten::GPUContext::stream();
}

std::shared_ptr<CUDAContext> CUDADeviceContext::context() const {
  if (!thread_ctx_.count(this)) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "CUDADeviceContext call context() failed, make sure in the "
        "thread_local semantic."));
  }
  return thread_ctx_.at(this);
}

stream::CUDAStream* CUDADeviceContext::GetCudaStream() const {
  return cuda_stream_.get();
}

stream::CUDAStream* CUDADeviceContext::SetCudaStream(
    stream::CUDAStream* new_stream_ptr) {
  auto* old_stream_ptr = cuda_stream_.release();
  cuda_stream_.reset(new_stream_ptr);
  return old_stream_ptr;
}

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

const Place& CUDAPinnedDeviceContext::GetPlace() const { return place_; }
#endif

#ifdef PADDLE_WITH_MKLDNN
MKLDNNDeviceContext::MKLDNNDeviceContext(CPUPlace place)
    : CPUDeviceContext(place), p_blobmap_() {
  p_blobmap_.reset(new BlobMap());
  p_exec_items_.reset(new ExecShape());
  p_mutex_.reset(new std::mutex());
}

MKLDNNDeviceContextThreadLocals::Body::Body()
    : cur_engine(dnnl::engine::kind::cpu, 0), cur_stream(cur_engine) {
  cur_mkldnn_session_id = kMKLDNNSessionID_Default;
  cur_input_shape_str = "";
  cur_input_shape_cache_capacity = 1;
  cur_paddle_data_layout = paddle::framework::DataLayout::kNCHW;
}

// When Thread finish we clear oneDNN cache
// This is needed when we have one executor used by many threads
// e.g. test_analyzer_detect. Thread ID is not part of caching key
// (for naive executor) so we need to clear cache when one thread finish
// and other is to start inference
// TODO(jczaja): Ideally it would be good to clear only part of cache
// related to thread that is to be terminated
MKLDNNDeviceContextThreadLocals::Body::~Body() {
  auto cpu_place = paddle::platform::CPUPlace();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  platform::MKLDNNDeviceContext* dev_ctx =
      (platform::MKLDNNDeviceContext*)pool.Get(cpu_place);
  dev_ctx->ResetBlobMap(exec_ptr_);
}

void MKLDNNDeviceContextThreadLocals::Body::set_cur_mkldnn_session_id(
    size_t sid) {
  cur_mkldnn_session_id = sid;
}
size_t MKLDNNDeviceContextThreadLocals::Body::get_cur_mkldnn_session_id(void) {
  return cur_mkldnn_session_id;
}

void MKLDNNDeviceContextThreadLocals::Body::set_cur_input_shape_str(
    std::string input_shape_str) {
  cur_input_shape_str = input_shape_str;
}
void MKLDNNDeviceContextThreadLocals::Body::set_cur_input_shape_cache_capacity(
    int input_shape_cache_capacity) {
  cur_input_shape_cache_capacity = input_shape_cache_capacity;
}

void MKLDNNDeviceContextThreadLocals::Body::set_cur_paddle_data_layout(
    framework::DataLayout dl) {
  cur_paddle_data_layout = dl;
}

framework::DataLayout
MKLDNNDeviceContextThreadLocals::Body::get_cur_paddle_data_layout(void) {
  return cur_paddle_data_layout;
}

void MKLDNNDeviceContextThreadLocals::Body::log_lib_version(void) {
  if (!said_once) {
    said_once = true;
    auto dv = dnnl::version();
    LOG(INFO) << "oneDNN v" << dv->major << "." << dv->minor << "."
              << dv->patch;
  }
}

const dnnl::engine& MKLDNNDeviceContextThreadLocals::Body::get_engine(void) {
  return cur_engine;
}

dnnl::stream& MKLDNNDeviceContextThreadLocals::Body::get_stream(void) {
  return cur_stream;
}

void MKLDNNDeviceContext::ResetBlobMap(void* ptr) {
  std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
  if (!block_next_cache_clearing_) {
    VLOG(3) << "Clearing DNNL cache.";
    // If no specific executor pointer then clear
    // everything. For executor pointer then clear only
    // objects allocated when using given executor
    if (ptr == nullptr) {
      p_blobmap_->clear();
    } else {
      // Iterate through all shapes and release
      // for each shape and active executor all entries
      // of this executor
      for (auto& s : *p_exec_items_) {
        for (auto& v : (*s.second)[ptr]) {
          (v.first)->erase(v.second);
        }
        s.second->erase(ptr);
      }
    }
  } else {
    VLOG(3) << "Prevented Clearing DNNL cache.";
    block_next_cache_clearing_ = false;
  }
}

void MKLDNNDeviceContext::RemoveShapeEntriesWithExecutor(void) const {
  p_exec_items_->erase(p_exec_items_->begin());
}

void MKLDNNDeviceContext::LinkEntryWithExecutor(BlobPtr_t<KeyBlob> pblob,
                                                KeyBlob::iterator it) const {
  // Take current input shape from TLS
  // Take current executor addess from TLS
  // and for this executor's items add the one defined with arguments
  auto key_it = p_exec_items_
                    ->insert(std::make_pair(tls().cur_input_shape_str,
                                            std::make_shared<ExecMap>()))
                    .first;
  (*key_it->second)[tls().get_curr_exec()].push_back(std::make_pair(pblob, it));

  VLOG(3) << "LinkEntryWithExecutor, shapes: " << p_exec_items_->size()
          << " curr exec size: "
          << (*key_it->second)[tls().get_curr_exec()].size() << "\n";
}

void MKLDNNDeviceContext::BlockNextCacheClearing() {
  std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
  VLOG(3) << "Next DNNL cache clearing has been blocked.";
  block_next_cache_clearing_ = true;
}

size_t MKLDNNDeviceContext::GetShapeBlobSize() const {
  std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
  BlobMap* pMap = p_blobmap_.get();
  auto map_it = pMap->find(tls().cur_mkldnn_session_id);
  if (map_it == pMap->end()) {
    PADDLE_THROW(platform::errors::NotFound(
        "MKLDNNDeviceContext don't find cur_mkldnn_session_id: %d.",
        tls().cur_mkldnn_session_id));
  }
  return map_it->second->size();
}

void MKLDNNDeviceContext::SetBlob(const std::string& name,
                                  BlobPtr_t<void> data) const {
  BlobMap* pMap = p_blobmap_.get();
  BlobPtr_t<ShapeBlob> sBlob = nullptr;
  BlobPtr_t<KeyBlob> pBlob = nullptr;

  int sid = tls().get_cur_mkldnn_session_id();

  std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);

  // Find ShapeBlob for current mkldnn session id.
  auto map_it = pMap->find(sid);

  if (map_it == pMap->end()) {
    // 1st time to set blob in current thread
    sBlob = std::make_shared<ShapeBlob>();
    (*pMap)[sid] = sBlob;
    VLOG(2) << "SetBlob: sid=" << sid << ", add new sid\n";
  } else {
    sBlob = map_it->second;
  }

  // Find KeyBlob for current input shape
  auto key_it = sBlob->find(tls().cur_input_shape_str);

  if (key_it == sBlob->end()) {
    // In cache clearing mode, cur_input_shape_cache_capacity defines
    // max pblob capacity
    if ((static_cast<size_t>(sid) ==
         MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_CacheClearing) &&
        sBlob->size() &&
        (sBlob->size() >=
         static_cast<size_t>(tls().cur_input_shape_cache_capacity))) {
      VLOG(2) << "sid=" << sid
              << ", remove all blobs of shape: " << sBlob->begin()->first;
      sBlob->erase(sBlob->begin()->first);
      RemoveShapeEntriesWithExecutor();
    }
    pBlob = std::make_shared<KeyBlob>();
    (*sBlob)[tls().cur_input_shape_str] = pBlob;
  } else {
    pBlob = key_it->second;
  }

  // Find Blob via name
  auto blob_it = pBlob->find(name);
  if (blob_it == pBlob->end()) {
    auto el =
        pBlob->insert(std::make_pair(name, data));  //  (*pBlob)[name] = data;
    // Register new element in per executor map
    // to have easily erased when executor terminated
    LinkEntryWithExecutor(pBlob, el.first);
  } else {
    blob_it->second = data;  // set data to existing blob
  }
  VLOG(2) << "SetBlob: sid=" << sid << ", add blob=" << name << "\n";
  // lock will be automatically released when out of scope
  return;
}

unsigned int MKLDNNDeviceContext::GetCachedObjectsNumber(void) const {
  unsigned int num_entries = 0;
  for (auto const& l3 : *p_blobmap_) {
    for (auto const& l2 : *(l3.second)) {
      num_entries += (l2.second)->size();
    }
  }
  return num_entries;
}

MKLDNNDeviceContext::BlobPtr_t<void> MKLDNNDeviceContext::GetBlob(
    const std::string& name) const {
  BlobMap* pMap = p_blobmap_.get();
  BlobPtr_t<ShapeBlob> sBlob = nullptr;
  BlobPtr_t<KeyBlob> pBlob = nullptr;

  int sid = tls().get_cur_mkldnn_session_id();

  std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);

  // Find ShapeBlob for current mkldnn session id firstly
  auto map_it = pMap->find(sid);
  // (jczaja): After first iteration of model's execution we
  // should have all elements cached (mostly) so failures are unlikely (less
  // likely for dynamic shapes)
  if (unlikely(map_it == pMap->end())) {
    VLOG(2) << "GetBlob: sid=" << sid << ", miss sid\n";
    return nullptr;
  }
  sBlob = map_it->second;

  // Find KeyBlob for current input shape secondly
  auto sBlob_it = sBlob->find(tls().cur_input_shape_str);
  if (unlikely(sBlob_it == sBlob->end())) {
    VLOG(2) << "GetBlob: sid=" << tls().cur_input_shape_str
            << ", miss input_shape_str\n";
    return nullptr;
  }
  pBlob = sBlob_it->second;

  // Find Blob via name
  auto key_it = pBlob->find(name);

  if (unlikely(key_it == pBlob->end())) {
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
