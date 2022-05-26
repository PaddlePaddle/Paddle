/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_context.h"

#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <mutex>

#include "glog/logging.h"

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/dynload/cusparse.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/phi/backends/dynload/nccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#endif  // PADDLE_WITH_HIP

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

// TODO(phi): remove fluid header.
#include "paddle/fluid/platform/enforce.h"

namespace phi {

namespace internal {

class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override {}

  void Reinitialize(gpuStream_t cuda_stream,
                    Allocator* allocator,
                    GPUPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    allocator_ = allocator;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const gpuStream_t& stream() const override { return stream_; }

  const gpuDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = allocator_->Allocate(num_bytes);
    VLOG(4) << "Eigen allocated at " << buf->ptr() << " requested "
            << num_bytes;
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
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream_));
#endif
    }
    return semaphore_;
  }

 private:
  GPUPlace place_;
  gpuStream_t stream_;                // not owned;
  Allocator* allocator_;              // not owned;
  const gpuDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void*, Allocator::AllocationPtr> allocations_;
};

#ifdef PADDLE_WITH_HIP
static void StreamCallbackFunc(gpuStream_t stream,
                               gpuError_t status,
                               void* user_data)
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
    static void CUDART_CB StreamCallbackFunc(void* user_data)
#else
    static void CUDART_CB
    StreamCallbackFunc(cudaStream_t stream, cudaError_t status, void* user_data)
#endif
#endif
{
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()>*>(user_data));
  (*func)();
}

}  // namespace internal

void DnnWorkspaceHandle::RunFuncSync(
    const std::function<void(void*)>& cudnn_func,
    size_t required_workspace_bytes,
    bool use_cached_allocation) {
  bool need_realloc = required_workspace_bytes > WorkspaceSize();
  if (need_realloc && !use_cached_allocation) {
    void* workspace_ptr = nullptr;
    size_t size = ((required_workspace_bytes + 255) >> 8) << 8;
    std::lock_guard<std::mutex> guard(*mtx_);
#ifdef PADDLE_WITH_HIP
    auto status = hipMalloc(&workspace_ptr, size);
#else
    auto status = cudaMalloc(&workspace_ptr, size);
#endif
    if (status == gpuSuccess) {
      cudnn_func(workspace_ptr);
      phi::backends::gpu::GpuStreamSync(stream_);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipFree(workspace_ptr));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
#endif
      return;
    }
  }

  RunFunc(cudnn_func, required_workspace_bytes);
  if (need_realloc) {
    // Release the workspace allocated in this running.
    ResetWorkspace();
  }
}

void DnnWorkspaceHandle::ResetWorkspace() { allocation_ = nullptr; }

void DnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
  if (required_workspace_bytes <= WorkspaceSize()) return;
  // reset allocation first before re-allocate to save memory
  allocation_.reset();
  allocation_ = allocator_->Allocate(required_workspace_bytes);
}

struct GPUContext::Impl {
  void Init() {
    owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    phi::InitGpuProperties(place_,
                           &compute_capability_,
                           &runtime_version_,
                           &driver_version_,
                           &multi_process_,
                           &max_threads_per_mp_,
                           &max_threads_per_block_,
                           &max_grid_dim_size_);
    phi::InitStream(&stream_);
    InitEigenDevice();
    InitDnnWorkspace();
  }

  void PartialInitWithoutAllocator() {
    owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    phi::InitGpuProperties(place_,
                           &compute_capability_,
                           &runtime_version_,
                           &driver_version_,
                           &multi_process_,
                           &max_threads_per_mp_,
                           &max_threads_per_block_,
                           &max_grid_dim_size_);
    phi::InitStream(&stream_);
  }

  void PartialInitWithAllocator() {
    owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    InitEigenDevice();
    InitDnnWorkspace();
  }

  Impl() : place_(GPUPlace()) {}

  explicit Impl(const GPUPlace& place) : place_(place) {}

  ~Impl() {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    if (owned_) {
      DestoryInternalWorkspace();
      DestoryInternalEigenDevice();
      phi::DestroySparseHandle(sparse_handle_);
      phi::DestroySolverHandle(solver_handle_);
      phi::DestroyDnnHandle(dnn_handle_);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      if (nccl_comm_) {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::ncclCommDestroy(nccl_comm_));
      }
#endif
      phi::DestroyBlasHandle(blas_handle_);
      phi::DestroyBlasHandle(blas_tensor_core_handle_);
      phi::DestroyBlasHandle(blas_tf32_tensor_core_handle_);
      phi::DestroyBlasLtHandle(blaslt_handle_);
      phi::DestoryStream(stream_);
    }
  }

  const Place& GetPlace() const { return place_; }

  bool IsTensorCoreAvailable() const {
    return blas_tensor_core_handle_ != nullptr;
  }

  void InitDnnWorkspace() {
    PD_CHECK(allocator_ != nullptr,
             "the device allocator for gpu context is nullptr.");
    workspace_ = new DnnWorkspaceHandle(allocator_, stream_);
  }

  void DestoryInternalWorkspace() {
    if (owned_ && workspace_ != nullptr) {
      delete workspace_;
      workspace_ = nullptr;
    }
  }

  // TODO(wilber): The return type is a pointer, to be modified later.
  // DnnWorkspaceHandle* GetDnnWorkspace() {
  //   PD_CHECK(workspace_ != nullptr, "the gpu cudnn workspace is nullptr.");
  //   return workspace_;
  // }
  DnnWorkspaceHandle GetDnnWorkspace() {
    PD_CHECK(allocator_ != nullptr,
             "the device allocator for gpu context is nullptr.");
    return DnnWorkspaceHandle(allocator_, stream_);
  }

  void SetStream(gpuStream_t stream) { stream_ = stream; }

  gpuStream_t GetStream() const {
    PD_CHECK(stream_ != nullptr, "the gpu stream is nullptr.");
    return stream_;
  }

  void InitEigenDevice() {
    PD_CHECK(allocator_ != nullptr,
             "the allocator for eigen device is nullptr.");
    eigen_stream_.reset(new internal::EigenGpuStreamDevice());
    eigen_stream_->Reinitialize(stream_, allocator_, place_);
    eigen_device_ = new Eigen::GpuDevice(eigen_stream_.get());
  }

  void DestoryInternalEigenDevice() {
    if (owned_ && eigen_device_ != nullptr) {
      delete eigen_device_;
      eigen_device_ = nullptr;
    }
  }

  void SetEigenDevice(Eigen::GpuDevice* device) { eigen_device_ = device; }

  Eigen::GpuDevice* eigen_device() const {
    PD_CHECK(eigen_device_ != nullptr, "the gpu eigen_device is nullptr.");
    return eigen_device_;
  }

  blasHandle_t GetBlasHandle() {
    if (!blas_handle_) phi::InitBlasHandle(&blas_handle_, stream_);
    PD_CHECK(blas_handle_ != nullptr, "the gpu blas handle is nullptr.");
    return blas_handle_;
  }

  void SetBlasHandle(blasHandle_t blas) { blas_handle_ = blas; }

  void SetBlasTensorCoreHandle(blasHandle_t handle) {
    blas_tensor_core_handle_ = handle;
  }

  void SetBlasTF32Handle(blasHandle_t handle) {
    blas_tf32_tensor_core_handle_ = handle;
  }

  void SetBlasLtHandle(blasLtHandle_t blaslt) { blaslt_handle_ = blaslt; }

  blasLtHandle_t GetBlasLtHandle() {
    if (!blaslt_handle_) phi::InitBlasLtHandle(&blaslt_handle_);
    PD_CHECK(blaslt_handle_ != nullptr, "the gpu blasLt handle is nullptr.");
    return blaslt_handle_;
  }

  dnnHandle_t GetDnnHandle() {
    if (!dnn_handle_) phi::InitDnnHandle(&dnn_handle_, stream_, place_);
    PD_CHECK(dnn_handle_ != nullptr, "the gpu dnn handle is nullptr.");
    return dnn_handle_;
  }

  void DestroyInternalDnnHandle() {
#ifdef PADDLE_WITH_HIP
    if (owned_ && dnn_handle_ != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDestroy(dnn_handle_));
      dnn_handle_ = nullptr;
    }
#else
    if (owned_ && dnn_handle_ != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDestroy(dnn_handle_));
      dnn_handle_ = nullptr;
    }
#endif  // PADDLE_WITH_HIP
  }

  void SetDnnHandle(dnnHandle_t handle) { dnn_handle_ = handle; }

  solverHandle_t GetSolverHandle() {
    if (!solver_handle_) InitSolverHandle(&solver_handle_, stream_);
    PD_CHECK(solver_handle_ != nullptr, "the gpu solver handle is nullptr.");
    return solver_handle_;
  }

  void SetSolverHandle(solverHandle_t handle) { solver_handle_ = handle; }

  sparseHandle_t GetSparseHandle() const {
    PD_CHECK(sparse_handle_ != nullptr, "the gpu sparse handle is nullptr.");
    return sparse_handle_;
  }

  void SetSparseHandle(sparseHandle_t handle) { sparse_handle_ = handle; }

  void Wait() const {
#ifdef PADDLE_WITH_HIP
    hipError_t e_sync = hipSuccess;
#if !defined(_WIN32)
    e_sync = hipStreamSynchronize(stream_);
#else
    while (e_sync = hipStreamQuery(stream_)) {
      if (e_sync == hipErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#else   // PADDLE_WITH_HIP
    cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
    e_sync = cudaStreamSynchronize(stream_);
#else
    while (e_sync = cudaStreamQuery(stream_)) {
      if (e_sync == cudaErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#endif  // PADDLE_WITH_HIP

    PADDLE_ENFORCE_GPU_SUCCESS(e_sync);
  }

  void WaitEvent(gpuEvent_t ev) const {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(stream_, ev, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream_, ev, 0));
#endif
  }

  ncclComm_t GetNcclComm() const {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // PD_CHECK(nccl_comm_ != nullptr, "the gpu nccl_comm is nullptr.");
    return nccl_comm_;
#endif
    return nullptr;
  }

  void SetNcclComm(ncclComm_t comm) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    nccl_comm_ = comm;
#endif
  }

  inline void CublasCall(const std::function<void(blasHandle_t)>& callback) {
    if (!blas_handle_) {
      phi::InitBlasHandle(&blas_handle_, stream_);
    }
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
    if (!blas_tensor_core_handle_) {
      phi::InitBlasHandle(&blas_tensor_core_handle_, stream_);
      PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
          blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
    }
#endif
#if CUDA_VERSION >= 11000
    if (!blas_tf32_tensor_core_handle_) {
      phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream_);
      PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
          blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
#endif
#endif
    if (blas_tf32_tensor_core_handle_ != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tf32_mtx_);
      callback(blas_tf32_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void TensorCoreCublasCallIfAvailable(
      const std::function<void(blasHandle_t)>& callback) {
    if (!blas_handle_) phi::InitBlasHandle(&blas_handle_, stream_);
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
    if (!blas_tensor_core_handle_) {
      phi::InitBlasHandle(&blas_tensor_core_handle_, stream_);
      PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
          blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
    }
#endif
#if CUDA_VERSION >= 11000
    if (!blas_tf32_tensor_core_handle_) {
      phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream_);
      PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
          blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
#endif
#endif
    if (blas_tensor_core_handle_ != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tensor_core_mtx_);
      callback(blas_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void CusparseCall(
      const std::function<void(sparseHandle_t)>& callback) const {
    std::lock_guard<std::mutex> guard(sparse_mtx_);
    callback(sparse_handle_);
  }

  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const {
    callback();
    RecordEvent(ev);
  }

  void RecordEvent(gpuEvent_t ev) const {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(ev, stream_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ev, stream_));
#endif
  }

  void AddStreamCallback(const std::function<void()>& callback) const {
    // NOTE(zhiqiu): better use threadpool here, otherwise "std::async" may
    // launch too
    // many threads and result in thread oversubscription.
    auto* callback_func = new std::function<void()>(std::move(callback));
    auto* func = new std::function<void()>([this, callback_func] {
      std::lock_guard<std::mutex> lock(stream_call_back_mtx_);
      VLOG(4) << "Stream callback";
      last_future_ = std::async(std::launch::async, [callback_func]() {
        std::unique_ptr<std::function<void()>> releaser(callback_func);
        (*callback_func)();
      });
    });

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipStreamAddCallback(stream_, internal::StreamCallbackFunc, func, 0));
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaLaunchHostFunc(stream_, internal::StreamCallbackFunc, func));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamAddCallback(stream_, internal::StreamCallbackFunc, func, 0));
#endif
#endif
  }

  void WaitStreamCallback() const {
#if defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_CUDA)
    phi::backends::gpu::GpuStreamSync(stream_);
#endif
    {
      std::lock_guard<std::mutex> lock(stream_call_back_mtx_);
      if (last_future_.valid()) {
        last_future_.wait();
      }
    }
  }

  bool owned_{false};
  Place place_;
  int compute_capability_;
  int runtime_version_;
  int driver_version_;
  int multi_process_;
  int max_threads_per_mp_;
  int max_threads_per_block_;
  std::array<int, 3> max_grid_dim_size_;

  gpuStream_t stream_{nullptr};
  Eigen::GpuDevice* eigen_device_{nullptr};
  blasHandle_t blas_handle_{nullptr};
  blasHandle_t blas_tensor_core_handle_{nullptr};
  blasHandle_t blas_tf32_tensor_core_handle_{nullptr};
  blasLtHandle_t blaslt_handle_{nullptr};
  dnnHandle_t dnn_handle_{nullptr};
  solverHandle_t solver_handle_{nullptr};
  sparseHandle_t sparse_handle_{nullptr};
  DnnWorkspaceHandle* workspace_{nullptr};

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  // NCCL communicator (single process version) for NCCL collective operations.
  // NCCL collective operations provides fast collectives over multiple GPUs
  // both within and across nodes.
  // But, this collectives is used for collectives over multiple GPUs within
  // nodes.

  // NOTE: Distributed communicator, distributed framework manages its
  // resources.
  ncclComm_t nccl_comm_{nullptr};
#endif

  mutable std::mutex blas_mtx_;
  mutable std::mutex blas_tensor_core_mtx_;
  mutable std::mutex blas_tf32_mtx_;
  mutable std::mutex sparse_mtx_;
  mutable std::mutex stream_call_back_mtx_;
  mutable std::future<void> last_future_;

  Allocator* allocator_{nullptr};  // external resource.
  // A internal resouce to initinalize eigen_device.
  std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_{nullptr};
};

GPUContext::GPUContext() : DeviceContext(), impl_(std::make_unique<Impl>()) {}

GPUContext::GPUContext(GPUContext&&) = default;

GPUContext& GPUContext::operator=(GPUContext&&) = default;

GPUContext::GPUContext(const GPUPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {}

GPUContext::~GPUContext() = default;

const Place& GPUContext::GetPlace() const { return impl_->GetPlace(); }

gpuStream_t GPUContext::stream() const { return impl_->GetStream(); }

dnnHandle_t GPUContext::cudnn_handle() const { return impl_->GetDnnHandle(); }

blasHandle_t GPUContext::cublas_handle() const {
  return impl_->GetBlasHandle();
}

blasLtHandle_t GPUContext::cublaslt_handle() const {
  return impl_->GetBlasLtHandle();
}

solverHandle_t GPUContext::cusolver_dn_handle() const {
  return impl_->GetSolverHandle();
}

sparseHandle_t GPUContext::cusparse_handle() const {
  return impl_->GetSparseHandle();
}

void GPUContext::Wait() const { impl_->Wait(); }

void GPUContext::WaitEvent(gpuEvent_t ev) const { impl_->WaitEvent(ev); }

bool GPUContext::tensor_core_available() const {
  return impl_->IsTensorCoreAvailable();
}

int GPUContext::GetComputeCapability() const {
  return impl_->compute_capability_;
}

int GPUContext::GetMaxPhysicalThreadCount() const {
  return impl_->multi_process_ * impl_->max_threads_per_mp_;
}

int GPUContext::GetSMCount() const { return impl_->multi_process_; }

int GPUContext::GetMaxThreadsPerBlock() const {
  return impl_->max_threads_per_block_;
}

std::array<int, 3> GPUContext::GetCUDAMaxGridDimSize() const {
  return impl_->max_grid_dim_size_;
}

Eigen::GpuDevice* GPUContext::eigen_device() const {
  return impl_->eigen_device();
}

DnnWorkspaceHandle GPUContext::cudnn_workspace_handle() const {
  return impl_->GetDnnWorkspace();
}

void GPUContext::CublasCall(
    const std::function<void(blasHandle_t)>& callback) const {
  impl_->CublasCall(callback);
}

void GPUContext::TensorCoreCublasCallIfAvailable(
    const std::function<void(blasHandle_t)>& callback) const {
  impl_->TensorCoreCublasCallIfAvailable(callback);
}

void GPUContext::CusparseCall(
    const std::function<void(sparseHandle_t)>& callback) const {
  impl_->CusparseCall(callback);
}

void GPUContext::RecordEvent(gpuEvent_t ev,
                             const std::function<void()>& callback) const {
  impl_->RecordEvent(ev, callback);
}

void GPUContext::RecordEvent(gpuEvent_t ev) const { impl_->RecordEvent(ev); }

void GPUContext::AddStreamCallback(
    const std::function<void()>& callback) const {
  impl_->AddStreamCallback(callback);
}

void GPUContext::WaitStreamCallback() const { impl_->WaitStreamCallback(); }

ncclComm_t GPUContext::nccl_comm() const { return impl_->GetNcclComm(); }

void GPUContext::set_nccl_comm(ncclComm_t comm) { impl_->SetNcclComm(comm); }

void GPUContext::Init() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());
  impl_->Init();
}

void GPUContext::SetStream(gpuStream_t stream) {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());
  impl_->SetStream(stream);
}

void GPUContext::SetEigenDevice(Eigen::GpuDevice* device) {
  impl_->SetEigenDevice(device);
}

void GPUContext::SetBlasHandle(blasHandle_t blas) {
  impl_->SetBlasHandle(blas);
}

void GPUContext::SetBlasTensorCoreHandle(blasHandle_t handle) {
  impl_->SetBlasTensorCoreHandle(handle);
}

void GPUContext::SetBlasTF32Handle(blasHandle_t handle) {
  impl_->SetBlasTF32Handle(handle);
}

void GPUContext::SetBlasLtHandle(blasLtHandle_t blaslt) {
  impl_->SetBlasLtHandle(blaslt);
}

void GPUContext::SetDnnHandle(dnnHandle_t handle) {
  impl_->SetDnnHandle(handle);
}

void GPUContext::SetSolverHandle(solverHandle_t handle) {
  impl_->SetSolverHandle(handle);
}

void GPUContext::SetSparseHandle(sparseHandle_t handle) {
  impl_->SetSparseHandle(handle);
}

void GPUContext::SetDnnWorkspaceHandle(DnnWorkspaceHandle* handle) {
  impl_->workspace_ = handle;
}

void GPUContext::PartialInitWithoutAllocator() {
  impl_->PartialInitWithoutAllocator();
}

void GPUContext::PartialInitWithAllocator() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());
  impl_->PartialInitWithAllocator();
}

void GPUContext::SetComputeCapability(int val) {
  impl_->compute_capability_ = val;
}

void GPUContext::SetMaxThreadsPerMultiProcessor(int val) {
  impl_->max_threads_per_mp_ = val;
}

void GPUContext::SetMultiProcessors(int val) { impl_->multi_process_ = val; }

void GPUContext::SetMaxThreadsPerBlock(int val) {
  impl_->max_threads_per_block_ = val;
}

void GPUContext::SetMaxGridDimSize(const std::array<int, 3>& val) {
  impl_->max_grid_dim_size_ = val;
}

void GPUContext::SetDriverVersion(int val) { impl_->driver_version_ = val; }

void GPUContext::SetRuntimeVersion(int val) { impl_->runtime_version_ = val; }

}  // namespace phi
