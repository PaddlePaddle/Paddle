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

#include "paddle/phi/api/ext/exception.h"

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
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
    InitGpuProperties();
    InitStream();
    InitEigenDevice();
    InitBlasHandle();
    InitBlasLtHandle();
    InitDNNHandle();
    InitSolverHandle();
    InitSparseHandle();
    InitDnnWorkspace();
  }

  void PartialInitWithoutAllocator() {
    owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    InitGpuProperties();
    InitStream();
    InitBlasHandle();
    InitBlasLtHandle();
    InitDNNHandle();
    InitSolverHandle();
    InitSparseHandle();
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
    DestoryInternalWorkspace();
    DestoryInternalEigenDevice();
    DestroyInternalSparseHandle();
    DestroyInternalSolverHandle();
    DestroyInternalDnnHandle();
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nccl_comm_) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::ncclCommDestroy(nccl_comm_));
    }
#endif
    DestroyInternalBlasHandle();
    DestroyInternalBlasLtHandle();
    DestoryInternalStream();
  }

  const Place& GetPlace() const { return place_; }

  bool IsTensorCoreAvailable() const {
    return blas_tensor_core_handle_ != nullptr;
  }

  void InitGpuProperties() {
    backends::gpu::GPUDeviceGuard guard(place_.GetDeviceId());
    compute_capability_ =
        backends::gpu::GetGPUComputeCapability(place_.GetDeviceId());
    multi_process_ = backends::gpu::GetGPUMultiProcessors(place_.GetDeviceId());
    max_threads_per_mp_ =
        backends::gpu::GetGPUMaxThreadsPerMultiProcessor(place_.GetDeviceId());
    max_grid_dim_size_ =
        backends::gpu::GetGpuMaxGridDimSize(place_.GetDeviceId());
    max_threads_per_block_ =
        backends::gpu::GetGPUMaxThreadsPerBlock(place_.GetDeviceId());
    driver_version_ = backends::gpu::GetGPUDriverVersion(place_.GetDeviceId());
    runtime_version_ =
        backends::gpu::GetGPURuntimeVersion(place_.GetDeviceId());

    // TODO(wilber): glog may be replaced in the future?
    LOG_FIRST_N(WARNING, 1)
        << "Please NOTE: device: " << static_cast<int>(place_.device)
        << ", GPU Compute Capability: " << compute_capability_ / 10 << "."
        << compute_capability_ % 10
        << ", Driver API Version: " << driver_version_ / 1000 << "."
        << (driver_version_ % 100) / 10
        << ", Runtime API Version: " << runtime_version_ / 1000 << "."
        << (runtime_version_ % 100) / 10;
#ifdef PADDLE_WITH_HIP
    size_t miopen_major, miopen_minor, miopen_patch;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch));
    auto cudnn_dso_ver =
        (miopen_major * 1000 + miopen_minor * 10 + miopen_patch) / 10;
    auto compile_miopen_version = MIOPEN_VERSION / 10;
    if (cudnn_dso_ver < static_cast<size_t>(compile_miopen_version)) {
      LOG_FIRST_N(WARNING, 1)
          << "WARNING: device: " << static_cast<int>(place_.device)
          << ". The installed Paddle is compiled with MIOPEN "
          << compile_miopen_version / 100 << "." << compile_miopen_version % 100
          << ", but MIOPEN version in your machine is " << cudnn_dso_ver / 100
          << "." << cudnn_dso_ver % 100
          << ", which may cause serious incompatible bug. "
          << "Please recompile or reinstall Paddle with compatible MIOPEN "
             "version.";
    }
#else
    size_t cudnn_dso_ver = dynload::cudnnGetVersion();
    LOG_FIRST_N(WARNING, 1) << "device: " << static_cast<int>(place_.device)
                            << ", cuDNN Version: " << cudnn_dso_ver / 1000
                            << "." << (cudnn_dso_ver % 1000) / 100 << ".";

    // Check CUDA/CUDNN version compatiblity
    auto local_cuda_version =
        (driver_version_ / 1000) * 10 + (driver_version_ % 100) / 10;
    auto compile_cuda_version =
        (CUDA_VERSION / 1000) * 10 + (CUDA_VERSION % 100) / 10;
    if (local_cuda_version < compile_cuda_version) {
      LOG_FIRST_N(WARNING, 1)
          << "WARNING: device: " << static_cast<int>(place_.device)
          << ". The installed Paddle is compiled with CUDA "
          << compile_cuda_version / 10 << "." << compile_cuda_version % 10
          << ", but CUDA runtime version in your machine is "
          << local_cuda_version / 10 << "." << local_cuda_version % 10
          << ", which may cause serious incompatible bug. "
          << "Please recompile or reinstall Paddle with compatible CUDA "
             "version.";
    }
#endif
  }

  void InitDnnWorkspace() {
    PD_CHECK(allocator_ != nullptr,
             "the device allocator for gpu context is nullptr.");
    workspace_ = new DnnWorkspaceHandle(allocator_);
  }

  void DestoryInternalWorkspace() {
    if (owned_ && workspace_ != nullptr) {
      delete workspace_;
      stream_ = nullptr;
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
    return DnnWorkspaceHandle(allocator_);
  }

  void InitStream() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipStreamCreateWithPriority(&stream_, hipStreamDefault, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, 0));
#endif
  }

  void DestoryInternalStream() {
    if (owned_ && stream_ != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_));
#endif
    }
    stream_ = nullptr;
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

  void InitBlasHandle() {
#ifdef PADDLE_WITH_HIP
    phi::dynload::rocblas_create_handle(&blas_handle_);
    phi::dynload::rocblas_set_stream(blas_handle_, stream_);
#else  // PADDLE_WITH_CUDA
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasCreate(&blas_handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cublasSetStream(blas_handle_, stream_));
#if CUDA_VERSION >= 9000
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cublasCreate(&blas_tensor_core_handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cublasSetStream(blas_tensor_core_handle_, stream_));
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
        blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
#if CUDA_VERSION >= 11000
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cublasCreate(&blas_tf32_tensor_core_handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cublasSetStream(blas_tf32_tensor_core_handle_, stream_));
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
        blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
#endif  // CUDA_VERSION >= 11000
#endif  // CUDA_VERSION >= 9000
#endif  // PADDLE_WITH_HIP
  }

  void DestroyInternalBlasHandle() {
#ifdef PADDLE_WITH_HIP
    if (owned_ && blas_handle_ != nullptr) {
      phi::dynload::rocblas_destroy_handle(blas_handle_);
      blas_handle_ = nullptr;
    }
#else
    if (owned_ && blas_handle_ != nullptr) {
      phi::dynload::cublasDestroy(blas_handle_);
      blas_handle_ = nullptr;
    }
    if (owned_ && blas_tensor_core_handle_ != nullptr) {
      phi::dynload::cublasDestroy(blas_tensor_core_handle_);
      blas_tensor_core_handle_ = nullptr;
    }
    if (owned_ && blas_tf32_tensor_core_handle_ != nullptr) {
      phi::dynload::cublasDestroy(blas_tf32_tensor_core_handle_);
      blas_tf32_tensor_core_handle_ = nullptr;
    }
#endif  // PADDLE_WITH_HIP
  }

  blasHandle_t GetBlasHandle() const {
    PD_CHECK(blas_handle_ != nullptr, "the gpu blas handle is nullptr.");
    return blas_handle_;
  }

  void SetBlasHandle(blasHandle_t blas) { blas_handle_ = blas; }

  void InitBlasLtHandle() {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
    phi::dynload::cublasLtCreate(&blaslt_handle_);
#endif
  }

  void DestroyInternalBlasLtHandle() {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
    phi::dynload::cublasLtDestroy(blaslt_handle_);
#endif
  }

  void SetBlasLtHandle(blasLtHandle_t blaslt) { blaslt_handle_ = blaslt; }

  blasLtHandle_t GetBlasLtHandle() const {
    PD_CHECK(blaslt_handle_ != nullptr, "the gpu blasLt handle is nullptr.");
    return blaslt_handle_;
  }

  void InitDNNHandle() {
    if (phi::dynload::HasCUDNN()) {
#ifdef PADDLE_WITH_HIP
      size_t miopen_major, miopen_minor, miopen_patch;
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenGetVersion(
          &miopen_major, &miopen_minor, &miopen_patch));
      auto local_miopen_version =
          (miopen_major * 1000 + miopen_minor * 10 + miopen_patch) / 10;
      auto compile_miopen_version = MIOPEN_VERSION / 10;
      if (local_miopen_version < static_cast<size_t>(compile_miopen_version)) {
        LOG_FIRST_N(WARNING, 1)
            << "WARNING: device: " << place_.device
            << ". The installed Paddle is compiled with MIOPEN "
            << compile_miopen_version / 100 << "."
            << compile_miopen_version % 100
            << ", but MIOPEN version in your machine is "
            << local_miopen_version / 100 << "." << local_miopen_version % 100
            << ", which may cause serious incompatible bug. "
            << "Please recompile or reinstall Paddle with compatible MIOPEN "
               "version.";
      }
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenCreate(&dnn_handle_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::miopenSetStream(dnn_handle_, stream_));
#else
      auto local_cudnn_version = phi::dynload::cudnnGetVersion() / 100;
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
      PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnCreate(&dnn_handle_));
      PADDLE_RETRY_CUDA_SUCCESS(
          phi::dynload::cudnnSetStream(dnn_handle_, stream_));
#endif
    } else {
      dnn_handle_ = nullptr;
    }
  }

  dnnHandle_t GetDnnHandle() {
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

  void InitSolverHandle() {
#ifndef PADDLE_WITH_HIP
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cusolverDnCreate(&solver_handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cusolverDnSetStream(solver_handle_, stream_));
#endif
  }

  void DestroyInternalSolverHandle() {
#ifndef PADDLE_WITH_HIP
    if (owned_ && solver_handle_ != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cusolverDnDestroy(solver_handle_));
      solver_handle_ = nullptr;
    }
#endif
  }

  solverHandle_t GetSolverHandle() const {
    PD_CHECK(solver_handle_ != nullptr, "the gpu solver handle is nullptr.");
    return solver_handle_;
  }

  void SetSolverHandle(solverHandle_t handle) { solver_handle_ = handle; }

  void InitSparseHandle() {
// ROCM is not yet supported
#if defined(PADDLE_WITH_CUDA)
// The generic APIs is supported from CUDA10.1
#if CUDA_VERSION >= 10010
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseCreate(&sparse_handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        dynload::cusparseSetStream(sparse_handle_, stream_));
#endif
#endif
  }

  void DestroyInternalSparseHandle() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (owned_ && sparse_handle_ != nullptr) {
      PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseDestroy(sparse_handle_));
      sparse_handle_ = nullptr;
    }
#endif
#endif
  }

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

  inline void CublasCall(
      const std::function<void(blasHandle_t)>& callback) const {
    if (blas_tf32_tensor_core_handle_ != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tf32_mtx_);
      callback(blas_tf32_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void TensorCoreCublasCallIfAvailable(
      const std::function<void(blasHandle_t)>& callback) const {
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

void GPUContext::SetStream(gpuStream_t stream) { impl_->SetStream(stream); }

void GPUContext::SetEigenDevice(Eigen::GpuDevice* device) {
  impl_->SetEigenDevice(device);
}

void GPUContext::SetBlasHandle(blasHandle_t blas) {
  impl_->SetBlasHandle(blas);
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
