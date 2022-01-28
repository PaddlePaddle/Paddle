/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/pten/backends/gpu/gpu_context.h"
#include <array>
#include <functional>
#include <memory>
#include <mutex>

#include "paddle/pten/api/ext/exception.h"

#include "paddle/pten/backends/gpu/gpu_decls.h"
#include "paddle/pten/backends/gpu/gpu_info.h"
#include "paddle/pten/common/float16.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/allocator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/pten/backends/dynload/cublas.h"
#include "paddle/pten/backends/dynload/cudnn.h"
#include "paddle/pten/backends/dynload/cusolver.h"
#include "paddle/pten/backends/dynload/cusparse.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/pten/backends/dynload/nccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/pten/backends/dynload/miopen.h"
#include "paddle/pten/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include "paddle/pten/backends/dynload/rccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#endif  // PADDLE_WITH_HIP

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

// TODO(pten): remove fluid header.
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/enforce.h"

namespace pten {

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
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
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

}  // namespace internal

struct GPUContext::Impl {
  void Init() {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    InitGpuProperties();
    InitStream();
    InitEigenDevice();
    InitBlasHandle();
    InitDNNHandle();
    InitSolverHandle();
    InitSparseHandle();
  }

  Impl() {}

  explicit Impl(Allocator* allocator, const GPUPlace& place = GPUPlace(0))
      : place_(place), eigen_allocator_(allocator) {
    Init();
  }

  explicit Impl(const GPUContextResource& ctx_res,
                const GPUPlace& place = GPUPlace(0))
      : place_(place), external_res_(ctx_res) {
    res_.compute_capability = external_res_.compute_capability;
    res_.runtime_version = external_res_.runtime_version;
    res_.driver_version = external_res_.driver_version;
    res_.multi_process = external_res_.multi_process;
    res_.max_threads_per_mp = external_res_.max_threads_per_mp;
    res_.max_threads_per_block = external_res_.max_threads_per_block;
    res_.max_grid_dim_size = external_res_.max_grid_dim_size;

    res_.stream = external_res_.stream;
    res_.blas_handle = external_res_.blas_handle;
    res_.blas_tensor_core_handle = external_res_.blas_tensor_core_handle;
    res_.blas_tf32_tensor_core_handle =
        external_res_.blas_tf32_tensor_core_handle;
    res_.dnn_handle = external_res_.dnn_handle;
  }

  ~Impl() {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    DestoryInternalStream();
    DestroyInternalBlasHandle();
    DestroyInternalDnnHandle();
    DestroyInternalSolverHandle();
    DestroyInternalSparseHandle();

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nccl_comm_) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::ncclCommDestroy(nccl_comm_));
    }
#endif
  }

  Place GetPlace() const { return place_; }

  int GetComputeCapability() const { return res_.compute_capability; }

  int GetMaxPhysicalThreadCount() const {
    return res_.multi_process * res_.max_threads_per_mp;
  }

  int GetSMCount() const { return res_.multi_process; }

  int GetMaxThreadsPerBlock() const { return res_.max_threads_per_block; }

  std::array<int, 3> GetCUDAMaxGridDimSize() const {
    return res_.max_grid_dim_size;
  }

  bool IsTensorCoreAvailable() const {
    return res_.blas_tensor_core_handle != nullptr;
  }

  void InitGpuProperties() {
    backends::gpu::GPUDeviceGuard guard(place_.GetDeviceId());
    res_.compute_capability =
        backends::gpu::GetGPUComputeCapability(place_.GetDeviceId());
    res_.multi_process =
        backends::gpu::GetGPUMultiProcessors(place_.GetDeviceId());
    res_.max_threads_per_mp =
        backends::gpu::GetGPUMaxThreadsPerMultiProcessor(place_.GetDeviceId());
    res_.max_grid_dim_size =
        backends::gpu::GetGpuMaxGridDimSize(place_.GetDeviceId());
    res_.max_threads_per_block =
        backends::gpu::GetGPUMaxThreadsPerBlock(place_.GetDeviceId());
    res_.driver_version =
        backends::gpu::GetGPUDriverVersion(place_.GetDeviceId());
    res_.runtime_version =
        backends::gpu::GetGPURuntimeVersion(place_.GetDeviceId());

    // TODO(wilber): glog may be replaced in the future?
    LOG_FIRST_N(WARNING, 1)
        << "Please NOTE: device: " << place_.device
        << ", GPU Compute Capability: " << res_.compute_capability / 10 << "."
        << res_.compute_capability % 10
        << ", Driver API Version: " << res_.driver_version / 1000 << "."
        << (res_.driver_version % 100) / 10
        << ", Runtime API Version: " << res_.runtime_version / 1000 << "."
        << (res_.runtime_version % 100) / 10;

    size_t cudnn_dso_ver = dynload::cudnnGetVersion();
    LOG_FIRST_N(WARNING, 1) << "device: " << place_.device
                            << ", cuDNN Version: " << cudnn_dso_ver / 1000
                            << "." << (cudnn_dso_ver % 1000) / 100 << ".";

    // Check CUDA/CUDNN version compatiblity
    auto local_cuda_version =
        (res_.driver_version / 1000) * 10 + (res_.driver_version % 100) / 10;
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
  }

  void InitStream() {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithPriority(&res_.stream, hipStreamDefault, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithPriority(&res_.stream, cudaStreamDefault, 0));
#endif
  }

  void DestoryInternalStream() {
    if (external_res_.stream == nullptr && res_.stream != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(res_.stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(res_.stream));
#endif
    }
    res_.stream = nullptr;
  }

  gpuStream_t GetStream() const {
    PD_CHECK(res_.stream != nullptr, "the gpu stream is nullptr.");
    return res_.stream;
  }

  void SetStream(gpuStream_t stream) {
    if (stream == nullptr) {
      return;
    }
    DestoryInternalStream();
    external_res_.stream = stream;
    res_.stream = stream;

    // Resource (stream dependent) update
    if (external_res_.eigen_device == nullptr && res_.eigen_device != nullptr) {
      delete res_.eigen_device;
      res_.eigen_device = nullptr;
      InitEigenDevice();
    }

    if (external_res_.blas_handle == nullptr && res_.blas_handle != nullptr) {
      BlasHandleSetStream(res_.blas_handle, stream);
    }
    if (external_res_.blas_tensor_core_handle == nullptr &&
        res_.blas_tensor_core_handle != nullptr) {
      BlasHandleSetStream(res_.blas_tensor_core_handle, stream);
    }
    if (external_res_.blas_tf32_tensor_core_handle == nullptr &&
        res_.blas_tf32_tensor_core_handle != nullptr) {
      BlasHandleSetStream(res_.blas_tf32_tensor_core_handle, stream);
    }

    if (external_res_.dnn_handle == nullptr && res_.dnn_handle != nullptr) {
      DnnHandleSetStream(res_.dnn_handle, stream);
    }

    if (external_res_.solver_handle == nullptr &&
        res_.solver_handle != nullptr) {
      SolverHandleSetStream(res_.solver_handle, stream);
    }

    if (external_res_.sparse_handle == nullptr &&
        res_.sparse_handle != nullptr) {
      SparseHandleSetStream(res_.sparse_handle, stream);
    }
  }

  void InitEigenDevice() {
    PD_CHECK(eigen_allocator_ != nullptr,
             "the allocator for eigen device is nullptr.");
    eigen_stream_.reset(new internal::EigenGpuStreamDevice());
    eigen_stream_->Reinitialize(res_.stream, eigen_allocator_, place_);
    res_.eigen_device = new Eigen::GpuDevice(eigen_stream_.get());
  }

  void SetEigenDevice(Eigen::GpuDevice* device) {
    if (device == nullptr) {
      return;
    }

    if (external_res_.eigen_device == nullptr && res_.eigen_device != nullptr) {
      delete res_.eigen_device;
      res_.eigen_device = nullptr;
    }

    external_res_.eigen_device = device;
    res_.eigen_device = device;
  }

  Eigen::GpuDevice* eigen_device() const {
    PD_CHECK(res_.eigen_device != nullptr, "the gpu eigen_device is nullptr.");
    return res_.eigen_device;
  }

  void InitBlasHandle() {
#ifdef PADDLE_WITH_HIP
    pten::dynload::rocblas_create_handle(&res_.blas_handle);
    pten::dynload::rocblas_set_stream(res_.blas_handle, res_.stream);
#else  // PADDLE_WITH_CUDA
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasCreate(&res_.blas_handle));
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cublasSetStream(res_.blas_handle, res_.stream));
#if CUDA_VERSION >= 9000
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cublasCreate(&res_.blas_tensor_core_handle));
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasSetStream(
        res_.blas_tensor_core_handle, res_.stream));
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasSetMathMode(
        res_.blas_tensor_core_handle, CUBLAS_TENSOR_OP_MATH));
#if CUDA_VERSION >= 11000
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cublasCreate(&res_.blas_tf32_tensor_core_handle));
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasSetStream(
        res_.blas_tf32_tensor_core_handle, res_.stream));
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasSetMathMode(
        res_.blas_tf32_tensor_core_handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif  // CUDA_VERSION >= 11000
#endif  // CUDA_VERSION >= 9000
#endif  // PADDLE_WITH_HIP
  }

  void BlasHandleSetStream(blasHandle_t blas, gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::rocblas_set_stream(blas, stream));
#else   // PADDLE_WITH_CUDA
    PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cublasSetStream(blas, stream));
#endif  // PADDLE_WITH_HIP
  }

  void DestroyInternalBlasHandle() {
#ifdef PADDLE_WITH_HIP
    if (external_res_.blas_handle == nullptr && res_.blas_handle != nullptr) {
      pten::dynload::rocblas_destroy_handle(res_.blas_handle);
      res_.blas_handle = nullptr;
    }
#else
    if (external_res_.blas_handle == nullptr && res_.blas_handle != nullptr) {
      pten::dynload::cublasDestroy(res_.blas_handle);
      res_.blas_handle = nullptr;
    }
    if (external_res_.blas_tensor_core_handle == nullptr &&
        res_.blas_tensor_core_handle != nullptr) {
      pten::dynload::cublasDestroy(res_.blas_tensor_core_handle);
      res_.blas_tensor_core_handle = nullptr;
    }
    if (external_res_.blas_tf32_tensor_core_handle == nullptr &&
        res_.blas_tf32_tensor_core_handle != nullptr) {
      pten::dynload::cublasDestroy(res_.blas_tf32_tensor_core_handle);
      res_.blas_tf32_tensor_core_handle = nullptr;
    }
#endif  // PADDLE_WITH_HIP
  }

  blasHandle_t GetBlasHandle() const {
    PD_CHECK(res_.blas_handle != nullptr, "the gpu blas handle is nullptr.");
    return res_.blas_handle;
  }

  void SetBlasHandle(blasHandle_t blas) {
    if (blas == nullptr) {
      return;
    }

    DestroyInternalBlasHandle();

    external_res_.blas_handle = blas;
    res_.blas_handle = blas;
  }

  void InitDNNHandle() {
    if (pten::dynload::HasCUDNN()) {
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
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenCreate(&res_.dnn_handle));
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::miopenSetStream(res_.dnn_handle, res_.stream));
#else
      auto local_cudnn_version = pten::dynload::cudnnGetVersion() / 100;
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
      PADDLE_RETRY_CUDA_SUCCESS(pten::dynload::cudnnCreate(&res_.dnn_handle));
      PADDLE_RETRY_CUDA_SUCCESS(
          pten::dynload::cudnnSetStream(res_.dnn_handle, res_.stream));
#endif
    } else {
      res_.dnn_handle = nullptr;
    }
  }

  dnnHandle_t GetDnnHandle() {
    PD_CHECK(res_.dnn_handle != nullptr, "the gpu dnn handle is nullptr.");
    return res_.dnn_handle;
  }

  void DestroyInternalDnnHandle() {
#ifdef PADDLE_WITH_HIP
    if (external_res_.dnn_handle == nullptr && res_.dnn_handle != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(pten::dynload::miopenDestroy(res_.dnn_handle));
      res_.dnn_handle = nullptr;
    }
#else
    if (external_res_.dnn_handle == nullptr && res_.dnn_handle != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(pten::dynload::cudnnDestroy(res_.dnn_handle));
      res_.dnn_handle = nullptr;
    }
#endif  // PADDLE_WITH_HIP
  }

  void SetDnnHandle(dnnHandle_t handle) {
    if (handle == nullptr) {
      return;
    }

    DestroyInternalDnnHandle();

    external_res_.dnn_handle = handle;
    res_.dnn_handle = handle;
  }

  void DnnHandleSetStream(dnnHandle_t handle, gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    PADDLE_RETRY_CUDA_SUCCESS(dynload::miopenSetStream(handle, stream));
#else   // PADDLE_WITH_CUDA
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cudnnSetStream(handle, stream));
#endif  // PADDLE_WITH_HIP
  }

  void InitSolverHandle() {
#ifndef PADDLE_WITH_HIP
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cusolverDnCreate(&res_.solver_handle));
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cusolverDnSetStream(res_.solver_handle, res_.stream));
#endif
  }

  void DestroyInternalSolverHandle() {
#ifndef PADDLE_WITH_HIP
    if (external_res_.solver_handle == nullptr &&
        res_.solver_handle != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          pten::dynload::cusolverDnDestroy(res_.solver_handle));
      res_.solver_handle = nullptr;
    }
#endif
  }

  void SolverHandleSetStream(solverHandle_t handle, gpuStream_t stream) {
#ifndef PADDLE_WITH_HIP
    PADDLE_RETRY_CUDA_SUCCESS(
        pten::dynload::cusolverDnSetStream(handle, stream));
#endif
  }

  solverHandle_t GetSolverHandle() const {
    PD_CHECK(res_.solver_handle != nullptr,
             "the gpu solver handle is nullptr.");
    return res_.solver_handle;
  }

  void SetSolverHandle(solverHandle_t handle) {
    if (handle == nullptr) {
      return;
    }

    DestroyInternalSolverHandle();

    external_res_.solver_handle = handle;
    res_.solver_handle = handle;
  }

  void InitSparseHandle() {
// ROCM is not yet supported
#if defined(PADDLE_WITH_CUDA)
// The generic APIs is supported from CUDA10.1
#if CUDA_VERSION >= 10010
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseCreate(&res_.sparse_handle));
    PADDLE_RETRY_CUDA_SUCCESS(
        dynload::cusparseSetStream(res_.sparse_handle, res_.stream));
#endif
#endif
  }

  void DestroyInternalSparseHandle() {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    if (external_res_.sparse_handle == nullptr &&
        res_.sparse_handle != nullptr) {
      PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseDestroy(res_.sparse_handle));
      res_.sparse_handle = nullptr;
    }
#endif
#endif
  }

  void SparseHandleSetStream(sparseHandle_t handle, gpuStream_t stream) {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10010
    PADDLE_RETRY_CUDA_SUCCESS(
        dynload::cusparseSetStream(res_.sparse_handle, res_.stream));
#endif
#endif
  }

  sparseHandle_t GetSparseHandle() const {
    PD_CHECK(res_.sparse_handle != nullptr,
             "the gpu sparse handle is nullptr.");
    return res_.sparse_handle;
  }

  void SetSparseHandle(sparseHandle_t handle) {
    if (handle == nullptr) {
      return;
    }

    DestroyInternalSparseHandle();

    external_res_.sparse_handle = handle;
    res_.sparse_handle = handle;
  }

  void Wait() const {
#ifdef PADDLE_WITH_HIP
    hipError_t e_sync = hipSuccess;
#if !defined(_WIN32)
    e_sync = hipStreamSynchronize(res_.stream);
#else
    while (e_sync = hipStreamQuery(res_.stream)) {
      if (e_sync == hipErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#else   // PADDLE_WITH_HIP
    cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
    e_sync = cudaStreamSynchronize(res_.stream);
#else
    while (e_sync = cudaStreamQuery(res_.stream)) {
      if (e_sync == cudaErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#endif  // PADDLE_WITH_HIP

    PADDLE_ENFORCE_GPU_SUCCESS(e_sync);
  }

  ncclComm_t GetNcclComm() const {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    PD_CHECK(nccl_comm_ != nullptr, "the gpu nccl_comm is nullptr.");
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
    if (res_.blas_tf32_tensor_core_handle != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tf32_mtx_);
      callback(res_.blas_tf32_tensor_core_handle);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(res_.blas_handle);
    }
  }

  inline void TensorCoreCublasCallIfAvailable(
      const std::function<void(blasHandle_t)>& callback) const {
    if (res_.blas_tensor_core_handle != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tensor_core_mtx_);
      callback(res_.blas_tensor_core_handle);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(res_.blas_handle);
    }
  }

  inline void CusparseCall(
      const std::function<void(sparseHandle_t)>& callback) const {
    std::lock_guard<std::mutex> guard(sparse_mtx_);
    callback(res_.sparse_handle);
  }

  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const {
    callback();
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(ev, res_.stream));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ev, res_.stream));
#endif
  }

  GPUPlace place_;
  GPUContextResource res_;
  GPUContextResource external_res_;
  std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_{nullptr};
  Allocator* eigen_allocator_{nullptr};

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
};

GPUContext::GPUContext() : DeviceContext() {
  // TODO(wilber): how to set allocator?
  impl_ = std::make_unique<Impl>(
      const_cast<Allocator*>(&this->GetDeviceAllocator()));
}

GPUContext::GPUContext(const GPUPlace& place) : DeviceContext() {
  impl_ = std::make_unique<Impl>(
      const_cast<Allocator*>(&this->GetDeviceAllocator()), place);
}

GPUContext::GPUContext(const GPUContext& other) : DeviceContext() {
  impl_ = std::make_unique<Impl>();

  impl_->SetStream(other.stream());
  impl_->SetEigenDevice(other.eigen_device());
  impl_->eigen_allocator_ = other.impl_->eigen_allocator_;

  impl_->SetBlasHandle(other.cublas_handle());
  impl_->external_res_.blas_tensor_core_handle =
      other.impl_->external_res_.blas_tensor_core_handle;
  impl_->res_.blas_tensor_core_handle =
      other.impl_->res_.blas_tensor_core_handle;
  impl_->external_res_.blas_tf32_tensor_core_handle =
      other.impl_->external_res_.blas_tf32_tensor_core_handle;
  impl_->res_.blas_tf32_tensor_core_handle =
      other.impl_->res_.blas_tf32_tensor_core_handle;

  impl_->SetDnnHandle(other.cudnn_handle());
  impl_->SetSolverHandle(other.cusolver_dn_handle());
  impl_->SetNcclComm(other.nccl_comm());
}

GPUContext::GPUContext(GPUContext&& other) : DeviceContext() {
  impl_ = std::move(other.impl_);
}

GPUContext::GPUContext(const GPUContextResource& ctx_res,
                       const GPUPlace& place) {
  impl_ = std::make_unique<Impl>(ctx_res, place);
}

GPUContext::~GPUContext() = default;

Place GPUContext::GetPlace() const { return impl_->GetPlace(); }

bool GPUContext::tensor_core_available() const {
  return impl_->IsTensorCoreAvailable();
}

int GPUContext::GetComputeCapability() const {
  return impl_->GetComputeCapability();
}

int GPUContext::GetMaxPhysicalThreadCount() const {
  return impl_->GetMaxPhysicalThreadCount();
}

int GPUContext::GetSMCount() const { return impl_->GetSMCount(); }

int GPUContext::GetMaxThreadsPerBlock() const {
  return impl_->GetMaxThreadsPerBlock();
}

std::array<int, 3> GPUContext::GetCUDAMaxGridDimSize() const {
  return impl_->GetCUDAMaxGridDimSize();
}

Eigen::GpuDevice* GPUContext::eigen_device() const {
  return impl_->eigen_device();
}

void GPUContext::SetEigenDevice(Eigen::GpuDevice* device) {
  impl_->SetEigenDevice(device);
}

gpuStream_t GPUContext::stream() const { return impl_->GetStream(); }

void GPUContext::SetStream(gpuStream_t stream) { impl_->SetStream(stream); }

blasHandle_t GPUContext::cublas_handle() const {
  return impl_->GetBlasHandle();
}

void GPUContext::SetBlasHandle(blasHandle_t blas) {
  impl_->SetBlasHandle(blas);
}

dnnHandle_t GPUContext::cudnn_handle() const { return impl_->GetDnnHandle(); }

solverHandle_t GPUContext::cusolver_dn_handle() const {
  return impl_->GetSolverHandle();
}

void GPUContext::SetSolverHandle(solverHandle_t handle) {
  impl_->SetSolverHandle(handle);
}

sparseHandle_t GPUContext::cusparse_handle() const {
  return impl_->GetSparseHandle();
}

void GPUContext::SetSparseHandle(sparseHandle_t handle) {
  impl_->SetSparseHandle(handle);
}

void GPUContext::Wait() const { impl_->Wait(); }

ncclComm_t GPUContext::nccl_comm() const { return impl_->GetNcclComm(); }

void GPUContext::set_nccl_comm(ncclComm_t comm) { impl_->SetNcclComm(comm); }

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

}  // namespace pten
