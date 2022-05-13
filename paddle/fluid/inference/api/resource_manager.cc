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

#include "paddle/fluid/inference/api/resource_manager.h"

#include <unordered_map>

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/generator.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace internal {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override {}

  void Reinitialize(gpuStream_t cuda_stream, phi::Allocator* allocator,
                    GPUPlace place) {
    stream_ = cuda_stream;
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
  gpuStream_t stream_;                // not owned;
  phi::Allocator* allocator_;         // not owned;
  const gpuDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void*, phi::Allocator::AllocationPtr> allocations_;
};
#endif
}  // namespace internal

ResourceManager::ResourceManager(const phi::Place& place, void* stream)
    : place_(place) {
  InitCPUResource();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  InitGPUResource(stream);
#endif
}

ResourceManager::~ResourceManager() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  DestroyGPUResource();
#endif
}

void ResourceManager::InitCPUResource() {
  cpu_eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* ResourceManager::GetCpuEigenDevice() {
  return cpu_eigen_device_.get();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void ResourceManager::InitGPUResource(void* stream) {
  if (stream == nullptr) {
    owned_stream_ = true;
    phi::InitStream(&stream_);
  } else {
    owned_stream_ = false;
    stream_ = reinterpret_cast<gpuStream_t>(stream);
  }

  InitGpuProperties();
  InitGpuEigenDevice();
  InitDnnHanlde();
  InitBlasHandle();
  InitBlasLtHandle();
  InitSolverHandle();
  InitSparseHandle();
}

void ResourceManager::DestroyGPUResource() {
  if (owned_stream_) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_));
#endif
    stream_ = nullptr;
  }

  DestroyDnnHandle();
  DestroyBlasHandle();
  DestroyBlasLtHandle();
  DestroySolverHandle();
  DestroySparseHandle();
}

void ResourceManager::InitGpuProperties() {
  phi::backends::gpu::GPUDeviceGuard guard(place_.device);
  phi::InitGpuProperties(place_, &compute_capability_, &runtime_version_,
                         &driver_version_, &multi_process_,
                         &max_threads_per_mp_, &max_threads_per_block_,
                         &max_grid_dim_size_);
}

void ResourceManager::InitGpuEigenDevice() {
  auto* allocator = paddle::memory::allocation::AllocatorFacade::Instance()
                        .GetAllocator(place_)
                        .get();
  eigen_stream_.reset(new internal::EigenGpuStreamDevice());
  eigen_stream_->Reinitialize(stream_, allocator, place_);
  gpu_eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
}

void ResourceManager::InitDnnHanlde() {
  phi::InitDnnHandle(&dnn_handle_, stream_, place_);
}

void ResourceManager::DestroyDnnHandle() { phi::DestroyDnnHandle(dnn_handle_); }

void ResourceManager::InitBlasHandle() {
  phi::InitBlasHandle(&blas_handle_, stream_);
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
  phi::InitBlasHandle(&blas_tensor_core_handle_, stream_);
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
      blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
#endif
#if CUDA_VERSION >= 11000
  phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream_);
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
      blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
#endif
#endif
}

void ResourceManager::DestroyBlasHandle() {
  phi::DestroyBlasHandle(blas_handle_);
  phi::DestroyBlasHandle(blas_tensor_core_handle_);
  phi::DestroyBlasHandle(blas_tf32_tensor_core_handle_);
}

void ResourceManager::InitBlasLtHandle() {
  phi::InitBlasLtHandle(&blaslt_handle_);
}

void ResourceManager::DestroyBlasLtHandle() {
  phi::DestroyBlasLtHandle(blaslt_handle_);
}

void ResourceManager::InitSolverHandle() {
  phi::InitSolverHandle(&solver_handle_, stream_);
}

void ResourceManager::DestroySolverHandle() {
  phi::DestroySolverHandle(solver_handle_);
}

void ResourceManager::InitSparseHandle() {
  phi::InitSparseHandle(&sparse_handle_, stream_);
}

void ResourceManager::DestroySparseHandle() {
  phi::DestroySparseHandle(sparse_handle_);
}

gpuStream_t ResourceManager::GetStream() const { return stream_; }

dnnHandle_t ResourceManager::GetDnnHandle() const { return dnn_handle_; }

blasHandle_t ResourceManager::GetBlasHandle() const { return blas_handle_; }

blasHandle_t ResourceManager::GetBlasTensorCoreHandle() const {
  return blas_tensor_core_handle_;
}

blasHandle_t ResourceManager::GetBlasTF32Handle() const {
  return blas_tf32_tensor_core_handle_;
}

blasLtHandle_t ResourceManager::GetBlasLtHandle() const {
  return blaslt_handle_;
}

phi::solverHandle_t ResourceManager::GetSolverDnHandle() const {
  return solver_handle_;
}

phi::sparseHandle_t ResourceManager::GetSparseHandle() const {
  return sparse_handle_;
}

Eigen::GpuDevice* ResourceManager::GetGpuEigenDevice() const {
  return gpu_eigen_device_.get();
}

int ResourceManager::GetGpuComputeCapability() const {
  return compute_capability_;
}

int ResourceManager::GetGpuRuntimeVersion() const { return runtime_version_; }

int ResourceManager::GetGpuDriverVersion() const { return driver_version_; }

int ResourceManager::GetGPUMultiProcessors() const { return multi_process_; }

int ResourceManager::GetGpuMaxThreadsPerMp() const {
  return max_threads_per_mp_;
}

int ResourceManager::GetGpuMaxThreadsPerBlock() const {
  return max_threads_per_block_;
}

std::array<int, 3> ResourceManager::GetGpuMaxGridDimSize() const {
  return max_grid_dim_size_;
}

#endif
}  // namespace paddle
