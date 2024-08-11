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

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "paddle/common/macros.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/cpu/forwards.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/test_macros.h"
#include "unsupported/Eigen/CXX11/Tensor"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#endif

namespace paddle {
namespace internal {
class EigenGpuStreamDevice;
}  // namespace internal

class CPUContextResource {
 public:
  CPUContextResource();
  Eigen::DefaultDevice* GetCPUEigenDevice() const;

 private:
  void InitCPUResource();

 private:
  std::unique_ptr<Eigen::DefaultDevice> cpu_eigen_device_;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class GPUContextResource {
 public:
  explicit GPUContextResource(const phi::Place& place, void* stream);
  TEST_API ~GPUContextResource();
  phi::Place Place() const;

  std::function<phi::dnnHandle_t()> GetDnnHandleCreator();
  std::function<phi::blasHandle_t()> GetBlasHandleCreator();
  std::function<phi::blasHandle_t()> GetBlasTensorCoreHandleCreator();
  std::function<phi::blasHandle_t()> GetBlasTF32TensorCoreHandleCreator();
  std::function<phi::blasLtHandle_t()> GetBlasLtHandleCreator();
  std::function<phi::solverHandle_t()> GetSolverDnHandleCreator();
  std::function<phi::sparseHandle_t()> GetSparseHandleCreator();
  std::function<Eigen::GpuDevice*()> GetGpuEigenDeviceCreator();

  gpuStream_t GetStream() const;
  dnnHandle_t GetDnnHandle() const;
  blasHandle_t GetBlasHandle() const;
  blasHandle_t GetBlasTensorCoreHandle() const;
  blasHandle_t GetBlasTF32Handle() const;
  blasLtHandle_t GetBlasLtHandle() const;
  phi::solverHandle_t GetSolverDnHandle() const;
  phi::sparseHandle_t GetSparseHandle() const;
  Eigen::GpuDevice* GetGpuEigenDevice() const;
  int GetGpuComputeCapability() const;
  int GetGpuRuntimeVersion() const;
  int GetGpuDriverVersion() const;
  int GetGPUMultiProcessors() const;
  int GetGpuMaxThreadsPerMp() const;
  int GetGpuMaxThreadsPerBlock() const;
  std::array<unsigned int, 3> GetGpuMaxGridDimSize() const;

 private:
  void InitGPUResource(void* stream);
  void DestroyGPUResource();
  void InitGpuProperties();
  void InitGpuEigenDevice();
  void InitDnnHandle();
  void DestroyDnnHandle();
  void DestroyBlasHandle();
  void InitBlasLtHandle();
  void DestroyBlasLtHandle();
  void InitSolverHandle();
  void DestroySolverHandle();
  void InitSparseHandle();
  void DestroySparseHandle();

 private:
  phi::Place place_;

  int compute_capability_;
  int runtime_version_;
  int driver_version_;
  int multi_process_;
  int max_threads_per_mp_;
  int max_threads_per_block_;
  std::array<unsigned int, 3> max_grid_dim_size_;

  bool owned_stream_{true};
  gpuStream_t stream_;
  std::unique_ptr<Eigen::GpuDevice> gpu_eigen_device_;
  std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_;

  blasHandle_t blas_handle_{nullptr};
  blasHandle_t blas_tensor_core_handle_{nullptr};
  blasHandle_t blas_tf32_tensor_core_handle_{nullptr};
  blasLtHandle_t blaslt_handle_{nullptr};
  dnnHandle_t dnn_handle_{nullptr};
  phi::solverHandle_t solver_handle_{nullptr};
  phi::sparseHandle_t sparse_handle_{nullptr};
  // DnnWorkspaceHandle
};
#endif

class ResourceManager {
 public:
  ResourceManager() = default;
  TEST_API static ResourceManager& Instance();

  // CPU Resource
 public:
  void InitCPUResource();
  CPUContextResource* GetCPUResource() const;

 private:
  std::mutex cpu_mutex_;
  std::unique_ptr<CPUContextResource> cpu_resource_{nullptr};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // GPU Resource
 public:
  void* InitGPUResource(const phi::Place& place, void* stream);
  void DestroyGPUResource(void* stream);
  TEST_API GPUContextResource* GetGPUResource(void* stream) const;
  TEST_API int RefCount(void* stream) const;
  void GpuResourceSwitchStream(void* old_stream, void* new_stream);

 private:
  void Decrease(void* stream);
  void Increase(void* stream);

 private:
  std::mutex gpu_mutex_;
  // a stream corresponding to a series of resource.
  std::map<void* /*stream*/, std::atomic<int>> ref_count_;
  std::map<void* /*stream*/, std::unique_ptr<GPUContextResource>>
      gpu_resources_;
#endif

 private:
  DISABLE_COPY_AND_ASSIGN(ResourceManager);
};

}  // namespace paddle
