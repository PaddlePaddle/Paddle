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

#include "paddle/phi/backends/gpu/gpu_resources.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
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

#include "unsupported/Eigen/CXX11/Tensor"

// TODO(phi): remove fluid header.
#include "paddle/fluid/platform/enforce.h"

namespace phi {

void InitGpuProperties(Place place,
                       int* compute_capability,
                       int* runtime_version,
                       int* driver_version,
                       int* multi_process,
                       int* max_threads_per_mp,
                       int* max_threads_per_block,
                       std::array<int, 3>* max_grid_dim_size) {
  backends::gpu::GPUDeviceGuard guard(place.GetDeviceId());
  *compute_capability =
      backends::gpu::GetGPUComputeCapability(place.GetDeviceId());
  *multi_process = backends::gpu::GetGPUMultiProcessors(place.GetDeviceId());
  *max_threads_per_mp =
      backends::gpu::GetGPUMaxThreadsPerMultiProcessor(place.GetDeviceId());
  *max_grid_dim_size = backends::gpu::GetGpuMaxGridDimSize(place.GetDeviceId());
  *max_threads_per_block =
      backends::gpu::GetGPUMaxThreadsPerBlock(place.GetDeviceId());
  *driver_version = backends::gpu::GetGPUDriverVersion(place.GetDeviceId());
  *runtime_version = backends::gpu::GetGPURuntimeVersion(place.GetDeviceId());

  // TODO(wilber): glog may be replaced in the future?
  LOG_FIRST_N(WARNING, 1) << "Please NOTE: device: "
                          << static_cast<int>(place.device)
                          << ", GPU Compute Capability: "
                          << *compute_capability / 10 << "."
                          << *compute_capability % 10
                          << ", Driver API Version: " << *driver_version / 1000
                          << "." << (*driver_version % 100) / 10
                          << ", Runtime API Version: "
                          << *runtime_version / 1000 << "."
                          << (*runtime_version % 100) / 10;
#ifdef PADDLE_WITH_HIP
  size_t miopen_major, miopen_minor, miopen_patch;
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch));
  auto cudnn_dso_ver =
      (miopen_major * 1000 + miopen_minor * 10 + miopen_patch) / 10;
  auto compile_miopen_version = MIOPEN_VERSION / 10;
  if (cudnn_dso_ver < static_cast<size_t>(compile_miopen_version)) {
    LOG_FIRST_N(WARNING, 1)
        << "WARNING: device: " << static_cast<int>(place.device)
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
  LOG_FIRST_N(WARNING, 1) << "device: " << static_cast<int>(place.device)
                          << ", cuDNN Version: " << cudnn_dso_ver / 1000 << "."
                          << (cudnn_dso_ver % 1000) / 100 << ".";

  // Check CUDA/CUDNN version compatiblity
  auto local_cuda_version =
      (*driver_version / 1000) * 10 + (*driver_version % 100) / 10;
  auto compile_cuda_version =
      (CUDA_VERSION / 1000) * 10 + (CUDA_VERSION % 100) / 10;
#if defined(__linux__)
  PADDLE_ENFORCE_EQ(
      (local_cuda_version / 10 < compile_cuda_version / 10) &&
          (cudnn_dso_ver / 1000 < CUDNN_VERSION / 1000),
      false,
      phi::errors::InvalidArgument(
          "The installed Paddle is compiled with CUDA%d/cuDNN%d,"
          "but CUDA/cuDNN version in your machine is CUDA%d/cuDNN%d. "
          "which will cause serious incompatible bug. "
          "Please recompile or reinstall Paddle with compatible CUDA/cuDNN "
          "version.",
          compile_cuda_version / 10,
          CUDNN_VERSION / 1000,
          local_cuda_version / 10,
          cudnn_dso_ver / 1000));
#endif
  if (local_cuda_version < compile_cuda_version) {
    LOG_FIRST_N(WARNING, 1)
        << "WARNING: device: " << static_cast<int>(place.device)
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

void InitStream(gpuStream_t* stream) {
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipStreamCreateWithPriority(stream, hipStreamDefault, 0));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamCreateWithPriority(stream, cudaStreamDefault, 0));
#endif
}

void DestoryStream(gpuStream_t stream) {
  if (stream != nullptr) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream));
#endif
  }
  stream = nullptr;
}

void InitBlasHandle(blasHandle_t* blas_handle, gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
  phi::dynload::rocblas_create_handle(blas_handle);
  phi::dynload::rocblas_set_stream(*blas_handle, stream);
#else   // PADDLE_WITH_CUDA
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasCreate(blas_handle));
  PADDLE_RETRY_CUDA_SUCCESS(
      phi::dynload::cublasSetStream(*blas_handle, stream));
#endif  // PADDLE_WITH_HIP
}

void DestroyBlasHandle(blasHandle_t handle) {
#ifdef PADDLE_WITH_HIP
  if (handle != nullptr) {
    phi::dynload::rocblas_destroy_handle(handle);
    handle = nullptr;
  }
#else
  if (handle != nullptr) {
    phi::dynload::cublasDestroy(handle);
    handle = nullptr;
  }
#endif  // PADDLE_WITH_HIP
}

void InitBlasLtHandle(blasLtHandle_t* blaslt_handle) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
  phi::dynload::cublasLtCreate(blaslt_handle);
#endif
}

void DestroyBlasLtHandle(blasLtHandle_t handle) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
  if (handle != nullptr) {
    phi::dynload::cublasLtDestroy(handle);
    handle = nullptr;
  }
#endif
}

void InitDnnHandle(dnnHandle_t* handle, gpuStream_t stream, Place place) {
  if (phi::dynload::HasCUDNN()) {
#ifdef PADDLE_WITH_HIP
    size_t miopen_major, miopen_minor, miopen_patch;
    PADDLE_ENFORCE_GPU_SUCCESS(
        dynload::miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch));
    auto local_miopen_version =
        (miopen_major * 1000 + miopen_minor * 10 + miopen_patch) / 10;
    auto compile_miopen_version = MIOPEN_VERSION / 10;
    if (local_miopen_version < static_cast<size_t>(compile_miopen_version)) {
      LOG_FIRST_N(WARNING, 1)
          << "WARNING: device: " << place.device
          << ". The installed Paddle is compiled with MIOPEN "
          << compile_miopen_version / 100 << "." << compile_miopen_version % 100
          << ", but MIOPEN version in your machine is "
          << local_miopen_version / 100 << "." << local_miopen_version % 100
          << ", which may cause serious incompatible bug. "
          << "Please recompile or reinstall Paddle with compatible MIOPEN "
             "version.";
    }
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenCreate(handle));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenSetStream(*handle, stream));
#else
    auto local_cudnn_version = phi::dynload::cudnnGetVersion() / 100;
    auto compile_cudnn_version = CUDNN_VERSION / 100;
    if (local_cudnn_version < static_cast<size_t>(compile_cudnn_version)) {
      LOG_FIRST_N(WARNING, 1)
          << "WARNING: device: " << place.device
          << ". The installed Paddle is compiled with CUDNN "
          << compile_cudnn_version / 10 << "." << compile_cudnn_version % 10
          << ", but CUDNN version in your machine is "
          << local_cudnn_version / 10 << "." << local_cudnn_version % 10
          << ", which may cause serious incompatible bug. "
          << "Please recompile or reinstall Paddle with compatible CUDNN "
             "version.";
    }
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnCreate(handle));
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnSetStream(*handle, stream));
#endif
  } else {
    *handle = nullptr;
  }
}

void DestroyDnnHandle(dnnHandle_t handle) {
#ifdef PADDLE_WITH_HIP
  if (handle != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDestroy(handle));
    handle = nullptr;
  }
#else
  if (handle != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDestroy(handle));
    handle = nullptr;
  }
#endif  // PADDLE_WITH_HIP
}

void InitSolverHandle(solverHandle_t* handle, gpuStream_t stream) {
#ifndef PADDLE_WITH_HIP
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cusolverDnCreate(handle));
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cusolverDnSetStream(*handle, stream));
#endif
}

void DestroySolverHandle(solverHandle_t solver_handle) {
#ifndef PADDLE_WITH_HIP
  if (solver_handle != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDestroy(solver_handle));
    solver_handle = nullptr;
  }
#endif
}

void InitSparseHandle(sparseHandle_t* handle, gpuStream_t stream) {
// ROCM is not yet supported
#if defined(PADDLE_WITH_CUDA)
// The generic APIs is supported from CUDA10.1
#if CUDA_VERSION >= 11000
  PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseCreate(handle));
  PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseSetStream(*handle, stream));
#endif
#endif
}

void DestroySparseHandle(sparseHandle_t handle) {
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 11000
  if (handle != nullptr) {
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseDestroy(handle));
    handle = nullptr;
  }
#endif
#endif
}

}  // namespace phi
