/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifndef PADDLE_ONLY_CPU
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include "paddle/platform/cuda.h"
#define EIGEN_USE_GPU
#endif

#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace platform {

class DeviceContext {
  virtual ~Context() {}
};

class CpuDeviceContext : public DeviceContext {
  Eigen::DefaultDevice eigen_handle() {
    if (!eigen_handle_) {
      eigen_handle_ = new Eigen::DefaultDevice();
    }
    return *eigen_handle_;
  }

 private:
  Eigen::DefaultDevice* eigen_handle_{nullptr};
};

#ifndef PADDLE_ONLY_CPU
class DeviceGuard {
 public:
  explicit DeviceGuard(GPUPlace new_place) : previous_(GetCurrentDeviceId()) {
    if (previous_ != new_place) {
      cudaError_t err = cudaSetDevice(new_place.device);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
  }

  ~DeviceGuard() {
    cudaError_t err = cudaSetDevice(previous_.device);
    PADDLE_ENFORCE(err == cudaSuccess);
  }

 private:
  GPUPlace previous_;
};

class CudaDeviceContext : public DeviceContext {
 public:
  explicit CDUAContext(const GPUPlace gpu_place) : gpu_place_(gpu_place) {
    DeviceGuard(gpu_place_);
    cudaError_t err = cudaStreamCreate(&stream_);
    PADDLE_ENFORCE(err == cudaSuccess);

    eigen_stream_ = new Eigen::CudaStreamDevice(&stream_);
    eigen_handle_ = new Eigen::GpuDevice(eigen_stream_);
  }

  void Wait() {
    cudaError_t err = cudaStreamSynchronize(stream_);
    PADDLE_ENFORCE(err == cudaSuccess);
  }

  cudaStream_t stream() { return stream_; }

  Eigen::GpuDevice eigen_handle() { return *eigen_handle_; }

  cublasHandle_t cublas_handle() {
    if (!blas_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err = cublasCreate(&blas_handle_);
      PADDLE_ENFORCE(err == CUBLAS_STATUS_SUCCESS);
      cudaError_t err = cublasSetStream(blas_handle_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
    return blas_handle_;
  }

  cudnnHandle_t cudnn_handle() {
    if (!dnn_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err = cudnnCreate(&dnn_handle_);
      PADDLE_ENFORCE(err == CUDNN_STATUS_SUCCESS);
      cudaError_t err = cudnnSetStream(dnn_handle_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
    return dnn_handle_;
  }

  curandGenerator_t curand_handle() {
    if (!rand_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err =
          curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
      cudaError_t err =
          curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
      cudaError_t err = curandSetStream(curand_generator_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
    return rand_handle_;
  }

  ~CUDAContext() {
    Wait();

    if (blas_handle_) {
      cudaError_t err = cublasDestroy(blas_handle_);
      PADDLE_ENFORCE(err == CUBLAS_STATUS_SUCCESS);
    }

    if (dnn_handle_) {
      cudaError_t err = cudnnDestroy(dnn_handle_);
      PADDLE_ENFORCE(err == CUDNN_STATUS_SUCCESS);
    }

    if (rand_handle_) {
      cudaError_t err = curandDestroyGenerator(rand_handle_);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
    }

    delete eigen_stream_;
    delete eigen_handle_;
    cudaError_t err = cudaStreamDestroy(stream_);
    PADDLE_ENFORCE(err == cudaSuccess);
  }

 private:
  GPUPlace gpu_place_;
  cudaStream_t stream_;

  Eigen::CudaStreamDevice* eigen_stream_;
  Eigen::GpuDevice* eigen_handle_;

  cublasHandle_t blas_handle_{nullptr};

  cudnnHandle_t dnn_handle_{nullptr};

  int random_seed_;
  curandGenerator_t rand_handle_{nullptr};
};
#endif
}  // namespace platform
}  // namespace paddle
