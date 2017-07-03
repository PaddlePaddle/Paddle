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
      paddle::platform::SetDeviceId(new_place.device);
    }
  }

  ~DeviceGuard() { paddle::platform::SetDeviceId(previous_.device); }

 private:
  GPUPlace previous_;
};

class CudaDeviceContext : public DeviceContext {
 public:
  explicit CDUAContext(const GPUPlace gpu_place) : gpu_place_(gpu_place) {
    DeviceGuard(gpu_place_);
    paddle::platform::throw_on_error(cudaStreamCreate(&stream_),
                                     "cudaStreamCreate failed");
    eigen_stream_ = new Eigen::CudaStreamDevice(&stream_);
    eigen_handle_ = new Eigen::GpuDevice(eigen_stream_);
  }

  void Wait() {
    paddle::platform::throw_on_error(cudaStreamSynchronize(stream_),
                                     "cudaStreamSynchronize failed");
  }

  cudaStream_t stream() { return stream_; }

  Eigen::GpuDevice eigen_handle() { return *eigen_handle_; }

  cublasHandle_t cublas_handle() {
    if (!blas_handle_) {
      DeviceGuard guard(gpu_place_);
      paddle::platform::throw_on_error(cublasCreate(&blas_handle_),
                                       "cublasCreate failed");
      paddle::platform::throw_on_error(cublasSetStream(blas_handle_, stream_),
                                       "cublasSetStream failed");
    }
    return blas_handle_;
  }

  cudnnHandle_t cudnn_handle() {
    if (!dnn_handle_) {
      DeviceGuard guard(gpu_place_);
      paddle::platform::throw_on_error(cudnnCreate(&dnn_handle_),
                                       "cudnnCreate failed");
      paddle::platform::throw_on_error(cudnnSetStream(dnn_handle_, stream_),
                                       "cudnnSetStream failed");
    }
    return dnn_handle_;
  }

  curandGenerator_t curand_handle() {
    if (!rand_handle_) {
      DeviceGuard guard(gpu_place_);
      paddle::platform::throw_on_error(
          curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT),
          "curandCreateGenerator failed");
      paddle::platform::throw_on_error(
          curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_),
          "curandSetPseudoRandomGeneratorSeed failed");
      paddle::platform::throw_on_error(
          curandSetStream(curand_generator_, stream_),
          "curandSetStream failed");
    }
    return rand_handle_;
  }

  ~CUDAContext() {
    Wait();

    if (blas_handle_) {
      paddle::platform::throw_on_error(cublasDestroy(blas_handle_),
                                       "cublasDestroy failed");
    }

    if (dnn_handle_) {
      paddle::platform::throw_on_error(cudnnDestroy(dnn_handle_),
                                       "cudnnDestroy failed");
    }

    if (rand_handle_) {
      paddle::platform::throw_on_error(curandDestroyGenerator(rand_handle_),
                                       "curandDestroyGenerator failed");
    }

    delete eigen_stream_;
    delete eigen_handle_;

    paddle::platform::throw_on_error(cudaStreamDestroy(stream_),
                                     "cudaStreamDestroy failed");
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
