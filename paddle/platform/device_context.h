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

#include "paddle/framework/enforce.h"
#ifndef PADDLE_ONLY_CPU
#include "paddle/platform/cuda.h"
#include "paddle/platform/dynload/cublas.h"
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/dynload/curand.h"
#define EIGEN_USE_GPU
#endif
#include <paddle/platform/place.h>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

namespace paddle {
namespace platform {

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  template <typename DeviceType>
  DeviceType* get_eigen_device();
};

class CPUDeviceContext : public DeviceContext {
 public:
  Eigen::DefaultDevice* eigen_device() {
    if (!eigen_device_) {
      eigen_device_.reset(new Eigen::DefaultDevice());
    }
    return eigen_device_.get();
  }

  Place GetPlace() const override {
    Place retv = CPUPlace();
    return retv;
  }

 private:
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

#ifndef PADDLE_ONLY_CPU

class GPUPlaceGuard {
 public:
  explicit GPUPlaceGuard(GPUPlace new_place) : previous_(GetCurrentDeviceId()) {
    if (previous_ != new_place) {
      paddle::platform::SetDeviceId(new_place.device);
    }
  }

  ~GPUPlaceGuard() { paddle::platform::SetDeviceId(previous_.device); }

 private:
  GPUPlace previous_;
};

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(const GPUPlace gpu_place) : gpu_place_(gpu_place) {
    GPUPlaceGuard guard(gpu_place_);
    paddle::platform::throw_on_error(cudaStreamCreate(&stream_),
                                     "cudaStreamCreate failed");
    eigen_stream_.reset(new Eigen::CudaStreamDevice(&stream_));
    eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  }

  Place GetPlace() const override {
    Place retv = GPUPlace();
    return retv;
  }

  void Wait() {
    paddle::platform::throw_on_error(cudaStreamSynchronize(stream_),
                                     "cudaStreamSynchronize failed");
  }

  cudaStream_t stream() { return stream_; }

  Eigen::GpuDevice* eigen_device() { return eigen_device_.get(); }

  cublasHandle_t cublas_handle() {
    if (!blas_handle_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::cublasCreate(&blas_handle_) ==
                         CUBLAS_STATUS_SUCCESS,
                     "cublasCreate failed");
      PADDLE_ENFORCE(paddle::platform::dynload::cublasSetStream(
                         blas_handle_, stream_) == CUBLAS_STATUS_SUCCESS,
                     "cublasSetStream failed");
    }
    return blas_handle_;
  }

  cudnnHandle_t cudnn_handle() {
    if (!dnn_handle_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::cudnnCreate(&dnn_handle_) ==
                         CUDNN_STATUS_SUCCESS,
                     "cudnnCreate failed");
      PADDLE_ENFORCE(paddle::platform::dynload::cudnnSetStream(
                         dnn_handle_, stream_) == CUDNN_STATUS_SUCCESS,
                     "cudnnSetStream failed");
    }
    return dnn_handle_;
  }

  curandGenerator_t curand_generator() {
    if (!rand_generator_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::curandCreateGenerator(
                         &rand_generator_, CURAND_RNG_PSEUDO_DEFAULT) ==
                         CURAND_STATUS_SUCCESS,
                     "curandCreateGenerator failed");
      PADDLE_ENFORCE(
          paddle::platform::dynload::curandSetPseudoRandomGeneratorSeed(
              rand_generator_, random_seed_) == CURAND_STATUS_SUCCESS,
          "curandSetPseudoRandomGeneratorSeed failed");
      PADDLE_ENFORCE(paddle::platform::dynload::curandSetStream(
                         rand_generator_, stream_) == CURAND_STATUS_SUCCESS,
                     "curandSetStream failed");
    }
    return rand_generator_;
  }

  ~CUDADeviceContext() {
    Wait();
    if (blas_handle_) {
      PADDLE_ENFORCE(paddle::platform::dynload::cublasDestroy(blas_handle_) ==
                         CUBLAS_STATUS_SUCCESS,
                     "cublasDestroy failed");
    }

    if (dnn_handle_) {
      PADDLE_ENFORCE(paddle::platform::dynload::cudnnDestroy(dnn_handle_) ==
                         CUDNN_STATUS_SUCCESS,
                     "cudnnDestroy failed");
    }

    if (rand_generator_) {
      PADDLE_ENFORCE(paddle::platform::dynload::curandDestroyGenerator(
                         rand_generator_) == CURAND_STATUS_SUCCESS,
                     "curandDestroyGenerator failed");
    }
    eigen_stream_.reset();
    eigen_device_.reset();
    paddle::platform::throw_on_error(cudaStreamDestroy(stream_),
                                     "cudaStreamDestroy failed");
  }

 private:
  GPUPlace gpu_place_;
  cudaStream_t stream_;

  std::unique_ptr<Eigen::CudaStreamDevice> eigen_stream_;
  std::unique_ptr<Eigen::GpuDevice> eigen_device_;

  cublasHandle_t blas_handle_{nullptr};

  cudnnHandle_t dnn_handle_{nullptr};

  int random_seed_;
  curandGenerator_t rand_generator_{nullptr};
};

#endif

}  // namespace platform
}  // namespace paddle
