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

#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"

#ifndef PADDLE_ONLY_CPU
#include "paddle/platform/dynload/cublas.h"
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/dynload/curand.h"
#include "paddle/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif
#include <memory>
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace platform {

template <typename T>
struct EigenDeviceConverter;

template <>
struct EigenDeviceConverter<platform::CPUPlace> {
  using EigenDeviceType = Eigen::DefaultDevice;
};

#ifndef PADDLE_ONLY_CPU
template <>
struct EigenDeviceConverter<platform::GPUPlace> {
  using EigenDeviceType = Eigen::GpuDevice;
};
#endif

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  template <typename PlaceType,
            typename DeviceType =
                typename EigenDeviceConverter<PlaceType>::EigenDeviceType>
  DeviceType* get_eigen_device() const;
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext() { eigen_device_.reset(new Eigen::DefaultDevice()); }

  Eigen::DefaultDevice* eigen_device() const { return eigen_device_.get(); }

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
    PADDLE_ENFORCE(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
    eigen_stream_.reset(new Eigen::CudaStreamDevice(&stream_));
    eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  }

  Place GetPlace() const override {
    Place retv = GPUPlace();
    return retv;
  }

  void Wait() {
    PADDLE_ENFORCE(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize failed");
  }

  cudaStream_t stream() { return stream_; }

  Eigen::GpuDevice* eigen_device() const { return eigen_device_.get(); }

  cublasHandle_t cublas_handle() {
    if (!blas_handle_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::cublasCreate(&blas_handle_),
                     "cublasCreate failed");
      PADDLE_ENFORCE(
          paddle::platform::dynload::cublasSetStream(blas_handle_, stream_),
          "cublasSetStream failed");
    }
    return blas_handle_;
  }

  cudnnHandle_t cudnn_handle() {
    if (!dnn_handle_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::cudnnCreate(&dnn_handle_),
                     "cudnnCreate failed");
      PADDLE_ENFORCE(
          paddle::platform::dynload::cudnnSetStream(dnn_handle_, stream_),
          "cudnnSetStream failed");
    }
    return dnn_handle_;
  }

  curandGenerator_t curand_generator() {
    if (!rand_generator_) {
      GPUPlaceGuard guard(gpu_place_);
      PADDLE_ENFORCE(paddle::platform::dynload::curandCreateGenerator(
                         &rand_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                     "curandCreateGenerator failed");
      PADDLE_ENFORCE(
          paddle::platform::dynload::curandSetPseudoRandomGeneratorSeed(
              rand_generator_, random_seed_),
          "curandSetPseudoRandomGeneratorSeed failed");
      PADDLE_ENFORCE(
          paddle::platform::dynload::curandSetStream(rand_generator_, stream_),
          "curandSetStream failed");
    }
    return rand_generator_;
  }

  ~CUDADeviceContext() {
    Wait();
    if (blas_handle_) {
      PADDLE_ENFORCE(paddle::platform::dynload::cublasDestroy(blas_handle_),
                     "cublasDestroy failed");
    }

    if (dnn_handle_) {
      PADDLE_ENFORCE(paddle::platform::dynload::cudnnDestroy(dnn_handle_),
                     "cudnnDestroy failed");
    }

    if (rand_generator_) {
      PADDLE_ENFORCE(
          paddle::platform::dynload::curandDestroyGenerator(rand_generator_),
          "curandDestroyGenerator failed");
    }
    eigen_stream_.reset();
    eigen_device_.reset();
    PADDLE_ENFORCE(cudaStreamDestroy(stream_), "cudaStreamDestroy failed");
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
