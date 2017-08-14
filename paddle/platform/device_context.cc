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

#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

template <>
Eigen::DefaultDevice* DeviceContext::get_eigen_device<Eigen::DefaultDevice>()
    const {
  return reinterpret_cast<const CPUDeviceContext*>(this)->eigen_device();
}

CPUDeviceContext::CPUDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CPUDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CPUDeviceContext::GetPlace() const { return CPUPlace(); }

#ifndef PADDLE_ONLY_CPU

template <>
Eigen::GpuDevice* DeviceContext::get_eigen_device<Eigen::GpuDevice>() const {
  return reinterpret_cast<const CUDADeviceContext*>(this)->eigen_device();
}

CUDADeviceContext::CUDADeviceContext(GPUPlace place) : place_(place) {
  SetDeviceId(place_.device);
  // TODO(qijun) Pass a created cuda stream to Eigen::CudaStreamDevice directly
  // here will cause segment fault. We must implement a class derived from
  // Eigen::StreamInterface, and reinitialize it with a cuda stream and a gpu id
  // later. Please refer to the implementation of class EigenCudaStreamDevice
  // in TensorFlow.
  //
  // We find that CUDA 7 introduces a new option, the per-thread default stream,
  // that has two effects. Please refer to https://devblogs.nvidia.com/
  // parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  //
  // So, we decide to use default stream and add –default-stream per-thread nvcc
  // flag. Than, two threads with two CUDADeviceContexts will run parallelly.

  cudaStreamCreate(&stream_);
  auto* sd = new EigenCudaStreamDevice(place_.device);
  sd->set_stream(&stream_);
  eigen_stream_.reset(sd);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  if (cublas_handle_) {
    PADDLE_ENFORCE(dynload::cublasDestroy(cublas_handle_));
  }

  if (cudnn_handle_) {
    PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
  }

  if (curand_generator_) {
    PADDLE_ENFORCE(dynload::curandDestroyGenerator(curand_generator_));
  }
  eigen_stream_.reset();
  eigen_device_.reset();
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  PADDLE_ENFORCE(cudaStreamSynchronize(0));
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

cublasHandle_t CUDADeviceContext::cublas_handle() {
  if (!cublas_handle_) {
    SetDeviceId(place_.device);
    PADDLE_ENFORCE(dynload::cublasCreate(&cublas_handle_));
  }
  return cublas_handle_;
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() {
  if (!cudnn_handle_) {
    SetDeviceId(place_.device);
    PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
  }
  return cudnn_handle_;
}

curandGenerator_t CUDADeviceContext::curand_generator() {
  if (!curand_generator_) {
    SetDeviceId(place_.device);
    PADDLE_ENFORCE(dynload::curandCreateGenerator(&curand_generator_,
                                                  CURAND_RNG_PSEUDO_DEFAULT));
    PADDLE_ENFORCE(
        dynload::curandSetPseudoRandomGeneratorSeed(curand_generator_, seed_));
  }
  return curand_generator_;
}

#endif  // PADDLE_ONLY_CPU

}  // namespace platform
}  // namespace paddle
