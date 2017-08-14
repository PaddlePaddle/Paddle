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

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  template <typename DeviceType>
  DeviceType* get_eigen_device() const;
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext();
  explicit CPUDeviceContext(CPUPlace);
  virtual ~CPUDeviceContext() {}

  Eigen::DefaultDevice* eigen_device() const;

  Place GetPlace() const override;

 private:
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

#ifndef PADDLE_ONLY_CPU
static const cudaStream_t default_stream = cudaStreamDefault;

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  // Use the default stream on the current device
  EigenCudaStreamDevice()
      : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
    cudaGetDevice(&device_);
    Eigen::initializeDeviceProp();
  }
  // Use the default stream on the specified device
  explicit EigenCudaStreamDevice(int device)
      : stream_(&default_stream),
        device_(device),
        scratch_(NULL),
        semaphore_(NULL) {
    Eigen::initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  explicit EigenCudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaGetDeviceCount(&num_devices);
      device_ = device;
    }
    Eigen::initializeDeviceProp();
  }

  virtual ~EigenCudaStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const cudaStream_t& stream() const { return *stream_; }
  void set_stream(const cudaStream_t* cuda_stream) { stream_ = cuda_stream; }
  const cudaDeviceProp& deviceProperties() const {
    return Eigen::m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaSetDevice(device_);
    void* result;
    cudaMalloc(&result, num_bytes);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaSetDevice(device_);
    cudaFree(buffer);
  }

  virtual void* scratchpad() const {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  virtual unsigned int* semaphore() const {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
    }
    return semaphore_;
  }

 private:
  const cudaStream_t* stream_;
  int device_;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(GPUPlace);
  virtual ~CUDADeviceContext();

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const;

  /*! \brief  Return place in the device context. */
  Place GetPlace() const override;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  // clang-format off
  /*! \brief  Return cublas handle in the device context. */
  cublasHandle_t    cublas_handle();

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t     cudnn_handle();

  /*! \brief  Return curand handle in the device context. */
  curandGenerator_t curand_generator();
  // clang-format on

 private:
  GPUPlace place_;

 private:
  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

 private:
  uint64_t seed_;

  // clang-format off
  cudnnHandle_t     cudnn_handle_     = nullptr;
  cublasHandle_t    cublas_handle_    = nullptr;
  curandGenerator_t curand_generator_ = nullptr;
  cudaStream_t      stream_           = nullptr;
  // clang-format on
};

#endif

}  // namespace platform
}  // namespace paddle
