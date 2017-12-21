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

#include <memory>
#include <unordered_map>

#ifdef PADDLE_WITH_CUDA
#include "paddle/platform/dynload/cublas.h"
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif

#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace platform {

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  virtual void Wait() const {}
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext();
  explicit CPUDeviceContext(CPUPlace place);

  Eigen::DefaultDevice* eigen_device() const;

  Place GetPlace() const override;

 private:
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice;

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(GPUPlace place);
  virtual ~CUDADeviceContext();

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return place in the device context. */
  Place GetPlace() const override;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Return cublas handle in the device context. */
  cublasHandle_t cublas_handle() const;

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t cudnn_handle() const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

 private:
  GPUPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
};

class CUDNNDeviceContext : public CUDADeviceContext {
 public:
  explicit CUDNNDeviceContext(CUDNNPlace place);
  virtual ~CUDNNDeviceContext();

  /*! \brief  Return place in the device context. */
  Place GetPlace() const final;

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t cudnn_handle() const;

 private:
  cudnnHandle_t cudnn_handle_;
  CUDNNPlace place_;
};

#endif

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  explicit DeviceContextPool(const std::vector<platform::Place>& places);

  static DeviceContextPool& Get() {
    PADDLE_ENFORCE_NOT_NULL(pool, "Need to Create DeviceContextPool first!");
    return *pool;
  }

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Create(const std::vector<platform::Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  /*! \brief  Return handle of single device context. */
  const platform::DeviceContext* Borrow(const platform::Place& place);

  /*! \brief  Return handle of multi-device context. */
  std::vector<const platform::DeviceContext*> Borrow(
      const std::vector<platform::Place>& places);

  ~DeviceContextPool() {}

 private:
  static DeviceContextPool* pool;
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const platform::Place& place) const {
      return hash_(place.which());
    }
  };
  std::unordered_multimap<const platform::Place, const platform::DeviceContext*,
                          Hash>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace platform
}  // namespace paddle
