/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "mkldnn.hpp"
#endif

#include <map>
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
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
  CPUPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<platform::CPUPlace> {
  using TYPE = CPUDeviceContext;
};

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice;
class CudnnHolder;

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(CUDAPlace place);
  virtual ~CUDADeviceContext();

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return place in the device context. */
  Place GetPlace() const override;

  /*! \brief  Return compute capability in the device context. */
  int GetComputeCapability() const;

  /*! \brief  Return the max physical thread count in the device context */
  int GetMaxPhysicalThreadCount() const;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Return cublas handle in the device context. */
  cublasHandle_t cublas_handle() const;

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t cudnn_handle() const;

  /*! \brief  Run a cudnn function with the workspace provided by
   * CUDADeviceContext */
  void RunCudnnFuncWithWorkspace(const std::function<void(void*)>& cudnn_func,
                                 size_t workspace_len) const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

  template <typename Callback>
  void RecordEvent(cudaEvent_t ev, Callback callback) {
    std::lock_guard<std::mutex> guard(mtx_);
    callback();
    PADDLE_ENFORCE(cudaEventRecord(ev, stream_));
  }

 private:
  CUDAPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;
  std::unique_ptr<CudnnHolder> cudnn_holder_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;

  int compute_capability;
  int multi_process;
  int max_threads_per_mp;

  std::mutex mtx_;
};

template <>
struct DefaultDeviceContextType<platform::CUDAPlace> {
  using TYPE = CUDADeviceContext;
};

// Currently, CUDAPinnedDeviceContext is only used to data copying.
class CUDAPinnedDeviceContext : public DeviceContext {
 public:
  CUDAPinnedDeviceContext();
  explicit CUDAPinnedDeviceContext(CUDAPinnedPlace place);

  Place GetPlace() const override;

  Eigen::DefaultDevice* eigen_device() const;

 private:
  CUDAPinnedPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

template <>
struct DefaultDeviceContextType<platform::CUDAPinnedPlace> {
  using TYPE = CUDAPinnedDeviceContext;
};
#endif

#ifdef PADDLE_WITH_MKLDNN
class MKLDNNDeviceContext : public CPUDeviceContext {
 public:
  explicit MKLDNNDeviceContext(CPUPlace place);

  /* \brief  Get the active engine */
  const mkldnn::engine& GetEngine() const { return engine_; }

  // Set data to blob (i.e. name/data pair). Create blob if not existing
  void SetBlob(const std::string& name, std::shared_ptr<void> data) const;

  // Find a saved blob. Return nullptr if not found
  std::shared_ptr<void> GetBlob(const std::string& name) const;

 private:
  mkldnn::engine engine_;
  std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<void>>>
      p_blobs_;
};
#endif

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  explicit DeviceContextPool(const std::vector<platform::Place>& places);

  static DeviceContextPool& Instance() {
    PADDLE_ENFORCE_NOT_NULL(pool, "Need to Create DeviceContextPool first!");
    return *pool;
  }

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Init(const std::vector<platform::Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  /*! \brief  Return handle of single device context. */
  platform::DeviceContext* Get(const platform::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  size_t size() const { return device_contexts_.size(); }

 private:
  static DeviceContextPool* pool;
  std::map<Place, std::unique_ptr<DeviceContext>> device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace platform
}  // namespace paddle
