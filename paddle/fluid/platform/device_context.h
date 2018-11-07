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

#include <future>  // NOLINT
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
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/stream_callback_manager.h"
#endif
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
class CudnnHolder {
 public:
  CudnnHolder(const cudaStream_t* stream, const CUDAPlace& place);
  ~CudnnHolder();
  cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }

 private:
  friend class CudnnWorkspaceHandle;
  void ReallocateWorkspace(size_t required_workspace_len);

  template <typename Callback>
  void RunFuncImpl(Callback&& cudnn_func, size_t required_workspace_len) {
    if (required_workspace_len > workspace_len_) {
      ReallocateWorkspace(required_workspace_len);
    }
    cudnn_func(workspace_);
  }

  std::mutex& Mutex() { return mtx_; }

  cudnnHandle_t cudnn_handle_;
  void* workspace_;
  size_t workspace_len_;

  const cudaStream_t* stream_;  // not owned;
  const CUDAPlace place_;

  std::mutex mtx_;
};

class CudnnWorkspaceHandle {
 public:
  /*! \brief The lock would not be acquired when constructor calls.
   *  The lock would be acquired when RunFunc() is called first time. */
  inline explicit CudnnWorkspaceHandle(CudnnHolder* holder) : holder_(holder) {}

  /*! \brief Thread which call RunFunc() would acquire the lock first
   *  before invoking cudnn functions. */
  template <typename Callback>
  inline void RunFunc(Callback&& cudnn_func, size_t required_workspace_len) {
    if (!guard_) {
      guard_.reset(new std::lock_guard<std::mutex>(holder_->Mutex()));
    }
    holder_->RunFuncImpl(std::forward<Callback>(cudnn_func),
                         required_workspace_len);
  }

  CudnnWorkspaceHandle(CudnnWorkspaceHandle&&) = default;
  CudnnWorkspaceHandle& operator=(CudnnWorkspaceHandle&&) = delete;

 private:
  CudnnHolder* holder_;  // not own
  std::unique_ptr<std::lock_guard<std::mutex>> guard_;
};

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

  /*! \brief  Return a cudnn workspace handle to call multiple cudnn
   *  functions without interrupting by other threads.
   *  Once the first cudnn function is called by the handle, a lock
   *  would be acquired to prevent other threads from accessing the
   *  workspace. Once the handle is destructed, the lock would be released.
   *  CudnnWorkspaceHandle is an RAII object to implement thread-safe
   *  sequential cudnn function calls. */
  CudnnWorkspaceHandle cudnn_workspace_handle() const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

  template <typename Callback>
  void RecordEvent(cudaEvent_t ev, Callback callback) {
    std::lock_guard<std::mutex> guard(mtx_);
    callback();
    PADDLE_ENFORCE(cudaEventRecord(ev, stream_));
  }

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(callback_mtx_);
    callback_manager_->AddCallback(callback);
  }

  void WaitStreamCallback() const {
    std::lock_guard<std::mutex> guard(callback_mtx_);
    callback_manager_->Wait();
  }

 private:
  CUDAPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;
  std::unique_ptr<CudnnHolder> cudnn_holder_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;

  int compute_capability_;
  int runtime_version_;
  int driver_version_;
  int multi_process_;
  int max_threads_per_mp_;

  mutable std::mutex mtx_;

  // This lock is only used by callback
  // If we use mtx_ for StreamCallbackManager, deadlock may occur sometimes
  mutable std::mutex callback_mtx_;
  std::unique_ptr<StreamCallbackManager> callback_manager_;
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
using KeyBlob = std::unordered_map<std::string, std::shared_ptr<void>>;
using BlobMap = std::unordered_map<int, std::shared_ptr<KeyBlob>>;

void set_cur_thread_id(int);
int get_cur_thread_id(void);

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
  std::shared_ptr<BlobMap> p_blobmap_;
  std::shared_ptr<std::mutex> p_mutex_;
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
  std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace platform
}  // namespace paddle
