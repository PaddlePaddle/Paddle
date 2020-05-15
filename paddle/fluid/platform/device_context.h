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
#include <utility>
#include <vector>
#include "paddle/fluid/memory/malloc.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_helper.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/dynload/nccl.h"
#endif
#include "paddle/fluid/platform/gpu_info.h"
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "mkldnn.hpp"
#include "paddle/fluid/framework/data_layout.h"
#endif

#include <map>
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/stream/cuda_stream.h"
#endif
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace platform {

class DeviceContext {
 public:
  virtual ~DeviceContext() PADDLE_MAY_THROW {}
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
class CudnnWorkspaceHandle;

class CUDAContext {
 public:
  CUDAContext() = default;
  explicit CUDAContext(
      const CUDAPlace& place,
      const stream::Priority& priority = stream::Priority::kNormal);

  ~CUDAContext();

  const CUDAPlace& Place() const { return place_; }

  const std::unique_ptr<Eigen::GpuDevice>& EigenDevice() const {
    return eigen_device_;
  }

  const std::unique_ptr<EigenCudaStreamDevice>& EigenStream() const {
    return eigen_stream_;
  }

  const std::unique_ptr<stream::CUDAStream>& Stream() const { return stream_; }

  const cudaStream_t& RawStream() { return stream_->raw_stream(); }

  const cudnnHandle_t& CudnnHandle() const { return cudnn_handle_; }

  const cusolverDnHandle_t& CusolverDnHandle() const {
    return cusolver_dn_handle_;
  }

  const std::unique_ptr<CublasHandleHolder>& CublasHandle() const {
    return cublas_handle_;
  }

  const std::unique_ptr<CublasHandleHolder>& CublasTensorCoreHandle() const {
    return cublas_tensor_core_handle_;
  }

  /*! \brief  Call cublas function safely. */
  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    cublas_handle_->Call(std::forward<Callback>(callback));
  }

  /*! \brief  Check whether tensor core is supported */
  bool tensor_core_available() const;

  /*! \brief  Call cublas function with Tensor Core safely. If
      Tensor Core is not available, use DEFAULT_MATH instead. */
  template <typename Callback>
  inline void TensorCoreCublasCallIfAvailable(Callback&& callback) const {
    if (cublas_tensor_core_handle_) {
      cublas_tensor_core_handle_->Call(std::forward<Callback>(callback));
    } else {
      cublas_handle_->Call(std::forward<Callback>(callback));
    }
  }

 private:
  void InitEigenContext();

  void InitCuBlasContext() {
    cublas_handle_.reset(
        new CublasHandleHolder(RawStream(), CUBLAS_DEFAULT_MATH));
    if (TensorCoreAvailable()) {
#if CUDA_VERSION >= 9000
      cublas_tensor_core_handle_.reset(
          new CublasHandleHolder(RawStream(), CUBLAS_TENSOR_OP_MATH));
#endif
    }
  }

  void InitCuDNNContext() {
    if (dynload::HasCUDNN()) {
      auto local_cudnn_version = dynload::cudnnGetVersion() / 100;
      auto compile_cudnn_version = CUDNN_VERSION / 100;
      if (local_cudnn_version < static_cast<size_t>(compile_cudnn_version)) {
        LOG_FIRST_N(WARNING, 1)
            << "WARNING: device: " << place_.device
            << ". The installed Paddle is compiled with CUDNN "
            << compile_cudnn_version / 10 << "." << compile_cudnn_version % 10
            << ", but CUDNN version in your machine is "
            << local_cudnn_version / 10 << "." << local_cudnn_version % 10
            << ", which may cause serious incompatible bug. "
            << "Please recompile or reinstall Paddle with compatible CUDNN "
               "version.";
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreate(&cudnn_handle_));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          dynload::cudnnSetStream(cudnn_handle_, RawStream()));
    } else {
      cudnn_handle_ = nullptr;
    }
  }

  void InitCuSolverContext() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cusolverDnCreate(&cusolver_dn_handle_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cusolverDnSetStream(cusolver_dn_handle_, RawStream()));
  }

  void DestoryCuDNNContext() {
    if (cudnn_handle_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroy(cudnn_handle_));
    }
    cudnn_handle_ = nullptr;
  }

  void DestoryCuBlasContext() {
    cublas_handle_.reset();
    cublas_tensor_core_handle_.reset();
  }

  void DestoryCuSolverContext() {
    if (cusolver_dn_handle_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          dynload::cusolverDnDestroy(cusolver_dn_handle_));
    }
  }

  CUDAPlace place_;
  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;
  std::unique_ptr<stream::CUDAStream> stream_;
  cudnnHandle_t cudnn_handle_;
  std::unique_ptr<CublasHandleHolder> cublas_handle_;
  std::unique_ptr<CublasHandleHolder> cublas_tensor_core_handle_;
  cusolverDnHandle_t cusolver_dn_handle_;
  DISABLE_COPY_AND_ASSIGN(CUDAContext);
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

  /*! \brief  Return the SM count in the device context */
  int GetSMCount() const;

  /*! \brief  Return the Max thread num of block in the device context */
  int GetMaxThreadsPerBlock() const;

  /*! \brief  Return the max grid dim size in the device context */
  dim3 GetCUDAMaxGridDimSize() const;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Call cublas function safely. */
  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    return context()->CublasCall(callback);
  }

  /*! \brief  Check whether tensor core is supported */
  bool tensor_core_available() const;

  /*! \brief  Call cublas function with Tensor Core safely. If
      Tensor Core is not available, use DEFAULT_MATH instead. */
  template <typename Callback>
  inline void TensorCoreCublasCallIfAvailable(Callback&& callback) const {
    return context()->TensorCoreCublasCallIfAvailable(callback);
  }

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

  cusolverDnHandle_t cusolver_dn_handle() const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

#if defined(PADDLE_WITH_NCCL)
  /*! \brief  Return nccl communicators. */
  ncclComm_t nccl_comm() const { return nccl_comm_; }

  /*! \brief  Set nccl communicators. */
  void set_nccl_comm(ncclComm_t comm) { nccl_comm_ = comm; }
#endif

  template <typename Callback>
  void RecordEvent(cudaEvent_t ev, Callback callback) const {
    return context()->Stream()->RecordEvent(ev, callback);
  }

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    return context()->Stream()->AddCallback(callback);
  }

  void WaitStreamCallback() const {
    return context()->Stream()->WaitCallback();
  }

  void ResetDefaultContext(const stream::Priority& priority) {
    default_ctx_.reset(new CUDAContext(place_, priority));
  }

  void ResetThreadContext(const stream::Priority& priority) {
    std::lock_guard<std::mutex> guard(ctx_mtx_);
    thread_ctx_[this].reset(new CUDAContext(place_, priority));
  }

  std::shared_ptr<CUDAContext> context() const {
    if (!thread_ctx_.count(this)) {
      return default_ctx_;
    }
    return thread_ctx_.at(this);
  }

 private:
  CUDAPlace place_;
  std::shared_ptr<CUDAContext> default_ctx_;

  // The thread_local static variable will be released before the
  // global static variable, so avoid using it in dtor.
  static thread_local std::unordered_map<const CUDADeviceContext*,
                                         std::shared_ptr<CUDAContext>>
      thread_ctx_;
  static thread_local std::mutex ctx_mtx_;

  mutable std::mutex cudnn_handle_mtx_;

#if defined(PADDLE_WITH_NCCL)
  // NCCL communicator (single process version) for NCCL collective operations.
  // NCCL collective operations provides fast collectives over multiple GPUs
  // both within and across nodes.
  // But, this collectives is used for collectives over multiple GPUs within
  // nodes.
  ncclComm_t nccl_comm_{nullptr};
#endif

  int compute_capability_;
  int runtime_version_;
  int driver_version_;
  int multi_process_;
  int max_threads_per_mp_;
  int max_threads_per_block_;
  dim3 max_grid_dim_size_;

  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

class CudnnWorkspaceHandle {
 public:
  inline CudnnWorkspaceHandle(const CUDADeviceContext& dev_ctx, std::mutex* mtx)
      : device_context_(dev_ctx), mtx_(mtx) {}

  template <typename Callback>
  inline void RunFunc(Callback&& cudnn_func, size_t required_workspace_bytes) {
    if (required_workspace_bytes > WorkspaceSize()) {
      ReallocWorkspace(required_workspace_bytes);
    }
    VLOG(2) << "Cudnn workspace size at RunFunc: "
            << static_cast<double>(WorkspaceSize()) / (1 << 20) << " MB";
    {
      std::lock_guard<std::mutex> guard(*mtx_);
      cudnn_func(allocation_ ? allocation_->ptr() : nullptr);
    }
  }

  /*! \brief Thread which call RunFuncSync() would release gpu memory after
   *  running the function. Currently this function is only used when cudnn
   *  exhaustive searching and callers have to guarantee that the input function
   *  is host blocking */
  template <typename Callback>
  inline void RunFuncSync(Callback&& cudnn_func,
                          size_t required_workspace_bytes) {
    RunFunc(cudnn_func, required_workspace_bytes);
    ResetWorkspace();
  }

  void ReallocWorkspace(size_t required_workspace_bytes);

  inline void ResetWorkspace() { allocation_ = nullptr; }

  inline size_t WorkspaceSize() {
    if (allocation_ == nullptr) {
      return 0;
    }
    return allocation_->size();
  }

  CudnnWorkspaceHandle(CudnnWorkspaceHandle&&) = default;
  CudnnWorkspaceHandle& operator=(CudnnWorkspaceHandle&&) = delete;

 private:
  memory::allocation::AllocationPtr allocation_;
  const CUDADeviceContext& device_context_;
  std::mutex* mtx_;
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

class MKLDNNDeviceContextThreadLocals {
  // default mkldnn session id

  typedef MKLDNNDeviceContextThreadLocals self;
  struct Body {
    size_t cur_mkldnn_session_id;
    // Current data input shape string.
    // - For fixed-shape, it's a null string in default.
    // - For dynamic-shape, it's user specific.
    std::string cur_input_shape_str;
    // the cache capacity of different input shapes for MKLDNN.
    // Default 1 means fixed input shape, not dynamic shape.
    int cur_input_shape_cache_capacity;
    // Recently registered data_format. This is needed to
    // know for converting MKL-DNN Tensor to non MKL-DNN
    paddle::framework::DataLayout cur_paddle_data_layout;

    Body();
    void set_cur_mkldnn_session_id(size_t sid);
    size_t get_cur_mkldnn_session_id(void);
    void set_cur_input_shape_str(std::string input_shape_str);
    void set_cur_input_shape_cache_capacity(int input_shape_cache_capacity);
    void set_cur_paddle_data_layout(framework::DataLayout dl);
    framework::DataLayout get_cur_paddle_data_layout(void);
  };
  MKLDNNDeviceContextThreadLocals() = default;
  MKLDNNDeviceContextThreadLocals(const MKLDNNDeviceContextThreadLocals& c) =
      delete;

 public:
  // default mkldnn session id
  static constexpr size_t kMKLDNNSessionID_Default = 0;
  // mkldnn session id for cache clearing mode
  static constexpr size_t kMKLDNNSessionID_CacheClearing = -1;
  static Body& fetch() {
    thread_local Body b;
    return b;
  }
};

class MKLDNNDeviceContext : public CPUDeviceContext {
 public:
  template <class T>
  using BlobPtr_t = std::shared_ptr<T>;
  template <class P1, class P2>
  using umap_value_smart_t = std::unordered_map<P1, BlobPtr_t<P2>>;
  template <class T>
  using umap_key_string_t = umap_value_smart_t<std::string, T>;

  // Following three maps are used to cache MKLDNN primitives.
  // There relations are:
  // - BlobMap = Map<cur_thread_id, ShapeBlob>
  // - ShapeBlob = Map<cur_input_shape_str, KeyBlob>
  // - KeyBlob  = Map<blob_name, blob>

  using KeyBlob = umap_key_string_t<void>;
  using ShapeBlob = umap_key_string_t<KeyBlob>;
  using BlobMap = umap_value_smart_t<int, ShapeBlob>;

  explicit MKLDNNDeviceContext(CPUPlace place);

  /* \brief  Get the active engine */
  const mkldnn::engine& GetEngine() const { return engine_; }

  // Remove all entries from the blob map
  void ResetBlobMap() const;

  // Get the ShapeBlob size in cur_mkldnn_session_id.
  size_t GetShapeBlobSize() const;

  // Set data to blob (i.e. name/data pair). Create blob if not existing
  void SetBlob(const std::string& name, std::shared_ptr<void> data) const;

  // Find a saved blob. Return nullptr if not found
  std::shared_ptr<void> GetBlob(const std::string& name) const;

  static auto tls() -> decltype(MKLDNNDeviceContextThreadLocals::fetch()) {
    return MKLDNNDeviceContextThreadLocals::fetch();
  }

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

  static void SetPool(DeviceContextPool* dev_pool) { pool = dev_pool; }

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
