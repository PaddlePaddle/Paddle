/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

// TODO(wilber): Remove after adding WITH_EIGEN macro
#include <memory>
#define PADDLE_WITH_EIGEN 1
#define PADDLE_WITH_CUSOLVER 1

#include <ThreadPool.h>
#include <vector>

#include "cuda_runtime.h"  // NOLINT
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/device_context.h"

#ifdef PADDLE_WITH_EIGEN
#include "unsupported/Eigen/CXX11/Tensor"
namespace Eigen {
struct GpuDevice;
}  // namespace Eigen
#endif

#ifdef PADDLE_WITH_NCCL
#include "nccl.h"  // NOLINT
#endif

#ifdef PADDLE_WITH_CUDNN

#endif

namespace pten {

using CUDAPlace = paddle::platform::CUDAPlace;

class CudnnWorkspaceHandle {
 public:
  explicit CudnnWorkspaceHandle(pten::Allocator* allocator, std::mutex* mtx);

  template <typename Callback>
  inline void RunFunc(Callback&& cudnn_func, size_t required_workspace_bytes);

  /*! \brief Thread which call RunFuncSync() would release gpu memory after
   *  running the function. Currently this function is only used when cudnn
   *  exhaustive searching and callers have to guarantee that the input function
   *  is host blocking */
  template <typename Callback>
  inline void RunFuncSync(Callback&& cudnn_func,
                          size_t required_workspace_bytes);

  void ReallocWorkspace(size_t required_workspace_bytes);

  inline void ResetWorkspace();

  inline size_t WorkspaceSize();

  CudnnWorkspaceHandle(CudnnWorkspaceHandle&&) = default;
  CudnnWorkspaceHandle& operator=(CudnnWorkspaceHandle&&) = delete;

 private:
  pten::Allocator* allocator_;
  pten::Allocation allocation_;
  size_t num_bytes_{0};
  std::mutex* mtx_;
};

class StreamCallbackManager {
 public:
  explicit StreamCallbackManager(const cudaStream_t stream);

  ~StreamCallbackManager() = default;

  void AddCallback(std::function<void()> callback) const;

  void Wait() const;

 private:
  const cudaStream_t stream_;
  mutable ::ThreadPool thread_pool_;
  mutable std::mutex mtx_;
  mutable std::future<void> last_future_;
};
class CUDAContext : public DeviceContext {
 public:
  explicit CUDAContext(CUDAPlace place) : place_(place) {}

  explicit CUDAContext(CUDAPlace place,
                       int device_id,
                       int compute_capability,
                       int driver_version,
                       int runtime_version,
                       int multi_process,
                       int max_threads_per_mp,
                       int max_threads_per_block,
                       int max_grid_dim_x,
                       int max_grid_dim_y,
                       int max_grid_dim_z,
                       bool tensor_core_available)
      : place_(place),
        device_id_(device_id),
        compute_capability_(compute_capability),
        driver_version_(driver_version),
        runtime_version_(runtime_version),
        multi_process_(multi_process),
        max_threads_per_mp_(max_threads_per_mp),
        max_threads_per_block_(max_threads_per_block),
        max_grid_dim_x_(max_grid_dim_x),
        max_grid_dim_y_(max_grid_dim_y),
        max_grid_dim_z_(max_grid_dim_z),
        tensor_core_available_(tensor_core_available) {}

  /*! \brief  Return compute capability in the device context. */
  void SetComputeCapability(int compute_capability) {
    compute_capability_ = compute_capability;
  }
  int GetComputeCapability() const { return compute_capability_; }

  /*! \brief  Return the max physical thread count in the device context */
  int GetMaxPhysicalThreadCount() const {
    return multi_process_ * max_threads_per_mp_;
  }

  /*! \brief  Return the SM count in the device context */
  void SetSMCount(int num) { multi_process_ = num; }
  int GetSMCount() const { return multi_process_; }

  /*! \brief  Return the Max thread num of block in the device context */
  int GetMaxThreadsPerBlock() const { return max_threads_per_block_; }
  void SetMaxThreadsPerBlock(int num) { max_threads_per_block_ = num; }

  /*! \brief  Return the max grid dim size in the device context */
  int GetCUDAMaxGridDimX() const { return max_grid_dim_x_; }
  void SetCUDAMaxGridDimX(int num) { max_grid_dim_x_ = num; }
  int GetCUDAMaxGridDimY() const { return max_grid_dim_y_; }
  void SetCUDAMaxGridDimY(int num) { max_grid_dim_y_ = num; }
  int GetCUDAMaxGridDimZ() const { return max_grid_dim_z_; }
  void SetCUDAMaxGridDimZ(int num) { max_grid_dim_z_ = num; }
  dim3 GetCUDAMaxGridDimSize() const;

  /*! \brief  Check whether tensor core is supported */
  bool tensor_core_available() const { return tensor_core_available_; }
  void SetTensorCoreAvailable(bool x) { tensor_core_available_ = x; }

  /*! \brief  Call cublas function safely. */
  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    if (cublas_tf32_tensor_core_handle_) {
      std::lock_guard<std::mutex> guard(cublas_tf32_tensor_core_handle_mtx_);
      callback(cublas_tf32_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(cublas_handle_mtx_);
      callback(cublas_handle_);
    }
  }

  /*! \brief  Call cublas function with Tensor Core safely. If
      Tensor Core is not available, use DEFAULT_MATH instead. */
  template <typename Callback>
  inline void TensorCoreCublasCallIfAvailable(Callback&& callback) const {
    if (cublas_tensor_core_handle_) {
      std::lock_guard<std::mutex> guard(cublas_tensor_core_handle_mtx_);
      callback(cublas_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(cublas_handle_mtx_);
      callback(cublas_handle_);
    }
  }

  // Streams
  cudaStream_t stream() const noexcept { return stream_; }
  void SetStream(cudaStream_t stream) noexcept { stream_ = stream; }

  cudaStream_t host_to_device_stream() const noexcept {
    return host_to_device_stream_;
  }
  void SetHostToDeviceStream(cudaStream_t stream) noexcept {
    host_to_device_stream_ = stream;
  }

  cudaStream_t device_to_host_stream() const noexcept {
    return device_to_host_stream_;
  }
  void SetDeviceToHostStream(cudaStream_t stream) noexcept {
    device_to_host_stream_ = stream;
  }

  std::vector<cudaStream_t>* device_to_device_streams() const noexcept {
    return device_to_device_streams_;
  }
  void SetDeviceToDeviceStreams(std::vector<cudaStream_t>* streams) {
    device_to_device_streams_ = streams;
  }

  template <typename Callback>
  void RecordEvent(cudaEvent_t ev,
                   cudaStream_t stream,
                   Callback callback) const {
    callback();
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(ev, stream));
  }

  void RecordEvent(cudaEvent_t ev, cudaStream_t stream) const {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(ev, stream));
  }

  template <typename Callback>
  void AddStreamCallback(Callback&& callback);

  void WaitStreamCallback();

  void Wait(cudaStream_t stream) const noexcept;

  Place GetPlace() const noexcept override { return place_; }

  cublasHandle_t cublas_handle() const;
  void SetCublasHandle(cublasHandle_t handle) { cublas_handle_ = handle; }

#ifdef PADDLE_WITH_CUDNN
  void SetCudnnHandle(cudnnHandle_t handle) { cudnn_handle_ = handle; }
  cudnnHandle_t cudnn_handle() const;

  /*! \brief  Return a cudnn workspace handle to call multiple cudnn
   *  functions without interrupting by other threads.
   *  Once the first cudnn function is called by the handle, a lock
   *  would be acquired to prevent other threads from accessing the
   *  workspace. Once the handle is destructed, the lock would be released.
   *  CudnnWorkspaceHandle is an RAII object to implement thread-safe
   *  sequential cudnn function calls. */
  CudnnWorkspaceHandle* cudnn_workspace_handle();
#endif

#ifdef PADDLE_WITH_CUSOLVER
  cusolverDnHandle_t cusolver_dn_handle() const;
  void SetCuSolverDnHandle(cusolverDnHandle_t handle) {
    cusolver_dn_handle_ = handle;
  }
#endif

#ifdef PADDLE_WITH_NCCL
  /*! \brief  Set nccl communicators. */
  void SetNcclComm(ncclComm_t comm) { nccl_comm_ = comm; }
  ncclComm_t nccl_comm() { return nccl_comm_; }
#endif

#ifdef PADDLE_WITH_EIGEN
  Eigen::GpuDevice* eigen_device() const { return eigen_device_; }
  void SetEigenDevice(Eigen::GpuDevice* eigen_device) {
    eigen_device_ = eigen_device;
  }
#endif

 private:
  // Streams
  cudaStream_t stream_{nullptr};
  cudaStream_t host_to_device_stream_{nullptr};
  cudaStream_t device_to_host_stream_{nullptr};
  // TODO(wilber): should be a vector ?
  std::vector<cudaStream_t>* device_to_device_streams_;

  // TODO(wilber): places or device_id_?
  CUDAPlace place_;
  int device_id_;

  // basic info.
  int compute_capability_;
  int driver_version_;
  int runtime_version_;
  int multi_process_;
  int max_threads_per_mp_;
  int max_threads_per_block_;
  int max_grid_dim_x_;
  int max_grid_dim_y_;
  int max_grid_dim_z_;

  bool tensor_core_available_;

  // Handles
  mutable std::mutex cublas_handle_mtx_;
  cublasHandle_t cublas_handle_{nullptr};

  mutable std::mutex cublas_tensor_core_handle_mtx_;
  cublasHandle_t cublas_tensor_core_handle_{nullptr};

  mutable std::mutex cublas_tf32_tensor_core_handle_mtx_;
  cublasHandle_t cublas_tf32_tensor_core_handle_{nullptr};

  std::unique_ptr<StreamCallbackManager> callback_manager_{nullptr};
  std::unique_ptr<CudnnWorkspaceHandle> cudnn_workspace_handle_{nullptr};

#if PADDLE_WITH_CUDNN
  mutable std::mutex cudnn_handle_mtx_;
  cudnnHandle_t cudnn_handle_{nullptr};
#endif

#if PADDLE_WITH_CUSOLVER
  cusolverDnHandle_t cusolver_dn_handle_{nullptr};
#endif

#if PADDLE_WITH_NCCL
  ncclComm_t nccl_comm_{nullptr};
#endif

#ifdef PADDLE_WITH_EIGEN
  Eigen::GpuDevice* eigen_device_{nullptr};
#endif
};

}  // namespace pten
