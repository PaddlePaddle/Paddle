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

#include "paddle/fluid/platform/place.h"

#include "paddle/pten/core/device_context.h"

#include "<miopen/miopen.h>"

namespace pten {

using Place = paddle::platform::Place;
using CUDAPlace = paddle::platform::CUDAPlace;

class ROCMContext : public DeviceContext {
 public:
  explicit ROCMContext(CUDAPlace place) : place_(place) {}

  explicit ROCMContext(CUDAPlace place,
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
        tensor_core_available_(false) {}

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
  void SetTensorCoreAvailable(bool x) { tensor_core_available_ = false; }

  /*! \brief  Call cublas function safely. */
  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(cublas_handle_mtx_);
    callback(cublas_handle_);
  }
  template <typename Callback>
  inline void TensorCoreCublasCallIfAvailable(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(cublas_handle_mtx_);
    callback(cublas_handle_);
  }

  // Streams
  hipStream_t stream() const noexcept { return stream_; }
  void SetStream(hipStream_t stream) noexcept { stream_ = stream; }

  cudaStream_t host_to_device_stream() const noexcept {
    return host_to_device_stream_;
  }
  void SetHostToDeviceStream(hipStream_t stream) noexcept {
    host_to_device_stream_ = stream;
  }

  hipStream_t device_to_host_stream() const noexcept {
    return device_to_host_stream_;
  }
  void SetDeviceToHostStream(hipStream_t stream) noexcept {
    device_to_host_stream_ = stream;
  }

  std::vector<hipStream_t>* device_to_device_streams() const noexcept {
    return device_to_device_streams_;
  }
  void SetDeviceToDeviceStreams(std::vector<hipStream_t>* streams) {
    device_to_device_streams_ = streams;
  }

  Place GetPlace() const noexcept override { return place_; }

  // hipStream_t* stream() noexcept { return stream_; }
  // void SetStream(hipStream_t* stream) noexcept { stream_ = stream; }

 private:
  // Streams
  hipStream_t stream_{nullptr};
  hipStream_t host_to_device_stream_{nullptr};
  hipStream_t device_to_host_stream_{nullptr};
  // TODO(wilber): should be a vector ?
  std::vector<hipStream_t>* device_to_device_streams_;

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

  bool tensor_core_available_{false};

  // Handles
  mutable std::mutex cublas_handle_mtx_;
  cublasHandle_t cublas_handle_{nullptr};

  //     hipStream_t* stream_{nullptr};
  //     hipStream_t* host_to_device_stream_{nullptr};
  //     hipStream_t* device_to_host_stream_{nullptr};
  //     // TODO(wilber): n device stream ?
  //     // hipStream_t* device_to_device_stream_;

  //     Allocator* allocator_{nullptr};

  //     rocblas_handle* blas_handle_;
  // #if PADDLE_WITH_CUDNN
  //     miopenHandle_t* dnn_handle_;
  // #endif
};

using CUDAContext = ROCMContext;

}  // namespace pten
