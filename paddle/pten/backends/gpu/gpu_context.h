/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <array>
#include <functional>
#include "paddle/pten/backends/gpu/forwards.h"
#include "paddle/pten/backends/gpu/gpu_decls.h"
#include "paddle/pten/backends/gpu/gpu_helper.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/device_context.h"

namespace pten {

class DnnWorkspaceHandle;

class GPUContext : public DeviceContext {
 public:
  GPUContext();

  explicit GPUContext(const GPUPlace& place);

  virtual ~GPUContext();

  /*! \brief  Return place in the device context. */
  const Place& GetPlace() const override;

  /*! \brief  Return gpu stream in the device context. */
  gpuStream_t stream() const;

  /*! \brief  Return cudnn  handle in the device context. */
  dnnHandle_t cudnn_handle() const;

  /*! \brief  Return cublas handle in the device context. */
  blasHandle_t cublas_handle() const;

  /*! \brief  Return cusolver handle in the device context. */
  solverHandle_t cusolver_dn_handle() const;

  /*! \brief  Return cusparse handle in the device context. */
  sparseHandle_t cusparse_handle() const;

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Check whether tensor core is supported */
  bool tensor_core_available() const;

  /*! \brief  Return compute capability in the device context. */
  int GetComputeCapability() const;

  /*! \brief  Return the max physical thread count in the device context */
  int GetMaxPhysicalThreadCount() const;

  /*! \brief  Return the SM count in the device context */
  int GetSMCount() const;

  /*! \brief  Return the Max thread num of block in the device context */
  int GetMaxThreadsPerBlock() const;

  /*! \brief  Return the max grid dim size in the device context */
  std::array<int, 3> GetCUDAMaxGridDimSize() const;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Return a cudnn workspace handle to call multiple cudnn
   *  functions without interrupting by other threads.
   *  Once the first cudnn function is called by the handle, a lock
   *  would be acquired to prevent other threads from accessing the
   *  workspace. Once the handle is destructed, the lock would be released.
   */
  DnnWorkspaceHandle* cudnn_workspace_handle();

 public:
  /*! \brief  Call cublas function safely. */
  void CublasCall(const std::function<void(blasHandle_t)>&) const;

  /*! \brief  Call cublas function with Tensor Core safely. If
      Tensor Core is not available, use DEFAULT_MATH instead. */
  void TensorCoreCublasCallIfAvailable(
      const std::function<void(blasHandle_t)>&) const;

  /*! \brief  Call cusparse function safely. */
  void CusparseCall(const std::function<void(sparseHandle_t)>&) const;

  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const;

 public:
  /*! \brief  Return nccl communicators. */
  ncclComm_t nccl_comm() const;

  /*! \brief  Set nccl communicators. */
  void set_nccl_comm(ncclComm_t comm);

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

 protected:
  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetStream(cudaStream_t);

  void SetEigenDevice(Eigen::GpuDevice*);

  void SetBlasHandle(blasHandle_t);

  void SetDnnHandle(dnnHandle_t);

  void SetSolverHandle(solverHandle_t);

  void SetSparseHandle(sparseHandle_t);

  void SetDnnWorkspaceHandle(DnnWorkspaceHandle*);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pten
