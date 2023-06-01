/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU_KP)

#include <array>
#include <functional>
#include <mutex>

#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

class CUDAStream;

class DnnWorkspaceHandle {
 public:
  inline DnnWorkspaceHandle(Allocator* allocator, gpuStream_t stream)
      : allocator_(allocator), stream_(stream) {
    mtx_.reset(new std::mutex());
  }

  inline void RunFunc(const std::function<void(void*)>& cudnn_func,
                      size_t required_workspace_bytes) {
    if (required_workspace_bytes > WorkspaceSize()) {
      ReallocWorkspace(required_workspace_bytes);
    }
    {
      std::lock_guard<std::mutex> guard(*mtx_);
      cudnn_func(allocation_ ? allocation_->ptr() : nullptr);
    }
  }

  /*! \brief Thread which call RunFuncSync() would release gpu memory after
   *  running the function. Currently this function is only used when cudnn
   *  exhaustive searching and callers have to guarantee that the input function
   *  is host blocking */
  void RunFuncSync(const std::function<void(void*)>& cudnn_func,
                   size_t required_workspace_bytes,
                   bool use_cached_allocation = true);

  inline size_t WorkspaceSize() {
    if (allocation_ == nullptr) {
      return 0;
    }
    return allocation_->size();
  }

  void ResetWorkspace();

  void ReallocWorkspace(size_t required_workspace_bytes);

  DnnWorkspaceHandle(DnnWorkspaceHandle&&) = default;
  DnnWorkspaceHandle& operator=(DnnWorkspaceHandle&&) = delete;

 private:
  Allocator::AllocationPtr allocation_{nullptr};
  Allocator* allocator_{nullptr};  // Not owned
  gpuStream_t stream_{nullptr};    // Not owned
  std::unique_ptr<std::mutex> mtx_;
};

class PADDLE_API GPUContext : public DeviceContext,
                              public TypeInfoTraits<DeviceContext, GPUContext> {
 public:
  explicit GPUContext(const GPUPlace& place,
                      bool init = true,
                      int stream_priority = 0);

  GPUContext(GPUContext&&);
  GPUContext& operator=(GPUContext&&);

  virtual ~GPUContext();

  /*! \brief  Return place in the device context. */
  const Place& GetPlace() const override;

  /*! \brief  Return gpu stream in the device context. */
  gpuStream_t stream() const;

  /*! \brief  Return CUDAStream in the device context. */
  CUDAStream* cuda_stream() const;

  /*! \brief  Return cudnn  handle in the device context. */
  dnnHandle_t cudnn_handle() const;

  /*! \brief  Return cublas handle in the device context. */
  blasHandle_t cublas_handle() const;

  /*! \brief  Return cublasLt handle in the device context. */
  blasLtHandle_t cublaslt_handle() const;

  /*! \brief  Return cusolver handle in the device context. */
  solverHandle_t cusolver_dn_handle() const;

  /*! \brief  Return cusparse handle in the device context. */
  sparseHandle_t cusparse_handle() const;

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Wait for event in the stream. */
  void WaitEvent(gpuEvent_t ev) const;

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
  // TODO(wilber): The return type is a pointer, to be modified later.
  DnnWorkspaceHandle cudnn_workspace_handle() const;

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

  void RecordEvent(gpuEvent_t ev) const;

  void AddStreamCallback(const std::function<void()>& callback) const;

  void WaitStreamCallback() const;

  // Several methods for adapting Dnn-specific attributes
  bool HasDnnAttr(const std::string& attr_name) const;
  const Attribute& GetDnnAttr(const std::string& attr_name) const;
  void SetDnnAttr(const std::string& attr_name, Attribute attr);
  void ClearDnnAttr();

  static const char* name() { return "GPUContext"; }

 public:
  /*! \brief  Return nccl communicators. */
  ncclComm_t nccl_comm() const;

  /*! \brief  Set nccl communicators. */
  void set_nccl_comm(ncclComm_t comm);

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  // Note that you must set the Allocator before calling Init function.
  void Init();

  // TODO(wilber): Why does the GetAllocator interface require a stream
  // parameter?
  // The temporary trick method bypasses this problem, and the following
  // interfaces
  // need to be deleted later.

  // Note that this is a trick implementation, which can be used to partially
  // initialize when the SetAllocator interface is not called.
  void PartialInitWithoutAllocator(int stream_priority = 0);
  // Note that this is a trick implementation that can be used to initialize
  // resources that require an Allocator when the SetAllocator interface is
  // called.
  void PartialInitWithAllocator();

  // Note that this function is a trick implementation since all 'set' methods
  // are protected by default.
  // clear: whether clear the original CUDAStream or not
  void SetCUDAStream(CUDAStream*, bool clear = true);

 protected:
  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetStream(gpuStream_t);

  void SetEigenDevice(Eigen::GpuDevice*);
  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&&);

  void SetBlasHandle(blasHandle_t);
  void SetBlasHandle(std::function<blasHandle_t()>&&);

  void SetBlasTensorCoreHandle(blasHandle_t);
  void SetBlasTensorCoreHandle(std::function<blasHandle_t()>&&);

  void SetBlasTF32Handle(blasHandle_t);
  void SetBlasTF32Handle(std::function<blasHandle_t()>&&);

  void SetBlasLtHandle(blasLtHandle_t);
  void SetBlasLtHandle(std::function<blasLtHandle_t()>&&);

  void SetDnnHandle(dnnHandle_t);
  void SetDnnHandle(std::function<dnnHandle_t()>&&);

  void SetSolverHandle(solverHandle_t);
  void SetSolverHandle(std::function<solverHandle_t()>&&);

  void SetSparseHandle(sparseHandle_t);
  void SetSparseHandle(std::function<sparseHandle_t()>&&);

  void SetDnnWorkspaceHandle(DnnWorkspaceHandle*);

  void SetComputeCapability(int val);

  void SetMaxThreadsPerMultiProcessor(int val);

  void SetMultiProcessors(int val);

  void SetMaxThreadsPerBlock(int val);

  void SetMaxGridDimSize(const std::array<int, 3>& val);

  void SetDriverVersion(int val);

  void SetRuntimeVersion(int val);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Note: In order to register the kernel of CUDNN, DnnContext is required.
// Currently, CUDNN kernel directly uses GPUContext. But if the kernel function
// has the same name, this will lead to duplicate instantiations of GPU kernel
// and Dnn kernel function, so if we using DnnContext = GPUContext, we
// must use different function name for cudnn kernel
using GPUDNNContext = GPUContext;

// KPS (Kernel PrimitiveS API) needs to exist as a kind of backend,
// because we want to implement a KPS-based kernel and make it run
// on GPU and XPU at the same time, so we need KPSContext when registering
// KPS Kernel. Note: XPU and GPU cannot be compiled at the same time!
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
using KPSContext = GPUContext;
#endif

}  // namespace phi

namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen

namespace phi {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// Currently, GPUPinnedContext is only used to data copying.
class GPUPinnedContext
    : public DeviceContext,
      public phi::TypeInfoTraits<DeviceContext, GPUPinnedContext> {
 public:
  GPUPinnedContext();
  explicit GPUPinnedContext(GPUPinnedPlace place);

  const Place& GetPlace() const override;

  Eigen::DefaultDevice* eigen_device() const;

  static const char* name() { return "GPUPinnedContext"; }

 private:
  GPUPinnedPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};
#endif
}  // namespace phi

#endif
