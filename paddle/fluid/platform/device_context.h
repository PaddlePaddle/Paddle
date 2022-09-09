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

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#include "paddle/fluid/platform/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/dynload/nccl.h"
#endif
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/device/gpu/gpu_helper.h"  // NOLINT
#include "paddle/fluid/platform/dynload/miopen.h"
#include "paddle/fluid/platform/dynload/rocblas.h"
#include "paddle/phi/backends/gpu/gpu_context.h"  // NOLINT
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/dynload/rccl.h"
#endif
#include "paddle/fluid/platform/device/gpu/gpu_info.h"  // NOLINT
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#endif

#include <map>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/enforce_npu.h"
#include "paddle/fluid/platform/device/npu/npu_stream.h"
#endif

#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/stream.h"

#if !defined(PADDLE_WITH_XPU_KP) || defined(__xpu_on_host__)
#include "unsupported/Eigen/CXX11/Tensor"
#endif

namespace Eigen {
struct DefaultDevice;
struct GpuDevice;
}  // namespace Eigen

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#include "acl/acl.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#endif

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/*Set the value of the global variable allow_tf32_cublas*/
void SetAllowTF32Cublas(bool active);
/*Get the global variable allow_tf32_cublas value*/
bool AllowTF32Cublas();
extern bool allow_tf32_cudnn;
/*Set the value of the global variable allow_tf32_cudnn*/
void SetAllowTF32Cudnn(bool active);
/*Get the global variable allow_tf32_cudnn value*/
bool AllowTF32Cudnn();
#endif  // PADDLE_WITH_CUDA

enum DeviceType {
  CPU = 0,
  CUDA = 1,
  NPU = 2,
  XPU = 3,
  IPU = 4,
  MLU = 5,
  CUSTOM_DEVICE = 6,

  MAX_DEVICE_TYPES = 7,
};

DeviceType Place2DeviceType(const platform::Place& place);

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kNPU = DeviceType::NPU;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kMLU = DeviceType::MLU;
constexpr DeviceType kCUSOTM_DEVICE = DeviceType::CUSTOM_DEVICE;

using DeviceContext = phi::DeviceContext;

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<platform::CPUPlace> {
  using TYPE = phi::CPUContext;
};

// Graphcore IPU
#ifdef PADDLE_WITH_IPU
class IPUDeviceContext : public DeviceContext {
 public:
  IPUDeviceContext() = delete;
  explicit IPUDeviceContext(IPUPlace place);
  virtual ~IPUDeviceContext();
  Eigen::DefaultDevice* eigen_device() const { return nullptr; }
  const Place& GetPlace() const override;
  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

 private:
  IPUPlace place_;
};
template <>
struct DefaultDeviceContextType<platform::IPUPlace> {
  using TYPE = IPUDeviceContext;
};
#endif

#ifdef PADDLE_WITH_MLU
class MLUDeviceContext;

template <>
struct DefaultDeviceContextType<platform::MLUPlace>;
#endif

#ifdef PADDLE_WITH_XPU
namespace xpu = baidu::xpu::api;
class XPUDeviceContext : public phi::XPUContext {
 public:
  XPUDeviceContext();
  explicit XPUDeviceContext(XPUPlace place);
  virtual ~XPUDeviceContext();
  Eigen::DefaultDevice* eigen_device() const { return nullptr; }
  xpuStream stream() const { return XPUContext::x_context()->xpu_stream; }
};

template <>
struct DefaultDeviceContextType<platform::XPUPlace> {
  using TYPE = XPUDeviceContext;
};
#endif

#ifdef PADDLE_WITH_ASCEND_CL
class NPUDeviceContext : public DeviceContext {
 public:
  explicit NPUDeviceContext(NPUPlace place);
  virtual ~NPUDeviceContext();
  Eigen::DefaultDevice* eigen_device() const { return nullptr; }
  const Place& GetPlace() const override;
  aclrtContext context() const;

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return npu stream in the device context. */
  aclrtStream stream() const;

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    return stream_->AddCallback(callback);
  }

  void WaitStreamCallback() const { return stream_->WaitCallback(); }

#if defined(PADDLE_WITH_ASCEND_CL)
  /*! \brief  Return hccl communicators. */
  HcclComm hccl_comm() const { return hccl_comm_; }

  /*! \brief  Set hccl communicators. */
  void set_hccl_comm(HcclComm comm) { hccl_comm_ = comm; }
#endif

  // template <typename Callback>
  // void AddStreamCallback(Callback&& callback) const {
  //   return stream_->AddCallback(callback);
  // }

  // void WaitStreamCallback() const { return stream_->WaitCallback(); }

 private:
  NPUPlace place_;
  aclrtContext context_;

#ifdef PADDLE_WITH_ASCEND_CL
  // HCCLContext_t hccl_context_;
  HcclComm hccl_comm_{nullptr};
#endif

  // Need to be the same with other DeviceContext,
  // Eventhough eigen_device_ is not used in NPU
  // NOTE(zhiqiu): why need?
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
  std::shared_ptr<stream::NPUStream> stream_;

  DISABLE_COPY_AND_ASSIGN(NPUDeviceContext);
};

template <>
struct DefaultDeviceContextType<platform::NPUPlace> {
  using TYPE = NPUDeviceContext;
};

// Currently, NPUPinnedDeviceContext is only used to data copying.
class NPUPinnedDeviceContext : public DeviceContext {
 public:
  NPUPinnedDeviceContext();
  explicit NPUPinnedDeviceContext(NPUPinnedPlace place);

  const Place& GetPlace() const override;

  Eigen::DefaultDevice* eigen_device() const;

 private:
  NPUPinnedPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

template <>
struct DefaultDeviceContextType<platform::NPUPinnedPlace> {
  using TYPE = NPUPinnedDeviceContext;
};

#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct DefaultDeviceContextType<platform::CUDAPlace> {
  using TYPE = phi::GPUContext;
};

// Currently, CUDAPinnedDeviceContext is only used to data copying.
class CUDAPinnedDeviceContext : public DeviceContext {
 public:
  CUDAPinnedDeviceContext();
  explicit CUDAPinnedDeviceContext(CUDAPinnedPlace place);

  const Place& GetPlace() const override;

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
using MKLDNNDeviceContextThreadLocals = phi::OneDNNContextThreadLocals;
using MKLDNNDeviceContext = phi::OneDNNContext;
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
class CustomDeviceContext : public phi::CustomContext {
 public:
  explicit CustomDeviceContext(CustomPlace place);
  virtual ~CustomDeviceContext();

  Eigen::DefaultDevice* eigen_device() const { return nullptr; }

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    return stream_->AddCallback(callback);
  }

  void WaitStreamCallback() const { return stream_->WaitCallback(); }

 private:
  std::shared_ptr<phi::stream::Stream> stream_;
};
template <>
struct DefaultDeviceContextType<platform::CustomPlace> {
  using TYPE = CustomDeviceContext;
};
#else
template <>
struct DefaultDeviceContextType<platform::CustomPlace> {
  using TYPE = DeviceContext;
};
#endif

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<platform::Place>& places,
    bool disable_setting_default_stream_for_allocator);

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  static DeviceContextPool& Instance() {
    PADDLE_ENFORCE_NOT_NULL(pool,
                            platform::errors::PreconditionNotMet(
                                "Need to Create DeviceContextPool firstly!"));
    return *pool;
  }

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Init(const std::vector<platform::Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  static bool IsInitialized() { return pool != nullptr; }

  static void SetPool(DeviceContextPool* dev_pool) { pool = dev_pool; }

  /*! \brief  Return handle of single device context. */
  platform::DeviceContext* Get(const platform::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  size_t size() const;

  const std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>&
  device_contexts() const;

  static void SetDeviceContexts(
      const std::map<Place,
                     std::shared_future<std::unique_ptr<DeviceContext>>>*);

 private:
  explicit DeviceContextPool(const std::vector<platform::Place>& places);

  static DeviceContextPool* pool;
  std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>
      device_contexts_;
  static thread_local const std::
      map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
          external_device_contexts_;  // not owned
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace platform
}  // namespace paddle
