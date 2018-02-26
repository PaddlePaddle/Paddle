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
#include <unordered_map>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "glog/logging.h"

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

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(CUDAPlace place);
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
  CUDAPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
};

template <>
struct DefaultDeviceContextType<platform::CUDAPlace> {
  using TYPE = CUDADeviceContext;
};

#endif

#ifdef PADDLE_WITH_MKLDNN
class MKLDNNDeviceContext : public CPUDeviceContext {
 public:
  explicit MKLDNNDeviceContext(CPUPlace place);

  /* \brief  Add new element: memory, primitive or primitive desc */
  template <typename T>
  void AddElement(const std::string& op_key, const T& value);

  /* \brief  Get existed element: memory, primitive or primitive desc */
  template <typename T>
  const T& GetElement(const std::string& op_key) const;

  /* \brief  Get element pool: memory, primitive or primitive desc pool */
  template <typename T>
  const std::unordered_map<const std::string, const T, std::hash<std::string>>&
  GetElementPool() const;

  /* \brief  Get the active engine */
  const MKLDNNEngine& engine() const { return *engine_; }

  /* \brief  Submit primitive to pipeline */
  void Submit(const MKLDNNPrimitivePtr& p) { pipeline_.push_back(*p); }

  /*! \brief  Execute all submitted primitives in pipeline */
  void Execute(bool block = true);

 protected:
  /*! \brief  Reset the stream to prepare next exectue */
  void ResetStream();

 private:
  std::unordered_map<const std::string, const MKLDNNMemoryPtr,
                     std::hash<std::string>>
      memory_pool_;
  std::unordered_map<const std::string, const MKLDNNPrimitivePtr,
                     std::hash<std::string>>
      primitive_pool_;
  std::unordered_map<const std::string, const MKLDNNPrimitiveDescPtr,
                     std::hash<std::string>>
      primitive_desc_pool_;
  std::vector<MKLDNNPrimitive> pipeline_;
  MKLDNNStreamPtr stream_;
  MKLDNNEnginePtr engine_;
  bool ready_;
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
  const platform::DeviceContext* Get(const platform::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  size_t size() const { return device_contexts_.size(); }

 private:
  static DeviceContextPool* pool;
  constexpr static int LEFT_SHIFT = 8;
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const platform::Place& place) const {
      int pre_hash = place.which() << LEFT_SHIFT;
      if (platform::is_gpu_place(place)) {
        pre_hash += boost::get<platform::CUDAPlace>(place).GetDeviceId();
      }
      return hash_(pre_hash);
    }
  };
  std::unordered_map<const platform::Place, const platform::DeviceContext*,
                     Hash>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace platform
}  // namespace paddle
