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

#include <memory>
#include "paddle/phi/backends/c_comm_lib.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace Eigen {
struct GpuDevice;
}  // namespace Eigen

namespace phi {

class CustomContext : public DeviceContext,
                      public TypeInfoTraits<DeviceContext, CustomContext> {
 public:
  explicit CustomContext(const CustomPlace&);

  virtual ~CustomContext();

  const Place& GetPlace() const override;

  /*! \brief  Return raw stream in the device context. */
  phi::stream::stream_t stream() const;

  /*! \brief  Return stream in the device context. */
  std::shared_ptr<phi::stream::Stream> GetStream() const;

  void SetStream(std::shared_ptr<phi::stream::Stream> stream);

  // Wait for all operations completion in the stream.
  void Wait() const override;

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    return GetStream()->AddCallback(callback);
  }

  void WaitStreamCallback() const { return GetStream()->WaitCallback(); }

  Eigen::GpuDevice* eigen_device() const;

  void WaitEvent(phi::event::event_t ev) const;

  void RecordEvent(phi::event::event_t ev,
                   const std::function<void()>& callback) const;

  void RecordEvent(phi::event::event_t ev) const;

  static const char* name() { return "CustomContext"; }

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

  // Note that this is a trick implementation, which can be used to partially
  // initialize when the SetAllocator interface is not called.
  void PartialInitWithoutAllocator();
  // Note that this is a trick implementation that can be used to initialize
  // resources that require an Allocator when the SetAllocator interface is
  // called.
  void PartialInitWithAllocator();

  /*! \brief  Return xccl communicators. */
  phi::ccl::CCLComm xccl_comm() const;

  /*! \brief  Set nccl communicators. */
  void set_xccl_comm(phi::ccl::CCLComm comm);

  /*! \brief  Return compute capability in the device context. */
  int GetComputeCapability() const;

  /*! \brief  Return the SM count in the device context */
  int GetSMCount() const;

  /*! \brief  Return the Max thread num of block in the device context */
  int GetMaxThreadsPerBlock() const;

  /*! \brief  Return the max grid dim size in the device context */
  std::array<unsigned int, 3> GetCUDAMaxGridDimSize() const;

  /*! \brief  Return the max physical thread count in the device context */
  int GetMaxPhysicalThreadCount() const;

  void SetEigenDevice(Eigen::GpuDevice*);
  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&&);

  void SetComputeCapability(int val);

  void SetMaxThreadsPerMultiProcessor(int val);

  void SetMultiProcessors(int val);

  void SetMaxThreadsPerBlock(int val);

  void SetMaxGridDimSize(const std::array<unsigned int, 3>& val);

  void SetDriverVersion(int val);

  void SetRuntimeVersion(int val);

 private:
  CustomContext();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
