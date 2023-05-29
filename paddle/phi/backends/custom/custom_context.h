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
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen

namespace phi {

class CustomContext : public DeviceContext,
                      public TypeInfoTraits<DeviceContext, CustomContext> {
 public:
  explicit CustomContext(const CustomPlace&);

  virtual ~CustomContext();

  const Place& GetPlace() const override;

  /*! \brief  Return raw stream in the device context. */
  void* stream() const;

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

  Eigen::DefaultDevice* eigen_device() const { return nullptr; }

  static const char* name() { return "CustomContext"; }

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

  /*! \brief  Return xccl communicators. */
  phi::ccl::CCLComm xccl_comm() const;

  /*! \brief  Set nccl communicators. */
  void set_xccl_comm(phi::ccl::CCLComm comm);

 private:
  CustomContext();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
