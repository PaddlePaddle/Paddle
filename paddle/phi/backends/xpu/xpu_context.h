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

#ifdef PADDLE_WITH_XPU

#include <memory>

#include "paddle/phi/backends/xpu/forwards.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen

namespace xpu = baidu::xpu::api;

namespace phi {

class XPUContext : public DeviceContext,
                   public TypeInfoTraits<DeviceContext, XPUContext> {
 public:
  XPUContext();

  explicit XPUContext(const XPUPlace&);

  virtual ~XPUContext();

  const Place& GetPlace() const override;

  backends::xpu::XPUVersion xpu_version() const;

  xpu::Context* x_context() const;

  // For multi-thread dataloader,
  // check if the current thread is Dataloader thread
  bool IsDataloader() const;

  // Return bkcl context.
  xpu::BKCLContext_t bkcl_context() const;
  void SetBkclContext(xpu::BKCLContext_t context);
  void CreateStream();

  // Wait for all operations completion in the stream.
  void Wait() const override;

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

 public:
  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetXContext(xpu::Context*);

  void SetL3Cache(int l3_size = 14155776);

  Eigen::DefaultDevice* eigen_device() const { return nullptr; }

  XPUStream stream() const;

  static const char* name() { return "XPUContext"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// KPS (Kernel PrimitiveS API) needs to exist as a kind of backend,
// because we want to implement a KPS-based kernel and make it run
// on GPU and XPU at the same time, so we need KPSContext when registering
// KPS Kernel. Note: XPU and GPU cannot be compiled at the same time!
#if PADDLE_WITH_XPU_KP
using KPSContext = XPUContext;
#endif

}  // namespace phi

#endif
