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

#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

#include "paddle/fluid/platform/place.h"
// TODO(wilber) need to remove!
#include "paddle/fluid/platform/xpu/xpu_info.h"

#include "paddle/pten/core/device_context.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif

namespace pten {

using Place = paddle::platform::Place;
using XPUPlace = paddle::platform::XPUPlace;
namespace xpu = baidu::xpu::api;

// TODO(wilber): xpu version?
// enum XPUVersion { XPU1, XPU2 };
using XPUVersion = paddle::platform::XPUVersion;

class XPUContext : public DeviceContext {
 public:
  explicit XPUContext(XPUPlace place) : place_(place) {}
  virtual ~XPUContext() {}

  Place GetPlace() const noexcept override { return place_; }

  xpu::Context* x_context() const { return context_; }
  void SetContext(xpu::Context* context);

  XPUVersion xpu_version() const { return xpu_version_; }
  void SetXpuVersion(XPUVersion ver) { xpu_version_ = ver; }

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() override;

#ifdef PADDLE_WITH_XPU_BKCL
  /*! \brief  Return bkcl context. */
  BKCLContext_t bkcl_context() const { return bkcl_context_; }

  /*! \brief  Set bkcl context. */
  void set_bkcl_context(BKCLContext_t context) { bkcl_context_ = context; }
#endif

 private:
  // TODO(wilber): place or device_id?
  XPUPlace place_;
  int device_id_;

  // TODO(wilber): need XPUVersion or not?
  XPUVersion xpu_version_;
  xpu::Context* context_;

#ifdef PADDLE_WITH_XPU_BKCL
  BKCLContext_t bkcl_context_;
#endif
};

}  // namespace pten
