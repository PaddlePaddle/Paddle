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

#include <memory>
#include "paddle/pten/backends/xpu/forwards.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/device_context.h"

#include "paddle/pten/backends/xpu/xpu_header.h"
#include "paddle/pten/backends/xpu/xpu_info.h"

namespace xpu = baidu::xpu::api;

namespace pten {

struct XPUContextResource {
  xpu::Context* context{nullptr};
};

class XPUContext : public DeviceContext {
 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  XPUContext();

  explicit XPUContext(const XPUPlace&);

  // NOTE: Share the same underlying resources, please ensure that resources are
  // not released.
  XPUContext(const XPUContext&);

  XPUContext(XPUContext&&);

  virtual ~XPUContext();

  Place GetPlace() const override;

  backends::xpu::XPUVersion xpu_version() const;

  xpu::Context* x_context() const;

  // Return bkcl context.
  xpu::BKCLContext_t bkcl_context() const;

  // Wait for all operations completion in the stream.
  void Wait() const override;

 public:
  // NOTE: External users manage resources. Used in inference scenarios.
  explicit XPUContext(const XPUContextResource&, const XPUPlace& = XPUPlace(0));

  void set_x_context(xpu::Context*);

  void set_bkcl_context(xpu::BKCLContext_t context);

 private:
  struct XPUImpl;
  std::unique_ptr<XPUImpl> impl_;
};

}  // namespace pten
