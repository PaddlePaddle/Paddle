//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/pten/backends/xpu/xpu_context.h"
#include "paddle/pten/api/ext/exception.h"

#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;

namespace pten {

struct XPUContext::XPUImpl {
  XPUContextResource res_;
  XPUPlace place_;
  paddle::platform::XPUVersion xpu_version_;
  xpu::Context* context_{nullptr};
  // NOTE: Distributed communicator, distributed framework manages its
  // resources, XPUContext only holds references.
  xpu::BKCLContext_t bkcl_context_{nullptr};

  void SetL3Cache() {
    const int MAX_XPU_NUM = 16;
    static void* l3ptrs[MAX_XPU_NUM] = {nullptr};

    int l3_size = 13.5 * 1024 * 1024;
    if (std::getenv("XPU_PADDLE_L3_SIZE") != nullptr) {
      l3_size = atoi(std::getenv("XPU_PADDLE_L3_SIZE"));
    }

    auto selected_xpus = GetXPUSelectedDevices();
    for (unsigned int i = 0; i < selected_xpus.size(); i++) {
      if (place.device == selected_xpus[i]) {
        if (l3ptrs[place.device] == nullptr) {
          xpu_malloc(
              static_cast<void**>(&l3ptrs[place.device]), l3_size, XPU_MEM_L3);
        }
        if (l3ptrs[place.device] != nullptr) {
          context_->_l3_mgr.set(l3ptrs[place.device], l3_size);
          VLOG(3) << "xpu place " << place.device << " set l3 size " << l3_size;
        }
        break;
      }
    }
  }

  XPUImpl() {
    context_ = xpu::create_context();
    xpu_version_ = get_xpu_version(place_.device);
  }

  explicit XPUImpl(XPUPlace place) : place_(place) {
    platform::XPUDeviceGuard guard(place.device);

    LOG_FIRST_N(WARNING, 1) << "Please NOTE: xpu device: "
                            << static_cast<int>(place_.device);

    context_ = xpu::create_context();
    SetL3Cache();
  }

  // Users need to manage external resources.
  explicit XPUImpl(const XPUContextResource& ctx_res,
                   const XPUPlace& place = XPUPlace(0))
      : res_(ctx_res), place_(place) {
    context_ = res_.context;
    // bkcl_context_ = res.bkcl_context;
    SetL3Cache();
  }

  ~XPUImpl() {
    // TODO(wilber): Why not delete resource?
  }

  Place GetPlace() const { return place_; }

  paddle::platform::XPUVersion xpu_version() const { return xpu_version_; }

  xpu::Context* GetXContext() const {
    .
        // PD_CHECK(context_ != nullptr, "the context_ is nullptr.");
        return context_;
  }

  BKCLContext_t GetBkclContext() const {
    // PD_CHECK(bkcl_context_ != nullptr, "the bkcl_context_ is nullptr.");
    return bkcl_context_;
  }

  void Wait() const {
    platform::SetXPUDeviceId(place_.GetDeviceId());
    xpu_wait(context_->xpu_stream);
  }

  void SetXContext(xpu::Context* context) {
    if (context == nullptr) {
      return;
    }
    res_.context = context;
    context_ = context;
  }

  void SetBkclContext(BKCLContext_t context) { bkcl_context_ = context; }
};

XPUContext::XPUContext() : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>();
}

XPUContext::XPUContext(const XPUContext& other) : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>();
  impl_->SetXContext(other.GetXContext());
  impl_->SetBkclContext(other.GetBkclContext());
}

XPUContext::XPUContext(XPUContext&& other) : DeviceContext() {
  impl_ = std::move(other.impl_);
}

XPUContext::~XPUContext() = default;

XPUContext::XPUContext(const XPUContextResource& ctx_res) : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>(ctx_res);
}

Place XPUContext::GetPlace() const { return impl_->GetPlace(); }

paddle::platform::XPUVersion XPUContext::xpu_version() const {
  return impl_->GetXpuVersion();
}

xpu::Context* XPUContext::x_context() const { return impl_->GetXContext(); }

BKCLContext_t XPUContext::bkcl_context() const {
  return impl_->GetBkclContext();
}

void XPUContext::Wait() const override { impl_->Wait(); }

void XPUContext::set_x_context(xpu::Context* context) {
  impl_->SetXContext(context);
}

void XPUContext::set_bkcl_context(BKCLContext_t context) {
  impl_->SetBkclContext(context);
}

}  // namespace pten
