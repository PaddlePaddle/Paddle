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
#include <memory>
#include "paddle/pten/api/ext/exception.h"

#include "paddle/pten/common/place.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;

namespace pten {

struct XPUContext::XPUImpl {
  void SetL3Cache() {
    const int MAX_XPU_NUM = 16;
    static void* l3ptrs[MAX_XPU_NUM] = {nullptr};

    int l3_size = 13.5 * 1024 * 1024;
    if (std::getenv("XPU_PADDLE_L3_SIZE") != nullptr) {
      l3_size = atoi(std::getenv("XPU_PADDLE_L3_SIZE"));
    }

    auto selected_xpus = backends::xpu::GetXPUSelectedDevices();
    for (unsigned int i = 0; i < selected_xpus.size(); i++) {
      if (place_.GetDeviceId() == selected_xpus[i]) {
        if (l3ptrs[place_.GetDeviceId()] == nullptr) {
          xpu_malloc(static_cast<void**>(&l3ptrs[place_.GetDeviceId()]),
                     l3_size,
                     XPU_MEM_L3);
        }
        if (l3ptrs[place_.GetDeviceId()] != nullptr) {
          context_->_l3_mgr.set(l3ptrs[place_.GetDeviceId()], l3_size);
          VLOG(3) << "xpu place " << place_.GetDeviceId() << " set l3 size "
                  << l3_size;
        }
        break;
      }
    }
  }

  XPUImpl() {
    context_ = xpu::create_context();
    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
  }

  explicit XPUImpl(XPUPlace place) : place_(place) {
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());

    LOG_FIRST_N(WARNING, 1) << "Please NOTE: xpu device: "
                            << static_cast<int>(place_.device);

    context_ = xpu::create_context();
    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
    SetL3Cache();
  }

  // Users need to manage external resources.
  explicit XPUImpl(const XPUContextResource& ctx_res,
                   const XPUPlace& place = XPUPlace(0))
      : res_(ctx_res), place_(place) {
    context_ = res_.context;
    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
    SetL3Cache();
  }

  ~XPUImpl() {
    if (res_.context == nullptr && context_ != nullptr) {
      xpu::destroy_context(context_);
      context_ = nullptr;
    }
  }

  Place GetPlace() const { return place_; }

  backends::xpu::XPUVersion GetXpuVersion() const { return xpu_version_; }

  xpu::Context* GetXContext() const {
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    return context_;
  }

  xpu::BKCLContext_t GetBkclContext() const { return bkcl_context_; }

  void Wait() const {
    backends::xpu::SetXPUDeviceId(place_.GetDeviceId());
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    xpu_wait(context_->xpu_stream);
  }

  void SetXContext(xpu::Context* context) {
    if (context == nullptr) {
      return;
    }

    // if owning resouce
    if (res_.context == nullptr && context_ != nullptr) {
      xpu::destroy_context(context_);
      context_ = nullptr;
    }

    res_.context = context;
    context_ = context;
  }

  void SetBkclContext(xpu::BKCLContext_t context) { bkcl_context_ = context; }

  XPUContextResource res_;
  XPUPlace place_;
  backends::xpu::XPUVersion xpu_version_;
  xpu::Context* context_{nullptr};
  // NOTE: Distributed communicator, distributed framework manages its
  // resources, XPUContext only holds references.
  xpu::BKCLContext_t bkcl_context_{nullptr};
};

XPUContext::XPUContext() : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>();
}

XPUContext::XPUContext(const XPUPlace& place) : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>(place);
}

XPUContext::XPUContext(const XPUContext& other) : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>(other.GetPlace());
  impl_->SetXContext(other.x_context());
  impl_->SetBkclContext(other.bkcl_context());
}

XPUContext::XPUContext(XPUContext&& other) : DeviceContext() {
  impl_ = std::move(other.impl_);
}

XPUContext::~XPUContext() = default;

XPUContext::XPUContext(const XPUContextResource& ctx_res, const XPUPlace& place)
    : DeviceContext() {
  impl_ = std::make_unique<XPUImpl>(ctx_res, place);
}

Place XPUContext::GetPlace() const { return impl_->GetPlace(); }

backends::xpu::XPUVersion XPUContext::xpu_version() const {
  return impl_->GetXpuVersion();
}

xpu::Context* XPUContext::x_context() const { return impl_->GetXContext(); }

xpu::BKCLContext_t XPUContext::bkcl_context() const {
  return impl_->GetBkclContext();
}

void XPUContext::Wait() const { impl_->Wait(); }

void XPUContext::set_x_context(xpu::Context* context) {
  impl_->SetXContext(context);
}

void XPUContext::set_bkcl_context(xpu::BKCLContext_t context) {
  impl_->SetBkclContext(context);
}

}  // namespace pten
