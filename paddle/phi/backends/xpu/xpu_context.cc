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

#include "paddle/phi/backends/xpu/xpu_context.h"

#include <memory>

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/place.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;

namespace phi {

struct XPUContext::Impl {
  void SetL3Cache(int l3_size = 14155776) {
    const int MAX_XPU_NUM = 16;
    static void* l3ptrs[MAX_XPU_NUM] = {nullptr};

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
          VLOG(3) << "xpu place " << static_cast<int>(place_.GetDeviceId())
                  << " set l3 size " << l3_size;
        }
        break;
      }
    }
  }

  Impl() : place_(XPUPlace()) {}

  explicit Impl(const Place& place) : place_(place) {}

  ~Impl() {
    if (owned_ && context_ != nullptr) {
      backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
      // manually destroy XPUStream here until xpu::api integrates this work
      // into Context dtor
      xpu_wait(context_->xpu_stream);
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(context_->xpu_stream));
      context_->xpu_stream = nullptr;
      xpu::destroy_context(context_);
      context_ = nullptr;
    }
  }

  const Place& GetPlace() const { return place_; }

  XPUStream stream() const {
    auto s = context_->xpu_stream;
    PD_CHECK(s != nullptr, "the xpu stream is nullptr.");
    return s;
  }

  xpu::Context* GetXContext() const {
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    return context_;
  }

  xpu::BKCLContext_t GetBkclContext() const {
    PD_CHECK(bkcl_context_ != nullptr, "the xpu bkcl_context is nullptr.");
    return bkcl_context_;
  }

  void Wait() const {
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    xpu_wait(context_->xpu_stream);
  }

  void Init() {
    owned_ = true;
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    LOG_FIRST_N(WARNING, 1)
        << "Please NOTE: xpu device: " << static_cast<int>(place_.device);
    context_ = xpu::create_context();
    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
    SetL3Cache();
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&context_->xpu_stream));
  }

  void SetXContext(xpu::Context* context) { context_ = context; }

  void SetBkclContext(xpu::BKCLContext_t context) { bkcl_context_ = context; }

  bool owned_{false};
  Place place_;
  backends::xpu::XPUVersion xpu_version_;
  xpu::Context* context_{nullptr};

  // NOTE: Distributed communicator, distributed framework manages its
  // resources, XPUContext only holds references.
  xpu::BKCLContext_t bkcl_context_{nullptr};
};

XPUContext::XPUContext() : DeviceContext(), impl_(std::make_unique<Impl>()) {}

XPUContext::XPUContext(const XPUPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {}

XPUContext::~XPUContext() = default;

const Place& XPUContext::GetPlace() const { return impl_->GetPlace(); }

XPUStream XPUContext::stream() const { return impl_->stream(); }

backends::xpu::XPUVersion XPUContext::xpu_version() const {
  return impl_->xpu_version_;
}

xpu::Context* XPUContext::x_context() const { return impl_->GetXContext(); }

xpu::BKCLContext_t XPUContext::bkcl_context() const {
  return impl_->GetBkclContext();
}

void XPUContext::Wait() const { impl_->Wait(); }

void XPUContext::SetXContext(xpu::Context* context) {
  impl_->SetXContext(context);
}

void XPUContext::SetL3Cache(int l3_size) { impl_->SetL3Cache(l3_size); }

void XPUContext::SetBkclContext(xpu::BKCLContext_t context) {
  impl_->SetBkclContext(context);
}

void XPUContext::Init() { impl_->Init(); }

}  // namespace phi
