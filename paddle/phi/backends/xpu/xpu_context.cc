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

#include "glog/logging.h"

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/os_info.h"
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
        if (l3ptrs[place_.GetDeviceId()] != nullptr) {
          xpu_free(l3ptrs[place_.GetDeviceId()]);
          l3ptrs[place_.GetDeviceId()] = nullptr;
        }
        xpu_malloc(static_cast<void**>(&l3ptrs[place_.GetDeviceId()]),
                   l3_size,
                   XPU_MEM_L3);
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
    for (auto& ctx_it : context_map_) {
      auto& ctx = ctx_it.second;
      if (ctx != nullptr) {
        xpu_wait(ctx->xpu_stream);
        if (ctx->xpu_stream) {
          xpu_stream_destroy(ctx->xpu_stream);
          ctx->xpu_stream = nullptr;
        }
        ctx = nullptr;
      }
    }
    context_map_.clear();

    if (owned_ && context_ != nullptr) {
      backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
      xpu_wait(context_->xpu_stream);
      if (context_->xpu_stream && stream_owned_) {
        // manually destroy XPUStream here until xpu::api integrates this work
        // into Context dtor
        xpu_stream_destroy(context_->xpu_stream);
        context_->xpu_stream = nullptr;
      }
      xpu::destroy_context(context_);
      context_ = nullptr;
    }
  }

  const Place& GetPlace() const { return place_; }

  XPUStream stream() const {
    xpu::Context* ctx_t = GetXdlCtx();
    if (ctx_t) {
      return ctx_t->xpu_stream;
    }
    return context_->xpu_stream;
  }

  // Set external stream for context
  void SetStream(void* stream) {
    if (context_->xpu_stream != nullptr && stream_owned_) {
      xpu_stream_destroy(context_->xpu_stream);
    }
    stream_owned_ = false;
    context_->set_stream(static_cast<XPUStream>(stream));
  }

  xpu::Context* GetXContext() const {
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    return context_;
  }

  xpu::BKCLContext_t GetBkclContext() const {
    PD_CHECK(bkcl_context_ != nullptr, "the xpu bkcl_context is nullptr.");
    return bkcl_context_;
  }

  // Overload GetXContext function to set and get
  // contexts of XPU Dataloader threads, and keep old GetXContext Method
  xpu::Context* GetXContext() {
    SetXdlCtx();
    xpu::Context* ctx_t = GetXdlCtx();
    if (ctx_t) {
      PD_CHECK(ctx_t != nullptr, "the xpu context is nullptr.");
      return ctx_t;
    }

    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    return context_;
  }

  void Wait() const {
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    xpu_wait(context_->xpu_stream);
    xpu::Context* ctx_t = GetXdlCtx();
    if (ctx_t) {
      PD_CHECK(ctx_t != nullptr, "the xpu context is nullptr.");
      xpu_wait(ctx_t->xpu_stream);
    }
  }

  void Init() {
    owned_ = true;
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    LOG_FIRST_N(WARNING, 1)
        << "Please NOTE: xpu device: " << static_cast<int>(place_.device);
    context_ = xpu::create_context();
    context_->set_option("XPUAPI_DEFAULT_SIZE", "1");
    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
    SetL3Cache();
  }

  void SetXContext(xpu::Context* context) {
    if (context_ != nullptr) {
      backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
      xpu_wait(context_->xpu_stream);
      if (context_->xpu_stream != nullptr && stream_owned_) {
        xpu_stream_destroy(context_->xpu_stream);
        stream_owned_ = false;
        context_->xpu_stream = nullptr;
      }
      if (owned_) {
        xpu::destroy_context(context_);
      }
    }
    context_ = context;
    owned_ = false;
  }

  void SetBkclContext(xpu::BKCLContext_t context) { bkcl_context_ = context; }

  void CreateStream() {
    if (context_->xpu_stream) {
      VLOG(3) << "xpu stream is already created for current context";
      return;
    }
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&context_->xpu_stream));
    stream_owned_ = true;
  }

  void SetXdlCtx() {
    std::string tname = phi::GetCurrentThreadName();
    if (tname.substr(0, 10) == "Dataloader" &&
        context_map_.find(tname) == context_map_.end()) {
      VLOG(4) << "Set XPU Dataloader Context with current thread name = "
              << tname << " currently " << context_map_.size()
              << " contexts existing";
      xpu::Context* ctx_t = xpu::create_context();
      context_map_[tname] = ctx_t;
    }
  }

  xpu::Context* GetXdlCtx() const {
    std::string tname = phi::GetCurrentThreadName();
    VLOG(4) << "Get XPU Context with current thread name = " << tname
            << " currently " << context_map_.size() << " contexts existing";
    if (tname.substr(0, 10) != "Dataloader") {
      return context_;
    } else {
      return (context_map_.find(tname) == context_map_.end())
                 ? nullptr
                 : context_map_.find(tname)->second;
    }
  }

  bool owned_{false};
  bool stream_owned_{false};
  Place place_;
  backends::xpu::XPUVersion xpu_version_;
  int runtime_version_;
  int driver_version_;
  xpu::Context* context_{nullptr};
  std::unordered_map<std::string, xpu::Context*> context_map_;

  // NOTE: Distributed communicator, distributed framework manages its
  // resources, XPUContext only holds references.
  xpu::BKCLContext_t bkcl_context_{nullptr};
};

XPUContext::XPUContext() : DeviceContext(), impl_(std::make_unique<Impl>()) {
  impl_->Init();
}

XPUContext::XPUContext(const XPUPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
  impl_->Init();
}

XPUContext::~XPUContext() = default;

const Place& XPUContext::GetPlace() const { return impl_->GetPlace(); }

XPUStream XPUContext::stream() const { return impl_->stream(); }

void XPUContext::SetStream(void* stream) { impl_->SetStream(stream); }

void XPUContext::SetXpuVersion(int version) {
  impl_->xpu_version_ = static_cast<backends::xpu::XPUVersion>(version);
}

void XPUContext::SetRuntimeVersion(int version) {
  impl_->runtime_version_ = version;
}

void XPUContext::SetDriverVersion(int version) {
  impl_->driver_version_ = version;
}

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

void XPUContext::CreateStream() { impl_->CreateStream(); }

void XPUContext::Init() { impl_->Init(); }
}  // namespace phi
