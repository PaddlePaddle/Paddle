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
#include "paddle/phi/backends/context_pool.h"

#ifdef PADDLE_WITH_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/phi/core/xpu_cuda_stream.h"
#endif

#include <memory>

#include "glog/logging.h"

#include "paddle/common/exception.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/os_info.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

#if !defined(PADDLE_WITH_XPU_KP) || defined(__xpu_on_host__)
#include "unsupported/Eigen/CXX11/Tensor"
#endif

namespace xpu = baidu::xpu::api;

namespace phi {

struct XPUContext::Impl {
  void SetL3Cache(int64_t l3_size = 1024) {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(context_->xpu_stream));
    context_->_l3_mgr.set(nullptr, 0, true);  // free origin l3
    void* l3_ptr = nullptr;
    xpu_malloc(static_cast<void**>(&l3_ptr), l3_size, XPU_MEM_L3);

    if (l3_ptr != nullptr) {
      VLOG(3) << "xpu place " << static_cast<int>(place_.GetDeviceId())
              << "context " << context_ << " set l3 size " << l3_size;
      context_->_l3_mgr.set(l3_ptr, l3_size, true);
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
  void SetStream(void* stream, bool clear = true) {
    if (clear && context_->xpu_stream != nullptr && stream_owned_) {
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

  void Wait() {
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    PD_CHECK(context_ != nullptr, "the xpu context is nullptr.");
    PADDLE_ENFORCE_XRE_SUCCESS(xpu_wait(context_->xpu_stream));
    xpu::Context* ctx_t = GetXdlCtx();
    if (ctx_t) {
      PD_CHECK(ctx_t != nullptr, "the xpu context is nullptr.");
      PADDLE_ENFORCE_XRE_SUCCESS(xpu_wait(ctx_t->xpu_stream));
    }

    ClearStashedMemory();
  }

  class XHPCBufferManager {
   public:
    void* Alloc(const Place& place, size_t size, XPUStream xpu_stream) {
      VLOG(3) << "Alloc " << size << " bytes from XHPC on stream "
              << xpu_stream;
      phi::Stream stream(reinterpret_cast<StreamId>(xpu_stream));
      auto allocation = memory_utils::Alloc(place, size, stream);
      void* ret = allocation.get()->ptr();
      allocations_to_free_.back().push_back(std::move(allocation));
      return ret;
    }

    void Save() {
      allocations_to_free_.emplace_back();
      VLOG(3) << "XHPC ctx_guard created, " << GetStackLevel()
              << " are in use now.";
    }

    void Free() {
      PADDLE_ENFORCE_GT(GetStackLevel(),
                        0,
                        errors::PreconditionNotMet(
                            "No ctx_guard when overload_free is called"));
      allocations_to_free_.pop_back();
      VLOG(3) << "XHPC ctx_guard destroyed, " << GetStackLevel()
              << " are in use now.";
    }

   private:
    size_t GetStackLevel() const { return allocations_to_free_.size(); }
    std::vector<std::vector<Allocator::AllocationPtr>> allocations_to_free_;
  };

  void Init(int64_t gm_default_size = 1024,
            int64_t l3_default_size = 1024,
            bool is_comm_context = false) {
    owned_ = true;
    backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    LOG_FIRST_N(WARNING, 1)
        << "Please NOTE: xpu device: " << static_cast<int>(place_.device);

    context_ = xpu::create_context();

    if (std::getenv("XPU_CDNN_CLUSTER_PARALLEL") != nullptr &&
        !is_comm_context) {
      XPUStream s;
      xpu_stream_create(&s);
      context_->set_stream(s);
    }

    if (std::getenv("XPU_PADDLE_DISABLE_ALLOC_OVERLOAD") == nullptr) {
      // overload ctx alloc/free to avoid xpu_malloc/xpu_wait
      auto overload_alloc_fn =
          [&xhpc_buf_mgr = xhpc_buf_mgr_,
           &place = place_,
           s = context_->get_stream()](size_t size) -> void* {
        return xhpc_buf_mgr.Alloc(place, size, s);
      };
      auto overload_save_fn = [&xhpc_buf_mgr = xhpc_buf_mgr_]() {
        xhpc_buf_mgr.Save();
      };
      auto overload_free_fn = [&xhpc_buf_mgr = xhpc_buf_mgr_]() {
        xhpc_buf_mgr.Free();
      };
      context_->set_overload_alloc(
          overload_alloc_fn, overload_free_fn, overload_save_fn);
      gm_default_size = 1;
      VLOG(1) << "XPUAPI_DEFUAULT_SIZE is disabled because you overload the "
                 "alloc of xhpc. If you want to use XPUAPI_DEFAULT_SIZE, "
                 "please set XPU_PADDLE_DISABLE_ALLOC_OVERLOAD=1";
    }

    context_->set_option("XPUAPI_DEFAULT_SIZE",
                         std::to_string(gm_default_size).c_str());
    VLOG(3) << "xpu place " << static_cast<int>(place_.GetDeviceId())
            << "context " << context_ << " set xpuapi_default_size "
            << gm_default_size;

    xpu_version_ = backends::xpu::get_xpu_version(place_.device);
    SetL3Cache(l3_default_size);
  }

  void SetXContext(xpu::Context* context) {
    if (context_ != nullptr) {
      backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
      PADDLE_ENFORCE_XRE_SUCCESS(xpu_wait(context_->xpu_stream));
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
      // DataLoader does not require a pre-allocated GM buffer
      // to avoid xpu_wait calls
      ctx_t->set_option("XPUAPI_DEFAULT_SIZE", "1");
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

  void AddStashedMemory(const DenseTensor& tensor) {
    stashed_mem_for_free_.push_back(tensor.Holder());
  }

  void ClearStashedMemory() { stashed_mem_for_free_.clear(); }

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
  XHPCBufferManager xhpc_buf_mgr_;
  std::vector<std::shared_ptr<Allocation>> stashed_mem_for_free_;
};

static int64_t get_gm_size(int i) {
  int64_t default_size = 1024;
  if (std::getenv("XPUAPI_DEFAULT_SIZE") != nullptr) {
    default_size = std::atoll(std::getenv("XPUAPI_DEFAULT_SIZE"));
  }
  std::string cur_env = std::string("XPUAPI_DEFAULT_SIZE") + std::to_string(i);
  if (std::getenv(cur_env.c_str()) != nullptr) {
    default_size = std::atoll(std::getenv(cur_env.c_str()));
  }
  return default_size;
}

static int64_t get_l3_size(int i) {
  int64_t default_size = 1024;
  if (std::getenv("XPU_PADDLE_L3_SIZE") != nullptr) {
    default_size = std::atoll(std::getenv("XPU_PADDLE_L3_SIZE"));
  }
  std::string cur_env = std::string("XPU_PADDLE_L3_SIZE") + std::to_string(i);
  if (std::getenv(cur_env.c_str()) != nullptr) {
    default_size = std::atoll(std::getenv(cur_env.c_str()));
  }
  return default_size;
}

XPUContext::XPUContext() : DeviceContext() {
  if (std::getenv("XPU_CDNN_CLUSTER_PARALLEL") != nullptr) {
    int default_num_stream = 2;
    if (std::getenv("XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER") != nullptr) {
      default_num_stream =
          atoi(std::getenv("XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER"));
    }
    for (int i = 0; i < default_num_stream; i++) {
      impls_.push_back(std::make_unique<Impl>());
      impls_[i]->Init(get_gm_size(i), get_l3_size(i));
    }
  } else {
    impls_.push_back(std::make_unique<Impl>());
    impls_[0]->Init(get_gm_size(0), get_l3_size(0));
    stream_pool.push_back(impls_[0]->context_->get_stream());
    idle_stream_flags.push_back(false);
    current_stream_handle =
        XPUStreamHandle(impls_[0]->context_->get_stream(), 0);
    if (std::getenv("XPU_DEFAULT_STREAM_NUMBER") != nullptr) {
      int default_num_stream = atoi(std::getenv("XPU_DEFAULT_STREAM_NUMBER"));
      for (int i = 0; i < default_num_stream; i++) {
        XPUStream s;
        PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&s));
        stream_pool.push_back(s);
        idle_stream_flags.push_back(true);
      }
    }
  }
  current_stream_idx = 0;
}

XPUContext::XPUContext(const XPUPlace& place, bool is_comm_context)
    : DeviceContext() {
  if (is_comm_context) {
    // for communication context init, with gm_size=1 and l3_size=1
    impls_.push_back(std::make_unique<Impl>(place));
    impls_[0]->Init(0, 0, true);
  } else if (std::getenv("XPU_CDNN_CLUSTER_PARALLEL") != nullptr) {
    int default_num_stream = 4;
    if (std::getenv("XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER") != nullptr) {
      default_num_stream =
          atoi(std::getenv("XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER"));
    }
    for (int i = 0; i < default_num_stream; i++) {
      impls_.push_back(std::make_unique<Impl>(place));
      impls_[i]->Init(get_gm_size(i), get_l3_size(i));
    }
    stream_pool.push_back(impls_[0]->context_->get_stream());
    idle_stream_flags.push_back(false);
  } else {
    impls_.push_back(std::make_unique<Impl>(place));
    impls_[0]->Init(get_gm_size(0), get_l3_size(0));
    stream_pool.push_back(impls_[0]->context_->get_stream());
    idle_stream_flags.push_back(false);
    current_stream_handle =
        XPUStreamHandle(impls_[0]->context_->get_stream(), 0);
  }

  current_stream_idx = 0;
}

XPUContext::~XPUContext() = default;

const Place& XPUContext::GetPlace() const { return impls_[0]->GetPlace(); }

XPUStream XPUContext::stream(int i) const {
  CheckValidStreamId(i);
  return impls_[i]->stream();
}

void XPUContext::SetStream(void* stream, int i) {
  CheckValidStreamId(i);
  impls_[i]->SetStream(stream);
  if (i == 0) {
    current_stream_handle.set_stream(static_cast<XPUStream>(stream));
  }
}

void XPUContext::CheckValidStreamId(int i) const {
  PADDLE_ENFORCE_GE(
      i,
      0,
      errors::InvalidArgument(
          "The stream index must be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      i,
      GetStreamNum(),
      errors::InvalidArgument("The stream index should be less than the number "
                              "of stream used (%d), but got %d",
                              GetStreamNum(),
                              i));
}

void XPUContext::CheckValidIdxInRange(int i, int i_max) const {
  PADDLE_ENFORCE_GE(
      i,
      0,
      errors::InvalidArgument(
          "The stream index must be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      i,
      i_max,
      errors::InvalidArgument("The stream index should be less than the number "
                              "of stream used (%d), but got %d",
                              i_max,
                              i));
}

void XPUContext::SetXpuVersion(int version) {
  impls_[0]->xpu_version_ = static_cast<backends::xpu::XPUVersion>(version);
}

void XPUContext::SetRuntimeVersion(int version) {
  impls_[0]->runtime_version_ = version;
}

void XPUContext::SetDriverVersion(int version) {
  impls_[0]->driver_version_ = version;
}

backends::xpu::XPUVersion XPUContext::xpu_version() const {
  return impls_[0]->xpu_version_;
}

xpu::Context* XPUContext::x_context(int i) const {
  CheckValidStreamId(i);
  return impls_[i]->GetXContext();
}

xpu::BKCLContext_t XPUContext::bkcl_context() const {
  return impls_[0]->GetBkclContext();
}

void XPUContext::Wait() const {
  for (uint64_t i = 0; i < impls_.size(); i++) {
    impls_[i]->Wait();
  }
}

void XPUContext::SetXContext(xpu::Context* context, int i) {
  CheckValidStreamId(i);
  impls_[i]->SetXContext(context);
}

void XPUContext::SetL3Cache(int64_t l3_size, int i) {
  CheckValidStreamId(i);
  impls_[i]->SetL3Cache(l3_size);
}

void XPUContext::SetBkclContext(xpu::BKCLContext_t context) {
  impls_[0]->SetBkclContext(context);
}

void XPUContext::CreateStream(int i) {
  CheckValidStreamId(i);
  impls_[i]->CreateStream();
}

void XPUContext::RecordEvent(XPUEvent event, int s) const {
  CheckValidStreamId(s);
  int r = xpu_event_record(event, stream(s));
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

void XPUContext::StreamWaitEvent(XPUEvent event, int s) const {
  CheckValidStreamId(s);
  int r = xpu_stream_wait_event(stream(s), event);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

void XPUContext::StreamWaitStream(int wait_stream, int record_stream) const {
  CheckValidStreamId(wait_stream);
  CheckValidStreamId(record_stream);
  XPUEvent event = XPUEventPool::Instance().CreateEventFromPool();
  RecordEvent(event, record_stream);
  StreamWaitEvent(event, wait_stream);
  impls_[record_stream]->ClearStashedMemory();
}

int64_t XPUContext::GetStreamNum() const { return impls_.size(); }

int XPUContext::SetCurrentStream(int idx) {
  int prev_stream_idx = current_stream_idx;
  if (prev_stream_idx != idx) {
    impls_[0]->SetStream(stream_pool[idx]);
    current_stream_handle.set_stream(stream_pool[idx]);
    current_stream_idx = idx;
    idle_stream_flags[prev_stream_idx] = true;
    idle_stream_flags[current_stream_idx] = false;
  }
  return prev_stream_idx;
}

void XPUContext::StreamWaitStreamInPool(int wait_stream,
                                        int record_stream) const {
  CheckValidIdxInRange(wait_stream, stream_pool.size());
  CheckValidIdxInRange(record_stream, stream_pool.size());
  XPUEvent event = XPUEventPool::Instance().CreateEventFromPool();
  int r = xpu_event_record(event, stream_pool[record_stream]);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
  r = xpu_stream_wait_event(stream_pool[wait_stream], event);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

void XPUContext::StreamWaitEventInPool(int wait_stream, XPUEvent event) const {
  CheckValidIdxInRange(wait_stream, stream_pool.size());
  int r = xpu_stream_wait_event(stream_pool[wait_stream], event);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

int XPUContext::get_idle_stream() {
  bool found_idle_stream = false;
  int stream_idx = 0;
  int num_streams = idle_stream_flags.size();
  for (; stream_idx < num_streams; stream_idx++) {
    if (idle_stream_flags[stream_idx]) {
      found_idle_stream = true;
      break;
    }
  }
  if (found_idle_stream) {
    idle_stream_flags[stream_idx] = false;
    return stream_idx;
  } else {
    add_stream_to_pool();
    return stream_pool.size() - 1;
  }
}

void XPUContext::add_stream_to_pool() {
  XPUStream s;
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&s));
  stream_pool.push_back(s);
  idle_stream_flags.push_back(false);
}

XPUStream XPUContext::get_stream_from_pool(int idx) const {
  PADDLE_ENFORCE_GE(
      idx,
      0,
      errors::InvalidArgument(
          "The stream index must be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      idx,
      stream_pool.size(),
      errors::InvalidArgument("The stream index should be less than the number "
                              "of stream used (%d), but got %d",
                              stream_pool.size(),
                              idx));
  return stream_pool[idx];
}

int XPUContext::get_current_stream_idx() { return current_stream_idx; }
void XPUContext::AddStashedMemory(int stream, const DenseTensor& tensor) {
  CheckValidStreamId(stream);
  impls_[stream]->AddStashedMemory(tensor);
}

XPUStream XPUContext::get_current_stream() { return impls_[0]->stream(); }

XPUStreamHandle* XPUContext::get_current_stream_handle() {
  if (impls_[0]->context_->get_stream() == nullptr) {
    XPUStream s;
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&s));
    impls_[0]->SetStream(s);
    stream_pool[current_stream_idx] = s;
    current_stream_handle.set_stream(s);
  }
  return &current_stream_handle;
}

void XPUContext::Init() { impls_[0]->Init(); }

XPUContext* get_xpu_context(int device_id) {
  auto place_tmp = phi::XPUPlace(
      device_id > -1 ? device_id : phi::backends::xpu::GetXPUCurrentDeviceId());
  phi::XPUContext* dev_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place_tmp));

  return dev_ctx;
}

XPUStreamHandle::XPUStreamHandle() {}

XPUStreamHandle::XPUStreamHandle(const int idx) {
  auto* dev_ctx = phi::get_xpu_context();
  stream_id = idx;
  stream = dev_ctx->get_stream_from_pool(stream_id);
}

XPUStreamHandle::XPUStreamHandle(const phi::XPUPlace& place) {
  phi::XPUContext* dev_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  stream_id = dev_ctx->get_idle_stream();
  stream = dev_ctx->get_stream_from_pool(stream_id);
}

XPUStreamHandle::XPUStreamHandle(const XPUStream xpu_stream, const int id) {
  stream = xpu_stream;
  stream_id = id;
}

void XPUStreamHandle::Init() {
  auto* dev_ctx = phi::get_xpu_context();
  stream_id = dev_ctx->get_idle_stream();
  stream = dev_ctx->get_stream_from_pool(stream_id);
}

void XPUStreamHandle::wait_event(XPUEvent event) const {
  int r = xpu_stream_wait_event(stream, event);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

void XPUStreamHandle::synchronize() const {
  int r = xpu_wait(stream);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

void XPUStreamHandle::set_stream(XPUStream stream_) { stream = stream_; }

void XPUStreamHandle::record_event(XPUEvent event) const {
  int r = xpu_event_record(event, stream);
  PADDLE_ENFORCE_XRE_SUCCESS(r);
}

XPUStreamHandle get_current_stream_handle(int device_id) {
  auto* dev_ctx = get_xpu_context(device_id);
  return *dev_ctx->get_current_stream_handle();
}

XPUStreamHandle get_stream_handle(int device_id) {
  auto* dev_ctx = get_xpu_context(device_id);
  return XPUStreamHandle(dev_ctx->get_idle_stream());
}

void set_current_stream(XPUStreamHandle* s) {
  auto* dev_ctx = get_xpu_context();
  dev_ctx->SetStream(s->raw_stream(), 0);
}

XPUEventPool& XPUEventPool::Instance() {
  static XPUEventPool pool;
  return pool;
}

XPUEventPool::~XPUEventPool() {
  const auto& DestroyEvent = [](XPUEvent event) {
    int r = xpu_event_destroy(event);
    PADDLE_ENFORCE_XRE_SUCCESS(r);
  };
  const auto& CheckComplishAndDestroy = [&](XPUEvent event) -> bool {
    if (xpu_event_query(event) == XPU_SUCCESS) {
      DestroyEvent(event);
      return true;
    } else {
      return false;
    }
  };
  std::unique_lock<std::mutex> lock(mtx_);
  while (!incomplished_events_.empty()) {
    XPUEvent event = incomplished_events_.front();
    if (!CheckComplishAndDestroy(event)) {
      LOG(ERROR) << "failed on destroying event when destroying event pool.";
    }
    incomplished_events_.pop();
  }
}

XPUEvent XPUEventPool::CreateEventFromPool() {
  std::unique_lock<std::mutex> lock(mtx_);

  const auto& CreateNewEvent = [&]() -> XPUEvent {
    XPUEvent new_event;
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_create(&new_event));
    incomplished_events_.push(new_event);
    return new_event;
  };

  const auto& CreateNewOrReuseEvent = [&]() -> XPUEvent {
    XPUEvent front_event = incomplished_events_.front();
    incomplished_events_.pop();
    incomplished_events_.push(front_event);
    if (xpu_event_query(front_event) == XPU_SUCCESS) {
      return front_event;
    }
    return CreateNewEvent();
  };

  if (incomplished_events_.empty()) {
    return CreateNewEvent();
  }
  return CreateNewOrReuseEvent();
}

XPUEventHandle::XPUEventHandle() {
  event_ = XPUEventPool::Instance().CreateEventFromPool();
}
XPUEventHandle::XPUEventHandle(XPUStream stream) {
  event_ = XPUEventPool::Instance().CreateEventFromPool();
  PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_record(event_, stream));
}

void XPUEventHandle::record(XPUStream stream) {
  PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_query(event_));
  PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_record(event_, stream));
}

bool XPUEventHandle::query() {
  int result = xpu_event_query(event_);
  if (result == XPU_SUCCESS) {
    return true;
  }
  return false;
}

void XPUEventHandle::synchronize() {
  PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_wait(event_));
}
#if defined(PADDLE_WITH_XPU)
XPUPinnedContext::XPUPinnedContext() {
  eigen_device_ = std::make_unique<Eigen::DefaultDevice>();
}

XPUPinnedContext::XPUPinnedContext(XPUPinnedPlace place) : place_(place) {
  eigen_device_ = std::make_unique<Eigen::DefaultDevice>();
}

Eigen::DefaultDevice* XPUPinnedContext::eigen_device() const {
  return eigen_device_.get();
}

const Place& XPUPinnedContext::GetPlace() const { return place_; }
#endif

}  // namespace phi
