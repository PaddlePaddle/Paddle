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

#include <cinttypes>
#include <memory>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "paddle/fluid/operators/mkldnn/axpy_handler.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

namespace plat = paddle::platform;

namespace {

template <typename T>
class AXPYHandler {
 public:
  AXPYHandler(const dnnl::engine mkldnn_engine, int n, float alpha) {
    platform::MKLDNNDeviceContext::tls().log_lib_version();
    auto md = dnnl::memory::desc({n}, plat::MKLDNNGetDataType<T>(),
                                 dnnl::memory::format_tag::x);
    src_mem_ = dnnl::memory(md, mkldnn_engine, DNNL_MEMORY_NONE);
    dst_mem_ = dnnl::memory(md, mkldnn_engine, DNNL_MEMORY_NONE);
    dnnl::primitive_attr reorder_attr;
    dnnl::post_ops post_operations;
    if (alpha != 1.f) {
      std::vector<float> scales(1, alpha);
      reorder_attr.set_output_scales(0, scales);
    }
    post_operations.append_sum(1.0f);

    reorder_attr.set_post_ops(post_operations);
    reorder_p_ = dnnl::reorder(src_mem_, dst_mem_, reorder_attr);
  }

  dnnl::memory &AcquireSrcMemory(const T *x) {
    src_mem_.set_data_handle(plat::to_void_cast<T>(x));
    return src_mem_;
  }

  dnnl::memory &AcquireDstMemory(T *y) {
    dst_mem_.set_data_handle(y);
    return dst_mem_;
  }

  const dnnl::reorder &AcquireReorder() { return reorder_p_; }

 private:
  dnnl::memory src_mem_;
  dnnl::memory dst_mem_;
  dnnl::reorder reorder_p_;
};

template class AXPYHandler<float>;
template class AXPYHandler<plat::bfloat16>;

template <typename T>
static void naive_axpy(int n, T alpha, const T *x, T *y) {
  while (n-- > 0) {
    *y += alpha * *x;
    ++y;
    ++x;
  }
}

}  // anonnymouse namespace

template <typename T>
class OneDNNAXPYHandler<T>::Impl {
 public:
  Impl(int64_t n, T alpha);
  void operator()(const T *x, T *y);

 private:
  std::unique_ptr<AXPYHandler<T>> handler_;
  int64_t n_;
  T alpha_;
};

template <typename T>
OneDNNAXPYHandler<T>::Impl::Impl(int64_t n, T alpha) : n_{n}, alpha_{alpha} {
  auto &pool = plat::DeviceContextPool::Instance();
  auto cpu_place = plat::CPUPlace();
  auto *dev_ctx =
      dynamic_cast<plat::MKLDNNDeviceContext *>(pool.Get(cpu_place));
  auto &cpu_engine = dev_ctx->GetEngine();
  handler_ = std::make_unique<AXPYHandler<T>>(cpu_engine, n,
                                              static_cast<float>(alpha));
}

template <typename T>
void OneDNNAXPYHandler<T>::Impl::operator()(const T *x, T *y) {
  if (this->n_ < 100) {
    naive_axpy(this->n_, this->alpha_, x, y);
    return;
  }

  auto &reorder_src_mem_p = handler_->AcquireSrcMemory(x);
  auto &reorder_dst_mem_p = handler_->AcquireDstMemory(y);
  auto reorder_p = handler_->AcquireReorder();
  auto &astream = plat::MKLDNNDeviceContext::tls().get_stream();
  reorder_p.execute(astream, reorder_src_mem_p, reorder_dst_mem_p);
  astream.wait();
}

template <typename T>
OneDNNAXPYHandler<T>::OneDNNAXPYHandler(int64_t n, T alpha)
    : pimpl_{new Impl{n, alpha}, [](Impl *impl) { delete impl; }} {
  VLOG(4) << "[OneDNN] OneDNNAXPYHandler<" << typeid(T).name() << ">, "
          << "n: " << n << ", alpha: " << alpha;
}

template <typename T>
void OneDNNAXPYHandler<T>::operator()(const T *x, T *y) {
  pimpl_->operator()(x, y);
}

template class OneDNNAXPYHandler<float>;
template class OneDNNAXPYHandler<plat::bfloat16>;

}  // namespace operators
}  // namespace paddle
