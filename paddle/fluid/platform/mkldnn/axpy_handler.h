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

#include "mkldnn.hpp"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace platform {

template <typename T>
class AXPYMKLDNNHandler : public platform::MKLDNNHandlerT<T, dnnl::reorder> {
 public:
  AXPYMKLDNNHandler(const platform::MKLDNNDeviceContext& dev_ctx,
                    const mkldnn::engine mkldnn_engine,
                    platform::Place cpu_place, int n, float alpha)
      : platform::MKLDNNHandlerT<T, dnnl::reorder>(
            dev_ctx, mkldnn_engine, cpu_place,
            platform::CreateKey(dev_ctx, static_cast<int64_t>(n),
                                platform::MKLDNNGetDataType<T>(), alpha,
                                "-axpy")),
        alpha_(alpha),
        n_(n) {}

  std::shared_ptr<mkldnn::memory> AcquireMemory(void* ptr,
                                                const std::string& suffix) {
    /*Generate key*/
    auto local_key = this->key_ + suffix;
    auto mem_p = std::static_pointer_cast<mkldnn::memory>(
        this->dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto md = mkldnn::memory::desc({n_}, platform::MKLDNNGetDataType<T>(),
                                     dnnl::memory::format_tag::x);
      mem_p = std::make_shared<mkldnn::memory>(md, this->engine_, ptr);
      this->dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(const T* x) {
    return this->AcquireMemory(platform::to_void_cast(x), "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(T* y) {
    return this->AcquireMemory(y, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::reorder> AcquireReorder(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    auto prim_key = this->key_ + "@reorder_p";
    auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
        this->dev_ctx_.GetBlob(prim_key));
    if (reorder_p == nullptr) {
      // Here we pass Postops to mimick y -> a*X + y
      mkldnn::primitive_attr reorder_attr;
      mkldnn::post_ops post_operations;
      if (this->alpha_ != 1.f) {
        std::vector<float> scales(1, this->alpha_);
        reorder_attr.set_output_scales(0, scales);
      }
      post_operations.append_sum(1.0f);

      reorder_attr.set_post_ops(post_operations);
      reorder_p = std::make_shared<mkldnn::reorder>(
          *(src_memory_p), *(dst_memory_p), reorder_attr);
      this->dev_ctx_.SetBlob(prim_key, reorder_p);
    }
    return reorder_p;
  }

 private:
  float alpha_;
  int n_;
};

}  // namespace platform
}  // namespace paddle
