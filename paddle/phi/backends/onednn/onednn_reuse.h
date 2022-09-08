/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace phi {
namespace funcs {

using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = dnnl::memory;

using OneDNNMemoryFormat = dnnl::memory::format_tag;

template <typename T,
          typename TForward,
          typename TBackward = onednn_dummy_primitive,
          typename TBackward_params = onednn_dummy_primitive>
class OneDNNHandlerT {
 public:
  OneDNNHandlerT(const OneDNNContext& dev_ctx,
                 dnnl::engine engine,
                 Place cpu_place,
                 const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        place_(cpu_place),
        key_common_(base_key),
        key_(ExtendKeyWithThreadInfoIfNeeded(dev_ctx, base_key)),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    OneDNNContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    const std::string key_p = key_ + "@fwd_p";
    auto forward_p =
        std::static_pointer_cast<TForward>(dev_ctx_.GetBlob(key_p));
    if (forward_p == nullptr) {
      forward_p = std::make_shared<TForward>(*fwd_pd_);
      dev_ctx_.SetBlob(key_p, forward_p);
    }
    return forward_p;
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    const std::string key_p = key_ + "@bwd_p";
    auto backward_p =
        std::static_pointer_cast<TBackward>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      backward_p = std::make_shared<TBackward>(*bwd_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    const std::string key_p = key_ + "@bwd_w_p";
    auto backward_p =
        std::static_pointer_cast<TBackward_params>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      PADDLE_ENFORCE_NOT_NULL(
          bwd_w_pd_,
          errors::Unavailable("BWD_PD should be set when "
                              "getting BWD prim witk key: %s .",
                              key_p));
      backward_p = std::make_shared<TBackward_params>(*bwd_w_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->src_desc(), to_void_cast<T>(input_data), "@src_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->dst_desc(), ptr, "@dst_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), "@dstt_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(const DenseTensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const DenseTensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_dst_desc(), to_void_cast<T>(ptr), "@diff_dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(DenseTensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_src_desc(), ptr, "@diff_src_mem_p");
  }

  // Buffer of given DenseTensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      DenseTensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        bwd_w_pd_->diff_weights_desc(), ptr, "@diff_wei_mem_p");
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(),
                                            "@diff_wei_mem_p");
  }

 protected:
  bool isCached() {
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    return (fwd_pd_ != nullptr);
  }

  bool isBwdCached() {
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    if (bwd_pd_ == nullptr) {
      return false;
    } else {
      if (std::is_same<TBackward_params, onednn_dummy_primitive>::value ==
          false) {
        const std::string key_bw_w_pd = key_ + "@bwd_w_pd";
        bwd_w_pd_ =
            std::static_pointer_cast<typename TBackward_params::primitive_desc>(
                dev_ctx_.GetBlob(key_bw_w_pd));
      }

      // When BWD is cached then still we need to Get FWD PD
      const std::string key_fpd = key_ + "@fwd_pd";
      fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
          dev_ctx_.GetBlob(key_fpd));
      PADDLE_ENFORCE_NOT_NULL(
          fwd_pd_,
          errors::Unavailable(
              "Error: FWD PD should be set when BWD PD is cached."));
      return true;
    }
  }

  // If your primitive descriptor requires attributes, pass them as a
  // first argument and paramters to descriptor constructor in the following
  // arguments. Otherwise, all arguments will be forwarded to descriptor
  // constructor, including the first one.
  template <typename Arg, typename... Args>
  void AcquireForwardPrimitiveDescriptor(Arg&& first_arg, Args&&... args) {
    // This is used when we can recreate FWD PD in BWD so
    // we do not need to pass FWD to BWD
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (fwd_pd_ == nullptr) {
      CreateForwardPrimitiveDescriptor(first_arg, std::forward<Args>(args)...);
      dev_ctx_.SetBlob(key_pd, fwd_pd_);
    }
  }

  // Using sfinae to specialise variadic function. Workaround for not having
  // if constexpr in C++ 11.
  template <class First, class... Args>
  typename std::enable_if<std::is_same<typename std::decay<First>::type,
                                       dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<Args>(args)...);
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        fwd_desc, first, engine_);
  }

  template <class First, class... Args>
  typename std::enable_if<!std::is_same<typename std::decay<First>::type,
                                        dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<First>(first),
                                            std::forward<Args>(args)...);
    fwd_pd_ =
        std::make_shared<typename TForward::primitive_desc>(fwd_desc, engine_);
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get OneDNN Forward primitive %s failed.",
                            key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (bwd_pd_ == nullptr) {
      auto bwd_desc = typename TBackward::desc(std::forward<Args>(args)...);
      bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
          bwd_desc, engine_, *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_pd_);
    }
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get OneDNN Forward primitive %s failed.",
                            key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_w_pd";
    bwd_w_pd_ =
        std::static_pointer_cast<typename TBackward_params::primitive_desc>(
            dev_ctx_.GetBlob(key_pd));
    if (bwd_w_pd_ == nullptr) {
      auto bwd_desc =
          typename TBackward_params::desc(std::forward<Args>(args)...);
      bwd_w_pd_ = std::make_shared<typename TBackward_params::primitive_desc>(
          bwd_desc, engine_, *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_w_pd_);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      const std::string& suffix) {
    return std::static_pointer_cast<dnnl::memory>(
        dev_ctx_.GetBlob(key_ + suffix));
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, void* ptr, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<dnnl::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<dnnl::memory>(md, engine_);
      dev_ctx_.SetBlob(local_key, mem_p);
    }
    return mem_p;
  }

  void AcquireReorder(const std::shared_ptr<dnnl::memory>& user_memory_p,
                      const std::shared_ptr<dnnl::memory>& target_memory_p) {
    auto reorder_p =
        std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();

    paddle::platform::RecordEvent record_reorder(
        "int_reorder",
        paddle::platform::TracerEventType::UserDefined,
        2,
        paddle::platform::EventRole::kUniqueOp);
    reorder_p->execute(
        astream,
        {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const std::string& suffix,
      bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {},
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    const auto target_key = key_ + suffix + "_target";
    const auto key_reorder_p = key_ + suffix + "reorder_p";
    const auto user_key = key_ + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      if (custom_reorder_func) {
        auto reordered_data =
            custom_reorder_func(reinterpret_cast<const F*>(ptr));
        dev_ctx_.SetBlob(key_reorder_p + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, engine_, ptr);
      if (user_md != target_md) {
        target_memory_p = std::make_shared<dnnl::memory>(target_md, engine_);
        dnnl::reorder::primitive_desc reorder_pdesc;
        if (is_int8<T>()) {
          dnnl::primitive_attr attr;
          attr.set_output_scales(mask, scale_data);
          reorder_pdesc = dnnl::reorder::primitive_desc(
              *user_memory_p, *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);

        auto& astream = OneDNNContext::tls().get_stream();
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            2,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx_.SetBlob(user_key, user_memory_p);
      dev_ctx_.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = OneDNNContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            2,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireMemory(const std::string& suffix) {
    const auto local_key = key_ + suffix;
    return std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
  }

  const OneDNNContext& dev_ctx_;
  dnnl::engine engine_;
  Place place_;
  std::string key_common_;
  std::string key_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T,
          typename TForward,
          typename TBackward = onednn_dummy_primitive,
          typename TBackward_params = onednn_dummy_primitive>
class OneDNNHandlerNoCachingT {
 public:
  OneDNNHandlerNoCachingT(dnnl::engine engine, Place cpu_place)
      : engine_(engine), place_(cpu_place), fwd_pd_(nullptr), bwd_pd_(nullptr) {
    OneDNNContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    return std::make_shared<TForward>(*fwd_pd_);
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    return std::make_shared<TBackward>(*bwd_pd_);
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    PADDLE_ENFORCE_NOT_NULL(bwd_w_pd_,
                            errors::Unavailable("BWD_PD should be set when "
                                                "getting BWD prim ."));
    return std::make_shared<TBackward_params>(*bwd_w_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), ptr);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc());
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(const DenseTensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const DenseTensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_dst_desc(),
                                            to_void_cast<T>(ptr));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(DenseTensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr);
  }

  // Buffer of given DenseTensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      DenseTensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(),
                                            ptr);
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc());
  }

 protected:
  // If your primitive descriptor requires attributes, pass them as a
  // first argument and paramters to descriptor constructor in the following
  // arguments. Otherwise, all arguments will be forwarded to descriptor
  // constructor, including the first one.
  template <typename Arg, typename... Args>
  void AcquireForwardPrimitiveDescriptor(Arg&& first_arg, Args&&... args) {
    CreateForwardPrimitiveDescriptor(first_arg, std::forward<Args>(args)...);
  }

  // Using sfinae to specialise variadic function. Workaround for not having
  // if constexpr in C++ 11.
  template <class First, class... Args>
  typename std::enable_if<std::is_same<typename std::decay<First>::type,
                                       dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<Args>(args)...);
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        fwd_desc, first, engine_);
  }

  template <class First, class... Args>
  typename std::enable_if<!std::is_same<typename std::decay<First>::type,
                                        dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<First>(first),
                                            std::forward<Args>(args)...);
    fwd_pd_ =
        std::make_shared<typename TForward::primitive_desc>(fwd_desc, engine_);
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get oneDNN Forward primitive %s failed."));
    auto bwd_desc = typename TBackward::desc(std::forward<Args>(args)...);
    bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
        bwd_desc, engine_, *fwd_pd_);
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get oneDNN Forward primitive %s failed."));
    auto bwd_desc =
        typename TBackward_params::desc(std::forward<Args>(args)...);
    bwd_w_pd_ = std::make_shared<typename TBackward_params::primitive_desc>(
        bwd_desc, engine_, *fwd_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, void* ptr) {
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md) {
    return std::make_shared<dnnl::memory>(md, engine_);
  }

  void AcquireReorder(const std::shared_ptr<dnnl::memory>& user_memory_p,
                      const std::shared_ptr<dnnl::memory>& target_memory_p) {
    auto reorder_p =
        std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();

    paddle::platform::RecordEvent record_reorder(
        "int_reorder",
        paddle::platform::TracerEventType::UserDefined,
        2,
        paddle::platform::EventRole::kUniqueOp);
    reorder_p->execute(
        astream,
        {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {}) {
    std::shared_ptr<dnnl::memory> target_memory_p;
    if (custom_reorder_func) {
      auto reordered_data =
          custom_reorder_func(reinterpret_cast<const F*>(ptr));
      ptr = reinterpret_cast<void*>(reordered_data.get());
    }
    auto user_memory_p = std::make_shared<dnnl::memory>(user_md, engine_, ptr);
    if (user_md != target_md) {
      target_memory_p = std::make_shared<dnnl::memory>(target_md, engine_);
      auto reorder_p =
          std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

      auto& astream = OneDNNContext::tls().get_stream();
      paddle::platform::RecordEvent record_reorder(
          "int_reorder",
          paddle::platform::TracerEventType::UserDefined,
          2,
          paddle::platform::EventRole::kUniqueOp);
      reorder_p->execute(
          astream,
          {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
      astream.wait();
    } else {
      target_memory_p = user_memory_p;
    }
    return target_memory_p;
  }

  dnnl::engine engine_;
  Place place_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T>
class ActivationOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ActivationOneDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm,
                                            x->mem_desc(),
                                            alpha,
                                            beta);
  }

  ActivationOneDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x,
                          const DenseTensor* dout)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm,
                                            x->mem_desc(),
                                            alpha,
                                            beta);
    this->AcquireBackwardPrimitiveDescriptor(
        algorithm, dout->mem_desc(), x->mem_desc(), alpha, beta);
  }

  std::shared_ptr<dnnl::memory> AcquireBackwardSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }
};

class ReorderOneDNNHandler {
 public:
  ReorderOneDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       DataType ptype,
                       dnnl::memory::data_type dtype,
                       dnnl::engine engine)
      : dims_(dims),
        ptype_(ptype),
        ptype_dst_(ptype),
        dtype_(dtype),
        dtype_dst_(dtype),
        engine_(engine) {}

  ReorderOneDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       DataType ptype,
                       dnnl::memory::data_type dtype,
                       DataType ptype_dst,
                       dnnl::memory::data_type dtype_dst,
                       dnnl::engine engine)
      : dims_(dims),
        ptype_(ptype),
        ptype_dst_(ptype_dst),
        dtype_(dtype),
        dtype_dst_(dtype_dst),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const dnnl::memory::desc& md,
                                                 void* ptr) {
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const OneDNNMemoryFormat& fmt,
                                                 void* ptr) {
    auto md = dnnl::memory::desc(dims_, dtype_, fmt);
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSubmemory(
      const std::vector<int64_t>& dims,
      const std::vector<int64_t>& offset,
      const std::shared_ptr<dnnl::memory>& mem_p) {
    auto sub_md = mem_p->get_desc().submemory_desc(dims, {offset});
    auto sub_mem_p = std::make_shared<dnnl::memory>(
        sub_md, engine_, mem_p->get_data_handle());
    return sub_mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output,
                                                 const OneDNNMemoryFormat& fmt,
                                                 Place place) {
    auto dst_md = OneDNNMemDesc(dims_, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(place, ptype_dst_, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      DenseTensor* output, const dnnl::memory::desc& src_md, Place place) {
    if (ptype_dst_ == ptype_) {
      auto dst_data =
          output->mutable_data(place, ptype_dst_, src_md.get_size());
      return std::make_shared<dnnl::memory>(src_md, engine_, dst_data);
    } else {
      auto dst_md = src_md;
      dst_md.data.data_type = static_cast<dnnl_data_type_t>(dtype_dst_);
      auto dst_data =
          output->mutable_data(place, ptype_dst_, dst_md.get_size());
      return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      DenseTensor* output,
      const std::vector<int64_t>& dims,
      const OneDNNMemoryFormat& fmt,
      Place place) {
    auto dst_md = OneDNNMemDesc(dims, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(place, ptype_dst_, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    return std::make_shared<dnnl::reorder>(*(src_memory_p), *(dst_memory_p));
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p,
      const dnnl::primitive_attr& attrs) {
    return std::make_shared<dnnl::reorder>(
        *(src_memory_p), *(dst_memory_p), attrs);
  }

 private:
  std::vector<int64_t> dims_;
  DataType ptype_, ptype_dst_;
  dnnl::memory::data_type dtype_, dtype_dst_;
  dnnl::engine engine_;
};

template <typename T>
class BinaryOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  BinaryOneDNNHandler(const dnnl::algorithm algo,
                      const int axis,
                      const dnnl::engine engine,
                      Place cpu_place,
                      const DenseTensor* x,
                      const DenseTensor* y,
                      DenseTensor* out,
                      float scale_x,
                      float scale_y,
                      float scale_out,
                      const dnnl::post_ops& post_ops = dnnl::post_ops{})
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src_x_tz = vectorize(x->dims());
    const auto src_y_tz = vectorize(y->dims());
    // if output tensor(z) is nullptr then we are computing into oneDNN
    // managed buffer
    auto rankdiff = x->dims().size() - y->dims().size();
    const auto dst_tz = (out == nullptr) ? (rankdiff > 0 ? src_x_tz : src_y_tz)
                                         : vectorize(out->dims());

    auto src0_md = x->mem_desc();
    auto src1_md = y->mem_desc();
    if (rankdiff > 0) {  // Second input is of smaller rank than first
      std::vector<int64_t> dims1_ex(rankdiff, 1);
      dims1_ex.insert(next(dims1_ex.begin(), (axis == -1 ? rankdiff : axis)),
                      src_y_tz.begin(),
                      src_y_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (OneDNNContext::tls().get_cur_paddle_data_layout() ==
          DataLayout::kNHWC) {
        std::rotate(dims1_ex.begin() + 1, dims1_ex.end() - 1, dims1_ex.end());
      }
      src1_md = src1_md.reshape(dims1_ex);
    } else if (rankdiff < 0) {  // First input is of smaller than second
      std::vector<int64_t> dims0_ex(-rankdiff, 1);
      dims0_ex.insert(next(dims0_ex.begin(), (axis == -1 ? -rankdiff : axis)),
                      src_x_tz.begin(),
                      src_x_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (OneDNNContext::tls().get_cur_paddle_data_layout() ==
          DataLayout::kNHWC) {
        std::rotate(dims0_ex.begin() + 1, dims0_ex.end() - 1, dims0_ex.end());
      }
      src0_md = src0_md.reshape(dims0_ex);
    }
    const auto dst_md =
        memory::desc(dst_tz, oneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    auto attributes =
        CreateAttributes(algo, scale_x, scale_y, scale_out, post_ops);

    if (x->numel() < y->numel()) {
      this->AcquireForwardPrimitiveDescriptor(
          attributes, algo, src1_md, src0_md, dst_md);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          attributes, algo, src0_md, src1_md, dst_md);
    }
  }
  std::shared_ptr<dnnl::memory> AcquireSecondSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src1_desc(),
                                            to_void_cast<T>(input_data));
  }

 private:
  static inline dnnl::primitive_attr CreateAttributes(
      dnnl::algorithm op,
      float scale_x,
      float scale_y,
      float scale_out,
      dnnl::post_ops post_ops = dnnl::post_ops{}) {
    // Scales set in attributes for inputs contibute to the output equation
    // in the following way (assuming no broadcasting takes place):
    // output_i = scale_0 * x_i <+ or *> scale_1 * y_i;
    // Hence we have to create scales that will:
    // 1. Dequantize both values, by multiplying with (1.0 / scale_x_or_y)
    // 2. Quantize their result to output scale range, by multiplying with
    // (scale_z)
    // If we combine these two, we end up with following equation
    // output = scale_out * (1/scale_x * x <* or +> 1/scale_y * y)
    // Hence, to mimic such behaviour using provided interface,
    // For add operation the equation is equal to:
    // output = (scale_out / scale_x) * x + (scale_out / scale_y) * y
    //                <scale_0>                  <scale_1>
    // For mul operation on the other hand
    // output = (scale_out / scale_x) * x * (1.0 / scale_y) * y
    //                <scale_0>                 <scale_1>
    float scale_0 = scale_out / scale_x;
    float scale_1 =
        op == dnnl::algorithm::binary_add ? scale_out / scale_y : 1.0 / scale_y;
    dnnl::primitive_attr attributes;
    attributes.set_scales(
        /* input_x_id = */ DNNL_ARG_SRC_0, /* mask = */ 0, {scale_0});
    attributes.set_scales(
        /* input_y_id = */ DNNL_ARG_SRC_1, /* mask = */ 0, {scale_1});
    if (post_ops.len() > 0) attributes.set_post_ops(post_ops);
    return attributes;
  }
};

template <typename T>
class BroadcastDataOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  BroadcastDataOneDNNHandler(const dnnl::algorithm algo,
                             const dnnl::engine engine,
                             Place cpu_place,
                             const DenseTensor* x,
                             DenseTensor* out,
                             float scale_x,
                             float scale_y,
                             const std::vector<int64_t>& extended_x_dims)
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src0_tz = vectorize(out->dims());
    const auto src0_md = dnnl::memory::desc(
        src0_tz, oneDNNGetDataType<T>(), GetPlainOneDNNFormat(src0_tz.size()));
    const auto src1_md = x->mem_desc().reshape(extended_x_dims);

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_SRC_0, 0, {scale_x});
    attributes.set_scales(DNNL_ARG_SRC_1, 0, {scale_y});

    this->AcquireForwardPrimitiveDescriptor(
        attributes, algo, src0_md, src1_md, src0_md);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireZeroedDstMemory(DenseTensor* out) {
    T_out* ptr = out->mutable_data<T_out>(this->place_,
                                          this->fwd_pd_->dst_desc().get_size());
    memset(ptr, 0, this->fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

template <typename T>
class ReductionOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::reduction> {
 public:
  ReductionOneDNNHandler(const dnnl::algorithm algo,
                         const float p,
                         const float eps,
                         const dnnl::engine engine,
                         Place cpu_place,
                         const DenseTensor* x,
                         const DenseTensor* out,
                         std::vector<int64_t> out_tz,
                         const dnnl::primitive_attr& attrs = NULL)
      : OneDNNHandlerNoCachingT<T, dnnl::reduction>(engine, cpu_place) {
    const auto out_md = memory::desc(
        out_tz, oneDNNGetDataType<T>(), dnnl::memory::format_tag::any);

    if (attrs)
      this->AcquireForwardPrimitiveDescriptor(
          attrs, algo, x->mem_desc(), out_md, p, eps);
    else
      this->AcquireForwardPrimitiveDescriptor(
          algo, x->mem_desc(), out_md, p, eps);
  }
};
}  // namespace funcs
}  // namespace phi
