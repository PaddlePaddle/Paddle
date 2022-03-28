/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include "boost/optional.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

using framework::DataLayout;
using framework::Tensor;
using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = dnnl::memory;

template <typename T, typename TForward,
          typename TBackward = mkldnn_dummy_primitive,
          typename TBackward_params = mkldnn_dummy_primitive>
class MKLDNNHandlerNoCachingT {
 public:
  MKLDNNHandlerNoCachingT(dnnl::engine engine, platform::Place cpu_place)
      : engine_(engine), place_(cpu_place), fwd_pd_(nullptr), bwd_pd_(nullptr) {
    platform::MKLDNNDeviceContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    return std::make_shared<TForward>(*fwd_pd_);
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    return std::make_shared<TBackward>(*bwd_pd_);
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_, platform::errors::Unavailable("BWD_PD should be set when "
                                                 "getting BWD prim ."));
    return std::make_shared<TBackward_params>(*bwd_w_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), ptr);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc());
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      const framework::Tensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const framework::Tensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_dst_desc(),
                                            to_void_cast<T>(ptr));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(
      framework::Tensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr);
  }

  // Buffer of given Tensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      framework::Tensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        platform::errors::Unavailable(
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
        platform::errors::Unavailable(
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
    PADDLE_ENFORCE_NOT_NULL(fwd_pd_,
                            platform::errors::Unavailable(
                                "Get MKLDNN Forward primitive %s failed."));
    auto bwd_desc = typename TBackward::desc(std::forward<Args>(args)...);
    bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
        bwd_desc, engine_, *fwd_pd_);
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(fwd_pd_,
                            platform::errors::Unavailable(
                                "Get MKLDNN Forward primitive %s failed."));
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

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    platform::RecordEvent record_reorder("int_reorder",
                                         platform::TracerEventType::UserDefined,
                                         2, platform::EventRole::kUniqueOp);
    reorder_p->execute(astream, {{DNNL_ARG_FROM, *user_memory_p},
                                 {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md, const dnnl::memory::desc& target_md,
      void* ptr, bool is_persistent = false,
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

      auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
      platform::RecordEvent record_reorder(
          "int_reorder", platform::TracerEventType::UserDefined, 2,
          platform::EventRole::kUniqueOp);
      reorder_p->execute(astream, {{DNNL_ARG_FROM, *user_memory_p},
                                   {DNNL_ARG_TO, *target_memory_p}});
      astream.wait();
    } else {
      target_memory_p = user_memory_p;
    }
    return target_memory_p;
  }

  dnnl::engine engine_;
  platform::Place place_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T, typename TForward,
          typename TBackward = mkldnn_dummy_primitive,
          typename TBackward_params = mkldnn_dummy_primitive>
class MKLDNNHandlerT {
 public:
  MKLDNNHandlerT(const MKLDNNDeviceContext& dev_ctx, dnnl::engine engine,
                 platform::Place cpu_place, const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        place_(cpu_place),
        key_common_(base_key),
        key_(platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, base_key)),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    platform::MKLDNNDeviceContext::tls().log_lib_version();
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
      PADDLE_ENFORCE_NOT_NULL(bwd_w_pd_, platform::errors::Unavailable(
                                             "BWD_PD should be set when "
                                             "getting BWD prim witk key: %s .",
                                             key_p));
      backward_p = std::make_shared<TBackward_params>(*bwd_w_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->src_desc(), to_void_cast<T>(input_data), "@src_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), ptr,
                                            "@dst_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), "@dstt_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      const framework::Tensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const framework::Tensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_dst_desc(), to_void_cast<T>(ptr), "@diff_dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(
      framework::Tensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr,
                                            "@diff_src_mem_p");
  }

  // Buffer of given Tensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      framework::Tensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        platform::errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(), ptr,
                                            "@diff_wei_mem_p");
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        platform::errors::Unavailable(
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
      if (std::is_same<TBackward_params, mkldnn_dummy_primitive>::value ==
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
          fwd_pd_, platform::errors::Unavailable(
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
        platform::errors::Unavailable("Get MKLDNN Forward primitive %s failed.",
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
        platform::errors::Unavailable("Get MKLDNN Forward primitive %s failed.",
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

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    platform::RecordEvent record_reorder("int_reorder",
                                         platform::TracerEventType::UserDefined,
                                         2, platform::EventRole::kUniqueOp);
    reorder_p->execute(astream, {{DNNL_ARG_FROM, *user_memory_p},
                                 {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md, const dnnl::memory::desc& target_md,
      void* ptr, const std::string& suffix, bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {},
      const std::vector<float>& scale_data = {1.0f}, int mask = 0) {
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
          reorder_pdesc = dnnl::reorder::primitive_desc(*user_memory_p,
                                                        *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);

        auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
        platform::RecordEvent record_reorder(
            "int_reorder", platform::TracerEventType::UserDefined, 2,
            platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{DNNL_ARG_FROM, *user_memory_p},
                                     {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx_.SetBlob(user_key, user_memory_p);
      dev_ctx_.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        platform::RecordEvent record_reorder(
            "int_reorder", platform::TracerEventType::UserDefined, 2,
            platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{DNNL_ARG_FROM, *user_memory_p},
                                     {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireMemory(const std::string& suffix) {
    const auto local_key = key_ + suffix;
    return std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
  }

  const MKLDNNDeviceContext& dev_ctx_;
  dnnl::engine engine_;
  platform::Place place_;
  std::string key_common_;
  std::string key_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T>
class BinaryMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  BinaryMKLDNNHandler(const dnnl::algorithm algo, const int axis,
                      const dnnl::engine engine, platform::Place cpu_place,
                      const Tensor* x, const Tensor* y, Tensor* z,
                      float scale_x, float scale_y, float scale_z,
                      const dnnl::post_ops& post_ops = dnnl::post_ops{})
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    PADDLE_ENFORCE_EQ(
        x->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "Wrong layout set for X tensor. Expected: %d (kMKLDNN), Actual: %d",
            DataLayout::kMKLDNN, x->layout()));
    PADDLE_ENFORCE_NE(x->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Wrong format set for X tensor : %d (undef)",
                          static_cast<unsigned int>(x->format())));

    PADDLE_ENFORCE_EQ(
        y->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "Wrong layout set for Y tensor. Expected: %d (kMKLDNN), Actual: %d",
            DataLayout::kMKLDNN, y->layout()));
    PADDLE_ENFORCE_NE(y->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Wrong format set for Y tensor : %d (undef)",
                          static_cast<unsigned int>(y->format())));

    const auto src_x_tz = phi::vectorize(x->dims());
    const auto src_y_tz = phi::vectorize(y->dims());
    // if output tensor(z) is nullptr then we are computing into oneDNN
    // managed buffer
    auto rankdiff = x->dims().size() - y->dims().size();
    const auto dst_tz = (z == nullptr) ? (rankdiff > 0 ? src_x_tz : src_y_tz)
                                       : phi::vectorize(z->dims());

    auto src0_md = dnnl::memory::desc(
        src_x_tz, platform::MKLDNNGetDataType<T>(), x->format());
    auto src1_md = dnnl::memory::desc(
        src_y_tz, platform::MKLDNNGetDataType<T>(), y->format());
    if (rankdiff > 0) {  // Second input is of smaller rank than first
      std::vector<int64_t> dims1_ex(rankdiff, 1);
      dims1_ex.insert(next(dims1_ex.begin(), (axis == -1 ? rankdiff : axis)),
                      src_y_tz.begin(), src_y_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (MKLDNNDeviceContext::tls().get_cur_paddle_data_layout() ==
          framework::DataLayout::kNHWC) {
        std::rotate(dims1_ex.begin() + 1, dims1_ex.end() - 1, dims1_ex.end());
      }
      src1_md = src1_md.reshape(dims1_ex);
    } else if (rankdiff < 0) {  // First input is of smaller than second
      std::vector<int64_t> dims0_ex(-rankdiff, 1);
      dims0_ex.insert(next(dims0_ex.begin(), (axis == -1 ? -rankdiff : axis)),
                      src_x_tz.begin(), src_x_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (MKLDNNDeviceContext::tls().get_cur_paddle_data_layout() ==
          framework::DataLayout::kNHWC) {
        std::rotate(dims0_ex.begin() + 1, dims0_ex.end() - 1, dims0_ex.end());
      }
      src0_md = src0_md.reshape(dims0_ex);
    }
    const auto dst_md = memory::desc(dst_tz, platform::MKLDNNGetDataType<T>(),
                                     MKLDNNMemoryFormat::any);

    auto attributes =
        CreateAttributes(algo, scale_x, scale_y, scale_z, post_ops);

    this->AcquireForwardPrimitiveDescriptor(attributes, algo, src0_md, src1_md,
                                            dst_md);
  }
  std::shared_ptr<dnnl::memory> AcquireSecondSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src1_desc(),
                                            to_void_cast<T>(input_data));
  }

 private:
  static inline dnnl::primitive_attr CreateAttributes(
      dnnl::algorithm op, float scale_x, float scale_y, float scale_z,
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
    float scale_0 = scale_z / scale_x;
    float scale_1 =
        op == dnnl::algorithm::binary_add ? scale_z / scale_y : 1.0 / scale_y;
    dnnl::primitive_attr attributes;
    attributes.set_scales(/* input_x_id = */ DNNL_ARG_SRC_0, /* mask = */ 0,
                          {scale_0});
    attributes.set_scales(/* input_y_id = */ DNNL_ARG_SRC_1, /* mask = */ 0,
                          {scale_1});
    if (post_ops.len() > 0) attributes.set_post_ops(post_ops);
    return attributes;
  }
};

template <typename T>
class BroadcastDataMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  BroadcastDataMKLDNNHandler(const dnnl::algorithm algo,
                             const dnnl::engine engine,
                             platform::Place cpu_place, const Tensor* out,
                             const Tensor* x, float scale_x, float scale_y,
                             const std::vector<int64_t>& input_dims)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    PADDLE_ENFORCE_EQ(
        x->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for X tensor."));
    PADDLE_ENFORCE_NE(
        x->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for X tensor."));

    const auto src0_tz = phi::vectorize(out->dims());

    const auto src0_md = dnnl::memory::desc(
        src0_tz, platform::MKLDNNGetDataType<T>(), out->format());
    const auto src1_md = dnnl::memory::desc(
        input_dims, platform::MKLDNNGetDataType<T>(), out->format());

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_SRC_0, 0, {scale_x});
    attributes.set_scales(DNNL_ARG_SRC_1, 0, {scale_y});

    this->AcquireForwardPrimitiveDescriptor(attributes, algo, src0_md, src1_md,
                                            src0_md);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr = output->mutable_data<T_out>(
        this->place_, this->fwd_pd_->dst_desc().get_size());
    memset(ptr, 0, this->fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

template <typename T>
class ReductionMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::reduction> {
 public:
  ReductionMKLDNNHandler(const dnnl::algorithm algo, const float p,
                         const float eps, const dnnl::engine engine,
                         platform::Place cpu_place, const Tensor* x,
                         const Tensor* y, std::vector<int64_t> y_tz,
                         const dnnl::primitive_attr& attr = NULL)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::reduction>(engine,
                                                              cpu_place) {
    PADDLE_ENFORCE_EQ(
        x->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for X tensor."));
    PADDLE_ENFORCE_NE(
        x->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for X tensor."));

    const auto x_tz = phi::vectorize(x->dims());

    const auto x_md =
        dnnl::memory::desc(x_tz, platform::MKLDNNGetDataType<T>(), x->format());
    const auto y_md =
        memory::desc(y_tz, platform::MKLDNNGetDataType<T>(), x->format());

    if (attr)
      this->AcquireForwardPrimitiveDescriptor(attr, algo, x_md, y_md, p, eps);
    else
      this->AcquireForwardPrimitiveDescriptor(algo, x_md, y_md, p, eps);
  }
};

template <typename T>
class MatMulV2MKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::matmul> {
 public:
  MatMulV2MKLDNNHandler(const dnnl::engine engine,
                        paddle::platform::Place cpu_place,
                        const std::vector<int64_t>& x_org_dims, bool trans_x,
                        const std::vector<int64_t>& y_org_dims, bool trans_y,
                        bool is_output_fused,
                        const std::vector<int64_t>& x_strides_override,
                        const std::vector<int64_t>& y_strides_override)
      : paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::matmul>(engine,
                                                                   cpu_place) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = x_dims.size() - 3;
    const int H_idx = x_dims.size() - 2;
    const int W_idx = x_dims.size() - 1;

    if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
    if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

    const memory::dim M = x_dims[H_idx];
    const memory::dim K = x_dims[W_idx];
    const memory::dim N = y_dims[W_idx];

    std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

    x_strides.reserve(x_dims.size());
    y_strides.reserve(x_dims.size());
    out_strides.reserve(x_dims.size());

    if (!x_strides_override.empty()) {
      x_strides = x_strides_override;
    } else {
      if (!trans_x) {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      } else {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      }
    }

    if (!y_strides_override.empty()) {
      y_strides = y_strides_override;
    } else {
      if (!trans_y) {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      } else {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      }
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = x_dims.size() - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      if (x_strides_override.empty()) {
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      }
      if (y_strides_override.empty()) {
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
      }
      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    if (is_output_fused) {
      out_strides = FakeTransposeStrides(out_ddims);
    }

    auto x_md = memory::desc(x_dims, MKLDNNGetDataType<T>(), x_strides);
    auto y_md = memory::desc(y_dims, MKLDNNGetDataType<T>(), y_strides);
    auto out_md = memory::desc(out_ddims, MKLDNNGetDataType<T>(), out_strides);

    this->AcquireForwardPrimitiveDescriptor(x_md, y_md, out_md);
  }

  std::vector<int64_t> FakeTransposeStrides(
      const std::vector<int64_t>& matmul_out_dims) const {
    // fuse matmul_v2 + transpose + reshape guarantees that output is 4D and
    // transpose axis are: {0, 2, 1, 3}
    std::vector<int64_t> transpose_axis = {0, 2, 1, 3};
    std::vector<int64_t> fake_strides(transpose_axis.size());
    int ndims = static_cast<int>(transpose_axis.size());

    int total_stride = 1;

    for (int i = ndims - 1; i >= 0; --i) {
      fake_strides[transpose_axis[i]] = total_stride;
      total_stride *= matmul_out_dims[transpose_axis[i]];
    }

    return fake_strides;
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<T>(input_data));
  }
};

template <typename T>
class ActivationMKLDNNHandler
    : public MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ActivationMKLDNNHandler(dnnl::algorithm algorithm,
                          const framework::ExecutionContext& ctx,
                          const dnnl::engine engine, Place cpu_place,
                          const framework::Tensor* in_x)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                          dnnl::eltwise_backward>(engine,
                                                                  cpu_place) {
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 0;
    float beta = ctx.HasAttr("beta") ? ctx.Attr<float>("beta") : 0;

    if (ctx.Type() == "scale") {
      bool bias_after_scale = ctx.Attr<bool>("bias_after_scale");
      auto* scale_tensor = ctx.Input<Tensor>("ScaleTensor");
      alpha = (scale_tensor == nullptr)
                  ? ctx.Attr<float>("scale")
                  : static_cast<float>(*(scale_tensor->data<T>()));
      beta = ctx.Attr<float>("bias");
      // if bias_after_scale == true
      //   out = scale*X + bias
      // else
      //   out = scale*(X + bias) = scale*X + scale*bias
      if (!bias_after_scale) {
        beta *= alpha;
      }
    } else if (ctx.Type() == "clip") {
      alpha = ctx.HasInput("Min") ? ctx.Input<Tensor>("Min")->data<float>()[0]
                                  : ctx.Attr<float>("min");
      beta = ctx.HasInput("Max") ? ctx.Input<Tensor>("Max")->data<float>()[0]
                                 : ctx.Attr<float>("max");
    } else {
      // paddle uses beta but mkldnn uses alpha for swish
      if (algorithm == dnnl::algorithm::eltwise_swish) {
        std::swap(alpha, beta);
      } else if (algorithm == dnnl::algorithm::eltwise_bounded_relu) {
        alpha = ctx.Attr<float>("threshold");
      }
    }

    PADDLE_ENFORCE(in_x->dims().size() >= 1 || in_x->dims().size() <= 6,
                   platform::errors::Unimplemented(
                       "Input dimension size can be 1, 2, 3, 4, "
                       "5, or 6, but now the dimension size is",
                       in_x->dims().size()));

    auto src_tz = phi::vectorize<int64_t>(in_x->dims());
    auto src_fmt = src_tz.size() == 2 ? MKLDNNMemoryFormat::nc : in_x->format();
    auto md =
        dnnl::memory::desc(src_tz, platform::MKLDNNGetDataType<T>(), src_fmt);

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm, md, alpha, beta);
  }

  ActivationMKLDNNHandler(dnnl::algorithm algorithm,
                          const framework::ExecutionContext& ctx,
                          const dnnl::engine engine, Place cpu_place,
                          const framework::Tensor* in_x, const Tensor* out_grad)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                          dnnl::eltwise_backward>(engine,
                                                                  cpu_place) {
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 0;
    float beta = ctx.HasAttr("beta") ? ctx.Attr<float>("beta") : 0;

    // paddle uses beta but mkldnn uses alpha for swish
    if (algorithm == dnnl::algorithm::eltwise_swish) {
      std::swap(alpha, beta);
    } else if (algorithm == dnnl::algorithm::eltwise_bounded_relu) {
      alpha = ctx.Attr<float>("threshold");
    }

    if (ctx.Type() == "clip_grad") {
      alpha = ctx.HasInput("Min") ? ctx.Input<Tensor>("Min")->data<float>()[0]
                                  : ctx.Attr<float>("min");
      beta = ctx.HasInput("Max") ? ctx.Input<Tensor>("Max")->data<float>()[0]
                                 : ctx.Attr<float>("max");
    }

    auto diff_dst_tz = phi::vectorize<int64_t>(out_grad->dims());

    auto src_fmt =
        diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : in_x->format();
    auto diff_fmt =
        diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : out_grad->format();

    auto dims = phi::vectorize(in_x->dims());
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), diff_fmt);
    auto src_md = platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), src_fmt);

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm, src_md, alpha, beta);
    this->AcquireBackwardPrimitiveDescriptor(algorithm, diff_dst_md, src_md,
                                             alpha, beta);
  }

  std::shared_ptr<dnnl::memory> AcquireBackwardSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }
};

class ReorderMKLDNNHandler {
 public:
  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       dnnl::memory::data_type dtype, dnnl::engine engine)
      : dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype),
        dtype_(dtype),
        dtype_dst_(dtype),
        engine_(engine) {}

  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       dnnl::memory::data_type dtype,
                       framework::proto::VarType::Type vtype_dst,
                       dnnl::memory::data_type dtype_dst, dnnl::engine engine)
      : dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype_dst),
        dtype_(dtype),
        dtype_dst_(dtype_dst),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const MKLDNNMemoryFormat& fmt,
                                                 void* ptr) {
    auto md = dnnl::memory::desc(dims_, dtype_, fmt);
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSubmemory(
      const std::vector<int64_t>& dims, const std::vector<int64_t>& offset,
      const std::shared_ptr<dnnl::memory>& mem_p) {
    auto sub_md = mem_p->get_desc().submemory_desc(dims, {offset});
    auto sub_mem_p = std::make_shared<dnnl::memory>(sub_md, engine_,
                                                    mem_p->get_data_handle());
    return sub_mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor* output,
                                                 const MKLDNNMemoryFormat& fmt,
                                                 platform::Place place) {
    auto dst_md = platform::MKLDNNMemDesc(dims_, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(
        place, framework::TransToPhiDataType(vtype_dst_), dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      framework::Tensor* output, const std::vector<int64_t>& dims,
      const MKLDNNMemoryFormat& fmt, platform::Place place) {
    auto dst_md = platform::MKLDNNMemDesc(dims, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(
        place, framework::TransToPhiDataType(vtype_dst_), dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    return std::make_shared<dnnl::reorder>(*(src_memory_p), *(dst_memory_p));
  }

 private:
  std::vector<int64_t> dims_;
  framework::proto::VarType::Type vtype_, vtype_dst_;
  dnnl::memory::data_type dtype_, dtype_dst_;
  dnnl::engine engine_;
};

template <typename T>
static void SetDstMemoryQuantized(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    std::vector<int64_t> dst_tz, const dnnl::engine& engine,
    std::shared_ptr<dnnl::memory::desc>& dst_md,  // NOLINT
    std::shared_ptr<dnnl::memory>& dst_memory,    // NOLINT
    MKLDNNMemoryFormat output_format) {
  T* output_data = output->mutable_data<T>(ctx.GetPlace());
  const size_t dst_dims = dst_tz.size();
  MKLDNNMemoryFormat dst_fmt;

  PADDLE_ENFORCE_LE(dst_dims, 5, platform::errors::InvalidArgument(
                                     "Dst memory for quantization can not have "
                                     "dims > 5. But received dst_dims is %d.",
                                     dst_dims));
  dst_fmt = platform::MKLDNNFormatForSize(dst_dims, output_format);

  auto tmp_dst_md = platform::MKLDNNMemDesc(
      {dst_tz}, paddle::framework::ToMKLDNNDataType(
                    framework::DataTypeTrait<T>::DataType()),
      dst_fmt);
  dst_md.reset(new dnnl::memory::desc(tmp_dst_md));
  dst_memory.reset(
      new dnnl::memory(*dst_md, engine, to_void_cast<T>(output_data)));
}

}  // namespace platform
}  // namespace paddle
