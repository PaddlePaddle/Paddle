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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = dnnl::memory;
using Place = phi::Place;

template <typename T,
          typename TForward,
          typename TBackward = paddle::platform::mkldnn_dummy_primitive,
          typename TBackward_params = paddle::platform::mkldnn_dummy_primitive>
class MKLDNNHandlerNoCachingT {
 public:
  MKLDNNHandlerNoCachingT(dnnl::engine engine, Place cpu_place)
      : engine_(engine), place_(cpu_place), fwd_pd_(nullptr), bwd_pd_(nullptr) {
    phi::OneDNNContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    return std::make_shared<TForward>(*fwd_pd_);
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    return std::make_shared<TBackward>(*bwd_pd_);
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        phi::errors::Unavailable("BWD_PD should be set when "
                                 "getting BWD prim ."));
    return std::make_shared<TBackward_params>(*bwd_w_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->src_desc(), paddle::platform::to_void_cast<T>(input_data));
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
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->dst_desc(),
        paddle::platform::to_void_cast<T_out>(output_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const DenseTensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_dst_desc(), paddle::platform::to_void_cast<T>(ptr));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(DenseTensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr);
  }

  // Buffer of given Tensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      DenseTensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        phi::errors::Unavailable(
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
        phi::errors::Unavailable(
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
        phi::errors::Unavailable("Get MKLDNN Forward primitive %s failed."));
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
        phi::errors::Unavailable("Get MKLDNN Forward primitive %s failed."));
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

    auto& astream = phi::OneDNNContext::tls().get_stream();

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

      auto& astream = phi::OneDNNContext::tls().get_stream();
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
class ActivationMKLDNNHandler
    : public MKLDNNHandlerNoCachingT<T,
                                     dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ActivationMKLDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x)
      : MKLDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm,
                                            x->mem_desc(),
                                            alpha,
                                            beta);
  }

  ActivationMKLDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x,
                          const DenseTensor* dout)
      : MKLDNNHandlerNoCachingT<T,
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
    return this->AcquireMemoryFromPrimitive(
        this->bwd_pd_->src_desc(),
        paddle::platform::to_void_cast<T>(input_data));
  }
};

}  // namespace funcs
}  // namespace phi
