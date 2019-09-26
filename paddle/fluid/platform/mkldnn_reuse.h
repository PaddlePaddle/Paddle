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

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "boost/optional.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = mkldnn::memory;

template <typename T, typename TForward, typename TBackward>
class MKLDNNHandlerT {
 public:
  MKLDNNHandlerT(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                 platform::Place cpu_place, const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        place_(cpu_place),
        key_common_(base_key),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    if (platform::get_cur_mkldnn_session_id() !=
        platform::kMKLDNNSessionID_Default) {
      key_ = key_common_;
    } else {
      key_ = key_common_ + "-t:" + ThreadIDasStr();
    }
  }

  template <typename... Args>
  std::shared_ptr<TForward> AcquireForwardPrimitive(Args&&... args) {
    const std::string key_p = key_ + "@forward_p";
    auto forward_p =
        std::static_pointer_cast<TForward>(dev_ctx_.GetBlob(key_p));
    if (forward_p == nullptr) {
      forward_p =
          std::make_shared<TForward>(*fwd_pd_, std::forward<Args>(args)...);
      dev_ctx_.SetBlob(key_p, forward_p);
    }
    return forward_p;
  }

  template <typename... Args>
  std::shared_ptr<TBackward> AcquireBackwardPrimitive(Args&&... args) {
    const std::string key_p = key_ + "@backward_p";
    auto backward_p =
        std::static_pointer_cast<TBackward>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      backward_p =
          std::make_shared<TBackward>(*bwd_pd_, std::forward<Args>(args)...);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->src_primitive_desc(),
                                            to_void_cast<T>(input_data),
                                            "@src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output) {
    T* ptr = output->mutable_data<T>(place_,
                                     fwd_pd_->dst_primitive_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const framework::Tensor* output) {
    const T* output_data = output->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_primitive_desc(),
                                            to_void_cast<T>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const framework::Tensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_dst_primitive_desc(),
                                            to_void_cast<T>(ptr),
                                            "@diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      framework::Tensor* diffsrc) {
    T* ptr = diffsrc->mutable_data<T>(
        place_, bwd_pd_->diff_src_primitive_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_primitive_desc(),
                                            ptr, "@diff_src_mem_p");
  }

 protected:
  template <typename... Args>
  void AcquireForwardPrimitiveDescriptor(Args&&... args) {
    // Forward PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_pd = key_common_ + "@forward_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (fwd_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
          dev_ctx_.GetBlob(key_pd));
      if (fwd_pd_ == nullptr) {
        auto fwd_desc = typename TForward::desc(std::forward<Args>(args)...);
        fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(fwd_desc,
                                                                      engine_);
        dev_ctx_.SetBlob(key_pd, fwd_pd_);
      }
    }
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    const std::string key_fwd_pd = key_common_ + "@forward_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_fwd_pd));
    PADDLE_ENFORCE_NOT_NULL(fwd_pd_);
    const std::string key_pd = key_ + "@backward_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (bwd_pd_ == nullptr) {
      auto bwd_desc = typename TBackward::desc(std::forward<Args>(args)...);
      bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
          bwd_desc, engine_, *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_pd_);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::primitive_desc mdp, void* ptr,
      const std::string& suffix) {
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(mdp, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  platform::Place place_;
  std::string key_;
  std::string key_common_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
};

// TODO(grygielski) this class will be deleted later.
class MKLDNNHandler {
 public:
  MKLDNNHandler(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                const std::string& base_key)
      : dev_ctx_(dev_ctx), engine_(engine), key_common_(base_key) {
    if (platform::get_cur_mkldnn_session_id() !=
        platform::kMKLDNNSessionID_Default) {
      key_ = key_common_;
    } else {
      key_ = key_common_ + "-t:" + ThreadIDasStr();
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::primitive_desc mdp, void* ptr,
      const std::string& suffix) {
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(mdp, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  // This incarnation of AcquireMemory can call user function eg. custom reorder
  // or preprocessing routine if needed
  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const mkldnn::memory::desc& md, void* ptr, const std::string& suffix,
      user_function custom_func = {}) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      // Call custom reorder/preprocessing func if available
      if (custom_func) {
        auto reordered_data = custom_func(reinterpret_cast<const float*>(ptr));
        dev_ctx_.SetBlob(local_key + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }

      mem_p = std::make_shared<mkldnn::memory>(
          mkldnn::memory::primitive_desc{md, engine_}, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::vector<int>& dims, const mkldnn::memory::data_type dtype,
      const MKLDNNMemoryFormat& fmt, void* ptr, const std::string& suffix) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto md = mkldnn::memory::desc(dims, dtype, fmt);

      mem_p = std::make_shared<mkldnn::memory>(
          mkldnn::memory::primitive_desc{md, engine_}, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::shared_ptr<mkldnn::memory>& user_memory_p,
      const std::shared_ptr<mkldnn::memory>& target_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto stored_reorder_p = std::static_pointer_cast<mkldnn::reorder>(
        dev_ctx_.GetBlob(key_reorder_p));

    if (stored_reorder_p) {
      pipeline.push_back(*stored_reorder_p);
    } else {
      auto reorder_p =
          std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
      dev_ctx_.SetBlob(key_reorder_p, reorder_p);
      pipeline.push_back(*reorder_p);
    }

    return target_memory_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      mkldnn::memory::primitive_desc& mpd,       // NOLINT
      mkldnn::memory::primitive_desc& user_mpd,  // NOLINT
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f}, int mask = 0) {
    // create reorder primitive if the input format is not the preferred one
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto target_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (target_memory_p == nullptr) {
      target_memory_p = user_memory_p;
      std::shared_ptr<mkldnn::primitive> reorder_p;
      if (mpd != user_mpd) {
        target_memory_p = std::make_shared<mkldnn::memory>(mpd);
        std::shared_ptr<mkldnn::reorder> reorder_p;
        if (is_INT8) {
          mkldnn::primitive_attr
              attri;  // attribute for int8 weights and bias data reorder.
          attri.set_output_scales(mask, scale_data);

          auto reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
              new mkldnn::reorder::primitive_desc(user_mpd, mpd, attri));
          reorder_p = std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(
              *reorder_pd, *user_memory_p, *target_memory_p));
        } else {
          reorder_p = std::make_shared<mkldnn::reorder>(*user_memory_p,
                                                        *target_memory_p);
        }
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);
        pipeline.push_back(*reorder_p);
      }
      dev_ctx_.SetBlob(local_key, target_memory_p);
    } else if (!is_persistent) {
      // Make reorder if needed
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        pipeline.push_back(*reorder_p);
      }
    }
    return target_memory_p;
  }

 protected:
  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  std::string key_;
  std::string key_common_;
};

class SumMKLDNNHandler : public MKLDNNHandler {
 public:
  SumMKLDNNHandler(const platform::MKLDNNDeviceContext& dev_ctx,
                   mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {}

  std::shared_ptr<mkldnn::sum::primitive_desc> AcquireSumPrimitiveDescriptor(
      const std::vector<std::shared_ptr<mkldnn::memory>>& src_mems,
      const std::vector<float>& scales, const mkldnn::memory::desc& dst_md) {
    const std::string key_sum_pd = key_ + "@sum_pd";

    sum_pd_ = std::static_pointer_cast<mkldnn::sum::primitive_desc>(
        dev_ctx_.GetBlob(key_sum_pd));
    if (sum_pd_ == nullptr) {
      // Get vector of inputs primitive descriptors
      std::vector<mkldnn::memory::primitive_desc> src_pds;
      for (auto& input_mem : src_mems) {
        src_pds.push_back(input_mem->get_primitive_desc());
      }

      sum_pd_.reset(new mkldnn::sum::primitive_desc(dst_md, scales, src_pds));
      dev_ctx_.SetBlob(key_sum_pd, sum_pd_);
    }

    return sum_pd_;
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    return this->AcquireMemoryFromPrimitive(sum_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSecondSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_src2_mem_p");
  }

  std::shared_ptr<mkldnn::sum> AcquireSum(
      std::shared_ptr<mkldnn::memory> dst_memory,
      std::vector<mkldnn::primitive::at>* inputs) {
    auto prim_key = key_ + "@sum_p";
    auto sum_p =
        std::static_pointer_cast<mkldnn::sum>(dev_ctx_.GetBlob(prim_key));
    if (sum_p == nullptr) {
      sum_p = std::make_shared<mkldnn::sum>(*(sum_pd_), *inputs, *(dst_memory));
      dev_ctx_.SetBlob(prim_key, sum_p);
    }
    return sum_p;
  }

 private:
  std::shared_ptr<mkldnn::sum::primitive_desc> sum_pd_;
};

template <typename T>
class ActivationMKLDNNHandler
    : public MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                            mkldnn::eltwise_backward> {
 public:
  ActivationMKLDNNHandler(const std::vector<int>& dims,
                          mkldnn::algorithm algorithm, float alpha, float beta,
                          const MKLDNNMemoryFormat fmt, bool is_test,
                          const platform::MKLDNNDeviceContext& dev_ctx,
                          platform::Place cpu_place,
                          const std::string& unique_name)

      : platform::MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                                 mkldnn::eltwise_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, algorithm, fmt, alpha, beta,
                                unique_name)) {
    auto md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);

    this->AcquireForwardPrimitiveDescriptor(
        is_test ? mkldnn::prop_kind::forward_inference
                : mkldnn::prop_kind::forward_training,
        algorithm, md, alpha, beta);
  }

  ActivationMKLDNNHandler(const std::vector<int>& dims,
                          mkldnn::algorithm algorithm, float alpha, float beta,
                          const MKLDNNMemoryFormat fmt,
                          const MKLDNNMemoryFormat diff_fmt,
                          const platform::MKLDNNDeviceContext& dev_ctx,
                          platform::Place cpu_place,
                          const std::string& unique_name)

      : platform::MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                                 mkldnn::eltwise_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, algorithm, fmt, alpha, beta,
                                unique_name)) {
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), diff_fmt);
    auto src_md =
        platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(), fmt);

    this->AcquireBackwardPrimitiveDescriptor(algorithm, diff_dst_md, src_md,
                                             alpha, beta);
  }

  std::shared_ptr<mkldnn::memory> AcquireBackwardSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_primitive_desc(),
                                            to_void_cast<T>(input_data),
                                            "@bwd-src_mem_p");
  }
};

template <typename T>
class LRNMKLDNNHandler
    : public MKLDNNHandlerT<T, mkldnn::lrn_forward, mkldnn::lrn_backward> {
 public:
  LRNMKLDNNHandler(const std::vector<int>& dims, const int n, const float alpha,
                   const float beta, const float k,
                   const MKLDNNMemoryFormat fmt, bool is_test,
                   const platform::MKLDNNDeviceContext& dev_ctx,
                   platform::Place cpu_place, const std::string& unique_name)

      : platform::MKLDNNHandlerT<T, mkldnn::lrn_forward, mkldnn::lrn_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, n, alpha, beta, k, fmt, unique_name)) {
    auto src_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
    this->AcquireForwardPrimitiveDescriptor(
        is_test ? mkldnn::prop_kind::forward_inference
                : mkldnn::prop_kind::forward_training,
        mkldnn::lrn_across_channels, src_md, n, alpha, beta, k);
  }

  LRNMKLDNNHandler(const std::vector<int>& dims, const int n, const float alpha,
                   const float beta, const float k,
                   const MKLDNNMemoryFormat fmt,
                   const MKLDNNMemoryFormat diff_fmt,
                   const platform::MKLDNNDeviceContext& dev_ctx,
                   platform::Place cpu_place, const std::string& unique_name)

      : platform::MKLDNNHandlerT<T, mkldnn::lrn_forward, mkldnn::lrn_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, n, alpha, beta, k, fmt, unique_name)) {
    auto src_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
    auto diff_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), diff_fmt);

    this->AcquireBackwardPrimitiveDescriptor(
        mkldnn::lrn_across_channels, src_md, diff_md, n, alpha, beta, k);
  }

  std::shared_ptr<mkldnn::memory> AcquireWorkspaceMemory(
      framework::Tensor* workspace) {
    T* ptr = workspace->mutable_data<T>(
        this->place_, this->fwd_pd_->dst_primitive_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->workspace_primitive_desc(), ptr, "@wrk_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireBackwardWorkspaceMemory(
      const framework::Tensor* workspace) {
    const T* workspace_data = workspace->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->workspace_primitive_desc(),
        to_void_cast<T>(workspace_data), "@bwd-wrk_mem_p");
  }
};

template <typename T>
class PoolingMKLDNNHandler : public MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                                   mkldnn::pooling_backward> {
 public:
  PoolingMKLDNNHandler(
      const std::vector<int>& src_dims, const std::vector<int>& dst_dims,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::string& pooling_type,
      bool ceil_mode, const MKLDNNMemoryFormat fmt,
      mkldnn::memory::data_type dt, bool is_test,
      const platform::MKLDNNDeviceContext& dev_ctx, platform::Place cpu_place,
      const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                 mkldnn::pooling_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(src_dims, pooling_type, ksize, strides,
                                paddings, dt, fmt, unique_name)) {
    auto src_md = mkldnn::memory::desc(src_dims, dt, fmt);
    /* create memory descriptor for pooling without specified format
     * ('any') which lets a primitive (pooling in this case) choose
     * the memory format preferred for best performance
     */
    auto dst_md =
        platform::MKLDNNMemDesc(dst_dims, dt, MKLDNNMemoryFormat::any);

    std::vector<int> padding_left_top(paddings);
    std::vector<int> padding_right_bottom(paddings);
    if (ceil_mode) {
      CorrectOutputSize(src_dims, dst_dims, ksize, paddings, strides,
                        padding_right_bottom);
    }

    this->AcquireForwardPrimitiveDescriptor(
        is_test ? mkldnn::prop_kind::forward_inference
                : mkldnn::prop_kind::forward_training,
        pooling_type == "max" ? mkldnn::algorithm::pooling_max
                              : mkldnn::algorithm::pooling_avg,
        src_md, dst_md, strides, ksize, padding_left_top, padding_right_bottom,
        mkldnn::padding_kind::zero);
  }

  PoolingMKLDNNHandler(
      const std::vector<int>& diff_dst_dims,
      const std::vector<int>& diff_src_dims, const std::vector<int>& ksize,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::string& pooling_type, bool ceil_mode,
      const MKLDNNMemoryFormat fmt, const MKLDNNMemoryFormat diff_dst_fmt,
      mkldnn::memory::data_type dt,
      const platform::MKLDNNDeviceContext& dev_ctx, platform::Place cpu_place,
      const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                 mkldnn::pooling_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(diff_src_dims, pooling_type, ksize, strides,
                                paddings, dt, fmt, unique_name)) {
    auto diff_dst_md = mkldnn::memory::desc(
        diff_dst_dims, platform::MKLDNNGetDataType<T>(), diff_dst_fmt);
    auto diff_src_md =
        mkldnn::memory::desc(diff_src_dims, platform::MKLDNNGetDataType<T>(),
                             MKLDNNMemoryFormat::any);

    this->AcquireBackwardPrimitiveDescriptor(
        pooling_type == "max" ? mkldnn::algorithm::pooling_max
                              : mkldnn::algorithm::pooling_avg,
        diff_src_md, diff_dst_md, strides, ksize, paddings, paddings,
        mkldnn::padding_kind::zero);
  }

  std::shared_ptr<mkldnn::memory> AcquireWorkspaceMemory(void) {
    mkldnn::memory::primitive_desc workspace_mpd =
        this->fwd_pd_->workspace_primitive_desc();
    // Pooling PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    auto local_key = this->key_common_ + "@workspace";
    auto mem_p = std::static_pointer_cast<mkldnn::memory>(
        this->dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      mem_p = std::static_pointer_cast<mkldnn::memory>(
          this->dev_ctx_.GetBlob(local_key));
      if (mem_p == nullptr) {
        mem_p = std::make_shared<mkldnn::memory>(workspace_mpd);
        this->dev_ctx_.SetBlob(local_key, mem_p);
      }
    }
    return mem_p;
  }

 private:
  static inline int ComputeCeiledOutput(int input_size, int kernel_size,
                                        int padding, int stride) {
    return (input_size - kernel_size + 2 * padding) / stride + 1;
  }

  static inline void CorrectOutputSize(
      const std::vector<int>& src_tz, const std::vector<int>& dst_tz,
      const std::vector<int>& kernel_size, const std::vector<int>& paddings,
      const std::vector<int>& strides,
      std::vector<int>& right_bot_padding) {  // NOLINT
    for (size_t i = 0; i < right_bot_padding.size(); i++) {
      int desired_size = ComputeCeiledOutput(src_tz[i + 2], kernel_size[i],
                                             paddings[i], strides[i]);
      if (desired_size != dst_tz[i + 2]) {
        right_bot_padding[i] += strides[i] - 1;
      }
    }
  }
};

class TransposeMKLDNNHandler : public MKLDNNHandler {
 public:
  TransposeMKLDNNHandler(std::vector<int>& dims,  // NOLINT
                         std::vector<int>& axis,  // NOLINT
                         const platform::MKLDNNDeviceContext& dev_ctx,
                         mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        axis_(axis),
        logical_axis_(dims.size(), 0) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const MKLDNNMemoryFormat& fmt, void* ptr) {
    auto local_key = key_ + "@user_src_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      // Make memory descriptor using input format, unless it
      // cannot be trusted (nchw) then make up memory fmt manually
      for (size_t i = 0; i < logical_axis_.size(); ++i) {
        logical_axis_[i] = i;
      }
      auto src_md = fmt != MKLDNNMemoryFormat::nchw
                        ? platform::MKLDNNMemDesc(
                              dims_, platform::MKLDNNGetDataType<float>(), fmt)
                        : Axis2MemoryDesc(dims_, logical_axis_);
      mem_p = std::make_shared<mkldnn::memory>(
          mkldnn::memory::primitive_desc{src_md, engine_}, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output,
                                                   platform::Place place) {
    auto local_key = key_ + "@user_dst_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto dst_mdp = mkldnn::memory::primitive_desc{
          Axis2MemoryDesc(dims_, axis_), engine_};

      auto dst_data = output->mutable_data<float>(place, dst_mdp.get_size());

      mem_p = std::make_shared<mkldnn::memory>(dst_mdp, dst_data);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      auto dst_data = output->mutable_data<float>(place);
      mem_p->set_data_handle(dst_data);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::reorder> AcquireTranspose(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    auto prim_key = key_ + "@transpose_p";
    auto transpose_p =
        std::static_pointer_cast<mkldnn::reorder>(dev_ctx_.GetBlob(prim_key));
    if (transpose_p == nullptr) {
      transpose_p =
          std::make_shared<mkldnn::reorder>(*(src_memory_p), *(dst_memory_p));
      dev_ctx_.SetBlob(prim_key, transpose_p);
    }
    return transpose_p;
  }

 protected:
  mkldnn_memory_desc_t Axis2MemoryDesc(std::vector<int>& nchw_tz,  // NOLINT
                                       std::vector<int>& axis      // NOLINT
                                       ) {
    mkldnn_memory_desc_t mem_fmt;

    mem_fmt.primitive_kind = mkldnn_memory;
    mem_fmt.ndims = axis.size();
    for (unsigned int i = 0; i < nchw_tz.size(); ++i) {
      mem_fmt.dims[i] = nchw_tz[i];  // logical dimensions (nchw format,
      // regardless physical layout)
    }
    mem_fmt.data_type = mkldnn_f32;
    mem_fmt.format = mkldnn_blocked;

    unsigned int total_stride = 1;
    for (int i = nchw_tz.size() - 1; i >= 0; --i) {
      mem_fmt.layout_desc.blocking.padding_dims[i] =
          nchw_tz[i];  // logical dimensions (nchw format, regardless physical
      // layout)
      mem_fmt.layout_desc.blocking.block_dims[i] = 1;
      mem_fmt.layout_desc.blocking.offset_padding_to_data[i] = 0;  // no offset
      mem_fmt.layout_desc.blocking.strides[0][axis[i]] = total_stride;
      mem_fmt.layout_desc.blocking.strides[1][axis[i]] = 1;
      total_stride *= nchw_tz[axis[i]];
    }
    mem_fmt.layout_desc.blocking.offset_padding = 0;  // no initial offset
    return mem_fmt;
  }

 private:
  std::vector<int> dims_;
  std::vector<int> axis_;
  std::vector<int> logical_axis_;
};

class ReorderMKLDNNHandler : public MKLDNNHandler {
 public:
  ReorderMKLDNNHandler(std::vector<int>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       mkldnn::memory::data_type dtype,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        vtype_(vtype),
        dtype_(dtype) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const MKLDNNMemoryFormat& fmt, void* ptr) {
    return this->AcquireMemory(dims_, dtype_, fmt, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      framework::Tensor* output, const MKLDNNMemoryFormat& fmt,
      platform::Place place) {
    auto local_key = key_ + "@user_dst_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto dst_md = platform::MKLDNNMemDesc(dims_, dtype_, fmt);
      auto dst_mdp = mkldnn::memory::primitive_desc{dst_md, engine_};

      auto dst_data = output->mutable_data(place, vtype_);

      mem_p = std::make_shared<mkldnn::memory>(dst_mdp, dst_data);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      auto dst_data = output->mutable_data(place, vtype_);
      mem_p->set_data_handle(dst_data);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::reorder> AcquireReorder(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    auto prim_key = key_ + "@reorder_p";
    auto reorder_p =
        std::static_pointer_cast<mkldnn::reorder>(dev_ctx_.GetBlob(prim_key));
    if (reorder_p == nullptr) {
      reorder_p =
          std::make_shared<mkldnn::reorder>(*(src_memory_p), *(dst_memory_p));
      dev_ctx_.SetBlob(prim_key, reorder_p);
    }
    return reorder_p;
  }

 private:
  std::vector<int> dims_;
  framework::proto::VarType::Type vtype_;
  mkldnn::memory::data_type dtype_;
};

template <typename T>
struct convolutional_algorithm;

template <>
struct convolutional_algorithm<mkldnn::convolution_forward> {
  static constexpr mkldnn::algorithm T = mkldnn::algorithm::convolution_direct;
};

template <>
struct convolutional_algorithm<mkldnn::deconvolution_forward> {
  static constexpr mkldnn::algorithm T =
      mkldnn::algorithm::deconvolution_direct;
};

template <class forward_t, class backward_data_t, class backward_weights_t>
class ConvMKLDNNTemplateHandler : public MKLDNNHandler {
 public:
  ConvMKLDNNTemplateHandler(const platform::MKLDNNDeviceContext& dev_ctx,
                            mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {}

  ConvMKLDNNTemplateHandler(
      std::shared_ptr<typename forward_t::primitive_desc> conv_pd,
      std::shared_ptr<typename backward_data_t::primitive_desc>
          conv_bwd_data_pd,
      std::shared_ptr<typename backward_weights_t::primitive_desc>
          conv_bwd_weights_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        conv_pd_(conv_pd),
        conv_bwd_weights_pd_(conv_bwd_weights_pd),
        conv_bwd_data_pd_(conv_bwd_data_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  size_t GetDstMemorySize() const {
    return conv_pd_->dst_primitive_desc().get_size();
  }

  MKLDNNMemoryFormat GetDstFormat() const {
    return static_cast<MKLDNNMemoryFormat>(
        conv_pd_->dst_primitive_desc().desc().data.format);
  }

  size_t GetDiffWeightsMemorySize() const {
    return conv_bwd_weights_pd_->diff_weights_primitive_desc().get_size();
  }

  size_t GetDiffSourceMemorySize() const {
    return conv_bwd_data_pd_->diff_src_primitive_desc().get_size();
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_bwd_weights_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p,
                               "@weights-src_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_weights_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@weights-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemoryFromWeightsPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_weights_pd_->diff_weights_primitive_desc(), ptr,
        "@diff_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_data_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@data-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto weights_pd = conv_bwd_data_pd_->weights_primitive_desc();
    auto user_pd = user_weights_memory_p->get_primitive_desc();
    return this->AcquireMemory(weights_pd, user_pd, user_weights_memory_p,
                               "@data-weights_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireResidualDataMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_residual_data_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromResidualDataMemory(
      const std::shared_ptr<mkldnn::memory>& user_residual_memory_p,
      void* dst_ptr,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    return this->AcquireMemory(user_residual_memory_p,
                               this->AcquireDstMemoryFromPrimitive(dst_ptr),
                               "@residual_data_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemoryFromDataPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_data_pd_->diff_src_primitive_desc(), ptr, "@diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    return this->AcquireMemoryFromPrimitive(conv_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p, "@src_mem_p",
                               pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemory(
      const mkldnn::memory::desc& md, void* ptr,
      user_function custom_func = {}) {
    return this->AcquireMemory(md, ptr, "@user_weights_mem_p", custom_func);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_bias_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f}, int mask = 0) {
    auto user_weights_pd = user_weights_memory_p->get_primitive_desc();
    auto weights_pd = conv_pd_->weights_primitive_desc();
    return this->AcquireMemory(
        weights_pd, user_weights_pd, user_weights_memory_p, "@weights_mem_p",
        pipeline, is_persistent, is_INT8, scale_data, mask);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_bias_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f},
      int mask = 0) {  // NOLINT
    auto user_bias_pd = user_bias_memory_p->get_primitive_desc();
    auto bias_pd = conv_pd_->bias_primitive_desc();
    return this->AcquireMemory(bias_pd, user_bias_pd, user_bias_memory_p,
                               "@bias_mem_p", pipeline, is_persistent, is_INT8,
                               scale_data, mask);
  }

  mkldnn::primitive_attr CreatePostOps(
      std::string fuse_activation, float fuse_alpha, float fuse_beta,
      bool fuse_residual_conn, const std::vector<float> output_shift_scale = {},
      float sum_scale = 1.0f) const {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    if (output_shift_scale.size() > 0) {
      int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
      conv_attr.set_output_scales(mask, output_shift_scale);
    }
    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     fuse_alpha, fuse_beta);
    }

    if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<typename forward_t::primitive_desc>
  AcquireConvolutionPrimitiveDescriptor(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& weights,
      boost::optional<const mkldnn::memory::desc&> bias,
      const mkldnn::memory::desc& dst, const std::vector<int>& strides,
      const std::vector<int>& paddings, const mkldnn::engine& engine,
      const std::string& fuse_activation, float fuse_alpha, float fuse_beta,
      const bool fuse_residual_conn, mkldnn::prop_kind fwd_prop_kind,
      const std::vector<float> output_shift_scale = {},
      const float sum_scale = 1.0f) {
    // Conv PD has to be passed to Grad op that
    // may be exxecuted by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_conv_pd = key_common_ + "@conv_pd";

    conv_pd_ = std::static_pointer_cast<typename forward_t::primitive_desc>(
        dev_ctx_.GetBlob(key_conv_pd));

    if (conv_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);

      conv_pd_ = std::static_pointer_cast<typename forward_t::primitive_desc>(
          dev_ctx_.GetBlob(key_conv_pd));
      if (conv_pd_ == nullptr) {
        mkldnn::memory::dims stride_dims = strides;
        mkldnn::memory::dims padding_dims = paddings;

        auto conv_desc =
            bias ? typename forward_t::desc(
                       fwd_prop_kind, convolutional_algorithm<forward_t>::T,
                       src, weights, *bias, dst, stride_dims, padding_dims,
                       padding_dims, mkldnn::padding_kind::zero)
                 : typename forward_t::desc(
                       fwd_prop_kind, convolutional_algorithm<forward_t>::T,
                       src, weights, dst, stride_dims, padding_dims,
                       padding_dims, mkldnn::padding_kind::zero);

        mkldnn::primitive_attr conv_attr =
            CreatePostOps(fuse_activation, fuse_alpha, fuse_beta,
                          fuse_residual_conn, output_shift_scale, sum_scale);

        conv_pd_.reset(new typename forward_t::primitive_desc(
            conv_desc, conv_attr, engine));
        // Save conv_pd/src_memory/weights_memory for backward pass
        dev_ctx_.SetBlob(key_conv_pd, conv_pd_);
      }
    }

    return conv_pd_;
  }

  std::shared_ptr<forward_t> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p =
        std::static_pointer_cast<forward_t>(dev_ctx_.GetBlob(prim_key));
    if (conv_p == nullptr) {
      conv_p = std::make_shared<forward_t>(*conv_pd_, *src_memory_p,
                                           *weights_memory_p, *dst_memory_p);

      dev_ctx_.SetBlob(prim_key, conv_p);
    }
    return conv_p;
  }

  std::shared_ptr<forward_t> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> bias_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p =
        std::static_pointer_cast<forward_t>(dev_ctx_.GetBlob(prim_key));
    if (conv_p == nullptr) {
      conv_p = std::make_shared<forward_t>(*conv_pd_, *src_memory_p,
                                           *weights_memory_p, *bias_memory_p,
                                           *dst_memory_p);

      dev_ctx_.SetBlob(prim_key, conv_p);
    }
    return conv_p;
  }

  std::shared_ptr<backward_weights_t> AcquireConvolutionBackwardWeights(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> diff_weights_memory_p) {
    auto prim_key = key_ + "@conv_bwd_weights_p";
    auto conv_bwd_weights_p = std::static_pointer_cast<backward_weights_t>(
        dev_ctx_.GetBlob(prim_key));
    if (conv_bwd_weights_p == nullptr) {
      // create backward conv primitive for weights
      conv_bwd_weights_p = std::make_shared<backward_weights_t>(
          *conv_bwd_weights_pd_, *src_memory_p, *diff_dst_memory_p,
          *diff_weights_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_weights_p);
    }
    return conv_bwd_weights_p;
  }

  std::shared_ptr<backward_data_t> AcquireConvolutionBackwardData(
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> diff_src_memory_p) {
    auto prim_key = key_ + "@conv_bwd_data_p";
    auto conv_bwd_data_p =
        std::static_pointer_cast<backward_data_t>(dev_ctx_.GetBlob(prim_key));
    if (conv_bwd_data_p == nullptr) {
      conv_bwd_data_p = std::make_shared<backward_data_t>(
          *conv_bwd_data_pd_, *diff_dst_memory_p, *weights_memory_p,
          *diff_src_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_data_p);
    }
    return conv_bwd_data_p;
  }

 private:
  std::shared_ptr<typename forward_t::primitive_desc> conv_pd_;
  std::shared_ptr<typename backward_weights_t::primitive_desc>
      conv_bwd_weights_pd_;
  std::shared_ptr<typename backward_data_t::primitive_desc> conv_bwd_data_pd_;
};

using ConvMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::convolution_forward,
                              mkldnn::convolution_backward_data,
                              mkldnn::convolution_backward_weights>;

using ConvTransposeMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::deconvolution_forward,
                              mkldnn::deconvolution_backward_data,
                              mkldnn::deconvolution_backward_weights>;

template <typename T>
static void SetDstMemoryQuantized(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    std::vector<int> dst_tz, const mkldnn::engine& engine,
    std::shared_ptr<mkldnn::memory::primitive_desc>& dst_pd,  // NOLINT
    std::shared_ptr<mkldnn::memory>& dst_memory) {            // NOLINT
  T* output_data = output->mutable_data<T>(ctx.GetPlace());
  const size_t dst_dims = dst_tz.size();
  MKLDNNMemoryFormat dst_fmt;
  PADDLE_ENFORCE_LE(dst_dims, 5,
                    "Dst memory for quantization can not have dims > 5");
  dst_fmt = platform::MKLDNNFormatForSize(dst_dims, MKLDNNMemoryFormat::nhwc);

  auto dst_md = platform::MKLDNNMemDesc(
      {dst_tz}, paddle::framework::ToMKLDNNDataType(
                    framework::DataTypeTrait<T>::DataType()),
      dst_fmt);
  dst_pd.reset(new mkldnn::memory::primitive_desc(dst_md, engine));
  dst_memory.reset(new mkldnn::memory(*dst_pd, to_void_cast<T>(output_data)));
}

}  // namespace platform
}  // namespace paddle
