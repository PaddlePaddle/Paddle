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

<<<<<<< HEAD
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
=======
#include "paddle/fluid/platform/mkldnn_reuse.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace operators {

<<<<<<< HEAD
using phi::funcs::CreateKey;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::RNNReorderType;
using OneDNNMemoryFormat = dnnl::memory::format_tag;

template <typename T, typename T_alg, typename T_out = T>
class RNNMKLDNNHandler : public phi::funcs::OneDNNHandlerT<T, T_alg> {
 public:
  RNNMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                   const phi::OneDNNContext& dev_ctx,
                   const dnnl::engine onednn_engine,
                   platform::Place cpu_place,
                   const phi::DenseTensor* input,
                   const phi::DenseTensor* weight_h,
                   const phi::DenseTensor* h0,
=======
using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using paddle::platform::CreateKey;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::MKLDNNMemDesc;
using phi::CPUContext;
using platform::to_void_cast;

template <typename T, typename T_alg, typename T_out = T>
class RNNMKLDNNHandler : public platform::MKLDNNHandlerT<T, T_alg> {
 public:
  RNNMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                   const platform::MKLDNNDeviceContext& dev_ctx,
                   const dnnl::engine mkldnn_engine,
                   platform::Place cpu_place,
                   const LoDTensor* input,
                   const Tensor* weight_h,
                   const Tensor* h0,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                   const bool is_reverse,
                   const int64_t N,
                   const int64_t Ti,
                   const int64_t IC,
                   const int64_t OC,
                   const int64_t G,
                   const std::string& unique_name)
<<<<<<< HEAD
      : phi::funcs::OneDNNHandlerT<T, T_alg>(
            dev_ctx,
            dev_ctx.GetEngine(),
            cpu_place,
            CreateKey(dev_ctx, unique_name, OneDNNGetDataType<T>(), Ti)),
=======
      : platform::MKLDNNHandlerT<T, T_alg>(
            dev_ctx,
            dev_ctx.GetEngine(),
            cpu_place,
            CreateKey(dev_ctx, unique_name, MKLDNNGetDataType<T>(), Ti)),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        N(N),
        Ti(Ti),
        IC(IC),
        OC(OC),
        G(G) {
    // Create memory key without Ti because weights, bias and h0 memories
    // do not depend on Ti size but primitive and input/output memory do
<<<<<<< HEAD
    memory_key_ = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx, CreateKey(dev_ctx, unique_name, OneDNNGetDataType<T>()));
=======
    memory_key_ = platform::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx, CreateKey(dev_ctx, unique_name, MKLDNNGetDataType<T>()));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    // Is it int8 kernel
    const bool is_INT8 = std::is_same<T, uint8_t>::value;

    if (is_INT8) {
      // Int8 attributes
      const float scale_data = ctx.Attr<float>("Scale_data");
      const float shift_data = ctx.Attr<float>("Shift_data");
      const auto scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");

      const int weights_scale_mask =
          0 +
          (1 << 3)  // bit, indicating the unique scales for `g` dim in `ldigo`
          +
          (1 << 4);  // bit, indicating the unique scales for `o` dim in `ldigo`

      attr_.set_rnn_data_qparams(scale_data, shift_data);
      attr_.set_rnn_weights_qparams(weights_scale_mask, scale_weights);
    }
  }

<<<<<<< HEAD
  bool is_NTC() { return this->is_NTC(this->fwd_pd_->dst_desc()); }

  bool is_NTC(const dnnl::memory::desc& md) {
    auto ntc_md = dnnl::memory::desc(
        md.dims(), md.data_type(), dnnl::memory::format_tag::ntc);
    return md == ntc_md;
=======
  bool is_NTC() {
    return (platform::GetMKLDNNFormat(this->fwd_pd_->dst_desc()) ==
            dnnl::memory::format_tag::ntc);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }

  void reorderRNNdata(void* input_data,
                      void* output_data,
                      std::vector<size_t> lod,
                      const bool is_reverse,
<<<<<<< HEAD
                      RNNReorderType reorder_type) {
    switch (reorder_type) {
      // Reorder input memory [WORDS, C] + LoD -> [N, T, C]
      case RNNReorderType::PP_NTC: {
=======
                      platform::RNNReorderType reorder_type) {
    switch (reorder_type) {
      // Reorder input memory [WORDS, C] + LoD -> [N, T, C]
      case platform::RNNReorderType::PP_NTC: {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * IC;
          const auto offset = is_reverse ? (Ti * IC - num_elements) : 0;
          memcpy(output_data_iter + n * Ti * IC + offset,
                 input_data_iter,
                 sizeof(T) * num_elements);
          input_data_iter += num_elements;
        }
      } break;
      // Reorder input memory [WORDS, C] + LoD -> [T, N, C]
<<<<<<< HEAD
      case RNNReorderType::PP_TNC: {
=======
      case platform::RNNReorderType::PP_TNC: {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]);
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter + (t + offset) * N * IC + n * IC,
                   input_data_iter,
                   sizeof(T) * IC);
            input_data_iter += IC;
          }
        }
      } break;
      // Reorder output values to PP format [N, T, C] -> [WORDS, C]
<<<<<<< HEAD
      case RNNReorderType::NTC_PP: {
=======
      case platform::RNNReorderType::NTC_PP: {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * OC;
          const auto offset = is_reverse ? (Ti * OC - num_elements) : 0;
          memcpy(output_data_iter,
                 input_data_iter + n * Ti * OC + offset,
                 sizeof(T_out) * num_elements);
          output_data_iter += num_elements;
        }
      } break;
      // Reorder output values to PP format [T, N, C] -> [WORDS, C]
<<<<<<< HEAD
      case RNNReorderType::TNC_PP: {
=======
      case platform::RNNReorderType::TNC_PP: {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = lod[n + 1] - lod[n];
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter,
                   input_data_iter + (t + offset) * N * OC + n * OC,
                   sizeof(T_out) * OC);
            output_data_iter += OC;
          }
        }
      } break;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireInputMemoryWithReorder(
<<<<<<< HEAD
      const phi::DenseTensor* input, const bool is_reverse) {
=======
      const LoDTensor* input, const bool is_reverse) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const auto name = this->key_ + "@input_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }

    const auto& input_lod = input->lod()[0];
<<<<<<< HEAD
    auto* x_data = phi::funcs::to_void_cast(input->data<T>());
=======
    auto* x_data = to_void_cast(input->data<T>());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    auto* x_onednn_data = memory_p->get_data_handle();
    memset(x_onednn_data, 0, sizeof(T) * N * Ti * IC);

<<<<<<< HEAD
    if (is_NTC(this->fwd_pd_->src_desc())) {
      reorderRNNdata(
          x_data, x_onednn_data, input_lod, is_reverse, RNNReorderType::PP_NTC);
    } else {
      reorderRNNdata(
          x_data, x_onednn_data, input_lod, is_reverse, RNNReorderType::PP_TNC);
=======
    if (platform::GetMKLDNNFormat(this->fwd_pd_->src_desc()) ==
        dnnl::memory::format_tag::ntc) {
      reorderRNNdata(x_data,
                     x_onednn_data,
                     input_lod,
                     is_reverse,
                     platform::RNNReorderType::PP_NTC);
    } else {
      reorderRNNdata(x_data,
                     x_onednn_data,
                     input_lod,
                     is_reverse,
                     platform::RNNReorderType::PP_TNC);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireOutputMemory() {
    const auto name = this->key_ + "@output_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->dst_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }
    return memory_p;
  }

  // TODO(grygielski) H0 is for now persistable
  // TODO(jczaja) H0 should be updated each iter and of T type (Fusion pass does
  // not support in yet)
  template <typename U>
<<<<<<< HEAD
  std::shared_ptr<dnnl::memory> AcquireH0Memory(const phi::DenseTensor* h0) {
=======
  std::shared_ptr<dnnl::memory> AcquireH0Memory(const Tensor* h0) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const std::string h0_key = memory_key_ + "@h0";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(h0_key));

    if (!memory_p) {
      auto user_h0_memory = dnnl::memory();
      if (h0) {
        user_h0_memory = dnnl::memory(
<<<<<<< HEAD
            {{1, 1, N, OC}, OneDNNGetDataType<U>(), OneDNNMemoryFormat::ldnc},
            this->engine_,
            phi::funcs::to_void_cast(h0->data<U>()));
      } else {
        user_h0_memory = dnnl::memory(
            {{1, 1, N, OC}, OneDNNGetDataType<U>(), OneDNNMemoryFormat::ldnc},
=======
            {{1, 1, N, OC}, MKLDNNGetDataType<U>(), MKLDNNMemoryFormat::ldnc},
            this->engine_,
            to_void_cast(h0->data<U>()));
      } else {
        user_h0_memory = dnnl::memory(
            {{1, 1, N, OC}, MKLDNNGetDataType<U>(), MKLDNNMemoryFormat::ldnc},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            this->engine_);
        memset(user_h0_memory.get_data_handle(), 0, sizeof(U) * N * OC);
      }
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_iter_desc(),
                                                this->engine_);

<<<<<<< HEAD
      auto& astream = phi::OneDNNContext::tls().get_stream();
=======
      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      dnnl::reorder(user_h0_memory, *memory_p, attr_)
          .execute(astream, user_h0_memory, *memory_p);

      this->dev_ctx_.SetBlob(h0_key, memory_p);
    }
    return memory_p;
  }

 protected:
  // RNN dimensions
  // N - Batch Size
  // Ti - Max sentence length
  // IC - Input Channels
  // OC - Output Channels
  // G  - Number of gates
  const int64_t N, Ti, IC, OC, G;

  // Memory size of weights, bias and h0 does not depend
  // on Ti size, thus we need another key to cache them
  std::string memory_key_;
  dnnl::primitive_attr attr_;
};
}  // namespace operators
}  // namespace paddle
