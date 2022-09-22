/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/expect.h"
#include "paddle/fluid/operators/fused/fusion_gru_op.h"
#include "paddle/fluid/operators/fused/mkldnn/fusion_rnn_mkldnn.h"

namespace paddle {
namespace operators {

using paddle::framework::LoDTensor;

using paddle::platform::MKLDNNGetDataType;
using paddle::platform::MKLDNNMemDesc;
using phi::CPUContext;
using platform::to_void_cast;

template <typename T, typename T_out = T>
class GRUMKLDNNHandler : public RNNMKLDNNHandler<T, dnnl::gru_forward, T_out> {
 public:
  GRUMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                   const platform::MKLDNNDeviceContext& dev_ctx,
                   const dnnl::engine mkldnn_engine,
                   platform::Place cpu_place,
                   const LoDTensor* input,
                   const phi::DenseTensor* weight_h,
                   const phi::DenseTensor* h0,
                   const bool is_reverse,
                   const int64_t N,
                   const int64_t Ti,
                   const int64_t IC,
                   const int64_t OC,
                   const std::string& unique_name)
      : RNNMKLDNNHandler<T, dnnl::gru_forward, T_out>(
            ctx,
            dev_ctx,
            mkldnn_engine,
            ctx.GetPlace(),
            input,
            weight_h,
            h0,
            is_reverse,
            N,
            Ti,
            IC,
            OC,
            3,
            ctx.InputName("X") + ctx.InputName("WeightH")) {
    const bool is_INT8 = std::is_same<T, uint8_t>::value;

    if (unlikely(!this->isCached())) {
      // oneDNN kernel has hardcoded activation functions
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("gate_activation"),
          "sigmoid",
          platform::errors::Unimplemented(
              "oneDNN fusion_gru supports only sigmoid as a gate activation."));
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("activation"),
          "tanh",
          platform::errors::Unimplemented(
              "oneDNN fusion_gru supports only tanh as an activation."));

      // Weights for int8 kernel are of a type s8
      const auto weights_dt =
          is_INT8 ? dnnl::memory::data_type::s8 : MKLDNNGetDataType<T>();

      // oneDNN RNN dimensions
      const int64_t D = 1;  // Directions
      const int64_t L = 1;  // Layers (PP supports only 1 stacked layer)
      const int64_t G = 3;  // Number of Gates, 3 for GRU

      // Create memory descriptors
      auto input_md = MKLDNNMemDesc(
          {Ti, N, IC}, MKLDNNGetDataType<T>(), MKLDNNMemoryFormat::ntc);
      auto weight_x_md =
          MKLDNNMemDesc({L, D, IC, G, OC}, weights_dt, MKLDNNMemoryFormat::any);
      auto weight_h_md =
          MKLDNNMemDesc({L, D, OC, G, OC}, weights_dt, MKLDNNMemoryFormat::any);
      auto bias_md = MKLDNNMemDesc(
          {L, D, G, OC}, MKLDNNGetDataType<float>(), MKLDNNMemoryFormat::ldgo);
      auto hidden_md = MKLDNNMemDesc(
          {Ti, N, OC}, MKLDNNGetDataType<T_out>(), MKLDNNMemoryFormat::ntc);
      auto h0_md = MKLDNNMemDesc(
          {L, D, N, OC}, MKLDNNGetDataType<T>(), MKLDNNMemoryFormat::ldnc);

      // Create GRU oneDNN primitive
      const auto direction =
          is_reverse ? dnnl::rnn_direction::unidirectional_right2left
                     : dnnl::rnn_direction::unidirectional_left2right;

      this->AcquireForwardPrimitiveDescriptor(
          this->attr_,
          dnnl::prop_kind::forward_inference,
          direction,
          input_md,
          h0_md,
          weight_x_md,
          weight_h_md,
          bias_md,
          hidden_md,
          dnnl::memory::desc());
    }
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightXMemory(
      const phi::DenseTensor* weight_x, const bool origin_mode) {
    const std::string wx_key = this->memory_key_ + "@weight_x";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wx_key));

    if (!memory_p) {
      auto user_md = MKLDNNMemDesc({1, 1, this->IC, this->G, this->OC},
                                   MKLDNNGetDataType<U>(),
                                   MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      auto* weight_x_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      memcpy(weight_x_data,
             weight_x->data<U>(),
             sizeof(U) * this->IC * this->G * this->OC);

      if (origin_mode == false) {
        for (int64_t i = 0; i < this->IC; ++i) {
          for (int64_t j = 0; j < this->OC; ++j) {
            U minus_one(-1.0f);
            weight_x_data[j] = minus_one * weight_x_data[j];
          }
          weight_x_data += 3 * this->OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_layer_desc(), this->engine_);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wx_key, memory_p);
    }
    return memory_p;
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightHMemory(
      const phi::DenseTensor* weight_h, const bool origin_mode) {
    const std::string wh_key = this->memory_key_ + "@weight_h";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wh_key));

    if (!memory_p) {
      auto user_md = MKLDNNMemDesc({1, 1, this->OC, this->G, this->OC},
                                   MKLDNNGetDataType<U>(),
                                   MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      // Reorder weights_h from PP format [OC, 2OC] + [OC, OC] to
      // oneDNN format [OC, 3OC]
      auto* weight_h_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      auto* user_weight_h_data = weight_h->data<U>();

      auto src1_iter = user_weight_h_data;
      auto src2_iter = user_weight_h_data + 2 * this->OC * this->OC;

      for (int64_t c = 0; c < this->OC; ++c) {
        memcpy(weight_h_data, src1_iter, 2 * this->OC * sizeof(U));
        memcpy(weight_h_data + 2 * this->OC, src2_iter, this->OC * sizeof(U));

        src1_iter += 2 * this->OC;
        src2_iter += this->OC;
        weight_h_data += 3 * this->OC;
      }

      weight_h_data = reinterpret_cast<U*>(user_memory.get_data_handle());

      if (origin_mode == false) {
        for (int64_t i = 0; i < this->OC; ++i) {
          for (int64_t j = 0; j < this->OC; ++j) {
            U minus_one(-1.0f);
            weight_h_data[j] = minus_one * weight_h_data[j];
          }
          weight_h_data += 3 * this->OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_iter_desc(), this->engine_);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wh_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(const phi::DenseTensor* bias,
                                                  const bool origin_mode) {
    const std::string bias_key = this->memory_key_ + "@bias";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(bias_key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->bias_desc(),
                                                this->engine_);
      auto* bias_data = reinterpret_cast<float*>(memory_p->get_data_handle());
      if (bias) {
        const float* user_bias_data =
            bias->data<float>();  // Bias in oneDNN is always float
        memcpy(bias_data, user_bias_data, sizeof(float) * this->G * this->OC);
      } else {
        // oneDNN always need bias memory, if it's not provided in PP, let
        // oneDNN allocate memory and set it to 0
        memset(bias_data, 0, sizeof(float) * this->G * this->OC);
      }

      if (origin_mode == false && bias) {
        for (int64_t i = 0; i < this->OC; ++i) {
          bias_data[i] *= -1;
        }
      }
      this->dev_ctx_.SetBlob(bias_key, memory_p);
    }
    return memory_p;
  }
};

template <typename T>
class FusionGRUMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool is_bf16 = std::is_same<T, paddle::platform::bfloat16>::value;
    const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    // BF16 does not support force output
    if (!is_bf16 && force_fp32_output) {
      RunKernel<float>(ctx);
    } else {
      RunKernel<T>(ctx);
    }
  }

  template <typename Tout = T>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    // Get Tensors
    const auto* input = ctx.Input<LoDTensor>("X");
    const auto* h0 = ctx.Input<phi::DenseTensor>("H0");
    const auto* weight_x = ctx.Input<phi::DenseTensor>("WeightX");
    const auto* weight_h = ctx.Input<phi::DenseTensor>("WeightH");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* hidden = ctx.Output<LoDTensor>("Hidden");
    auto x_dims = input->dims();
    auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                          ? phi::flatten_to_2d(x_dims, 1)
                          : x_dims;
    // Get attributes
    const bool is_reverse = ctx.Attr<bool>("is_reverse");
    const bool origin_mode = ctx.Attr<bool>("origin_mode");

    // Get tensor dimensions
    const auto x_mat_dims_vec = phi::vectorize(x_mat_dims);
    const auto weight_h_dims = phi::vectorize(weight_h->dims());
    const auto& input_lod = input->lod()[0];

    // Calculate RNN dimensions
    const int64_t N = input_lod.size() - 1;  // Number of sentences (batches)
    const int64_t Ti =  // Max length of the sentence in a batch
        [&input_lod]() {
          size_t res = 0;
          for (size_t i = 0; i < (input_lod.size() - 1); ++i) {
            res = std::max(res, input_lod[i + 1] - input_lod[i]);
          }
          return res;
        }();
    const int64_t IC = x_mat_dims_vec[1];  // Input channels
    const int64_t OC = weight_h_dims[0];   // Output channels

    GRUMKLDNNHandler<T, Tout> handler(
        ctx,
        dev_ctx,
        mkldnn_engine,
        ctx.GetPlace(),
        input,
        weight_h,
        h0,
        is_reverse,
        N,
        Ti,
        IC,
        OC,
        ctx.InputName("X") + ctx.InputName("WeightH"));

    auto input_memory_p =
        handler.AcquireInputMemoryWithReorder(input, is_reverse);

    std::shared_ptr<dnnl::memory> h0_memory_p, weight_h_memory_p,
        weight_x_memory_p;

    if (framework::TransToProtoVarType(weight_h->dtype()) ==
        paddle::framework::proto::VarType_Type_FP32) {
      h0_memory_p = handler.template AcquireH0Memory<float>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<float>(weight_x, origin_mode);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<float>(weight_h, origin_mode);
    } else if (framework::TransToProtoVarType(weight_h->dtype()) ==
               paddle::framework::proto::VarType_Type_BF16) {
      h0_memory_p =
          handler.template AcquireH0Memory<paddle::platform::bfloat16>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<paddle::platform::bfloat16>(
              weight_x, origin_mode);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<paddle::platform::bfloat16>(
              weight_h, origin_mode);
    } else {
      h0_memory_p = handler.template AcquireH0Memory<uint8_t>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<int8_t>(weight_x, origin_mode);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<int8_t>(weight_h, origin_mode);
    }

    auto bias_memory_p = handler.AcquireBiasMemory(bias, origin_mode);
    auto hidden_onednn_memory_p = handler.AcquireOutputMemory();

    std::unordered_map<int, dnnl::memory> gru_args = {
        {DNNL_ARG_SRC_LAYER, *input_memory_p},
        {DNNL_ARG_SRC_ITER, *h0_memory_p},
        {DNNL_ARG_WEIGHTS_LAYER, *weight_x_memory_p},
        {DNNL_ARG_WEIGHTS_ITER, *weight_h_memory_p},
        {DNNL_ARG_BIAS, *bias_memory_p},
        {DNNL_ARG_DST_LAYER, *hidden_onednn_memory_p}};

    auto gru_forward_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    gru_forward_p->execute(astream, gru_args);
    astream.wait();

    auto* hidden_onednn_data = hidden_onednn_memory_p->get_data_handle();
    auto* hidden_data =
        to_void_cast(hidden->mutable_data<Tout>(ctx.GetPlace()));
    if (handler.is_NTC()) {
      handler.reorderRNNdata(hidden_onednn_data,
                             hidden_data,
                             input_lod,
                             is_reverse,
                             platform::RNNReorderType::NTC_PP);
    } else {
      handler.reorderRNNdata(hidden_onednn_data,
                             hidden_data,
                             input_lod,
                             is_reverse,
                             platform::RNNReorderType::TNC_PP);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fusion_gru,
                   MKLDNN,
                   paddle::platform::CPUPlace,
                   ops::FusionGRUMKLDNNKernel<float>,
                   ops::FusionGRUMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FusionGRUMKLDNNKernel<uint8_t>);
