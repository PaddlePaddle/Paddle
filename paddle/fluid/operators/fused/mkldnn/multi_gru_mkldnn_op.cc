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

#include <initializer_list>
#include <iostream>
#include <memory>
#include "dnnl.hpp"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/multi_gru_op.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using paddle::platform::CPUDeviceContext;
using paddle::platform::CreateKey;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::MKLDNNMemDesc;
using platform::to_void_cast;
using phi::vectorize;
using Direction = dnnl::rnn_direction;

namespace {

// oneDNN RNN dimensions
const int64_t D = 1;  // Directions
const int64_t L = 1;  // Layers (PP supports only 1 stacked layer)
const int64_t G = 3;  // Number of Gates, 3 for GRU

constexpr Direction L2R = Direction::unidirectional_left2right;
constexpr Direction R2L = Direction::unidirectional_right2left;

constexpr const char* dir2str(Direction dir) {
  return dir == L2R ? "LR" : "RL";
}

}  // namespace

template <typename T, typename T_out = T>
class MultiGRUHandler {
 public:
  MultiGRUHandler(const paddle::framework::ExecutionContext& ctx,
                  const platform::MKLDNNDeviceContext& dev_ctx)
      : dev_ctx_(dev_ctx),
        engine_(dev_ctx.GetEngine()),
        place_(ctx.GetPlace()),
        origin_mode_(ctx.Attr<bool>("origin_mode")),
        layers_(ctx.Attr<int>("layers")),
        concat_pds_(layers_, std::shared_ptr<dnnl::concat::primitive_desc>()),
        x_(ctx.Input<LoDTensor>("X")),
        weights_x_(ctx.MultiInput<Tensor>("WeightX")),
        weights_h_(ctx.MultiInput<Tensor>("WeightH")),
        biases_(ctx.MultiInput<Tensor>("Bias")),
        hidden_(ctx.Output<LoDTensor>("Hidden")),
        x_lod_(x_->lod()[0]) {
    PADDLE_ENFORCE_EQ(
        weights_x_.size(), layers_ * 2,
        platform::errors::InvalidArgument("The number of WeightX inputs does "
                                          "not match the number of layers."));
    PADDLE_ENFORCE_EQ(
        weights_h_.size(), layers_ * 2,
        platform::errors::InvalidArgument("The number of WeightH inputs does "
                                          "not match the number of layers."));
    if (biases_.size() > 0)
      PADDLE_ENFORCE_EQ(
          biases_.size(), layers_ * 2,
          platform::errors::InvalidArgument("The number of Bias inputs does "
                                            "not match the number of layers."));
    // oneDNN kernel has hardcoded activation functions
    PADDLE_ENFORCE_EQ(
        ctx.Attr<std::string>("gate_activation"), "sigmoid",
        platform::errors::Unimplemented(
            "oneDNN fusion_gru supports only sigmoid as a gate activation."));
    PADDLE_ENFORCE_EQ(
        ctx.Attr<std::string>("activation"), "tanh",
        platform::errors::Unimplemented(
            "oneDNN fusion_gru supports only tanh as an activation."));

    N_ = x_lod_.size() - 1;  // Number of sentences (batches)
    Ti_ =                    // Max length of the sentence in a batch
        [this]() {
          size_t res = 0;
          for (size_t i = 0; i < (x_lod_.size() - 1); ++i) {
            res = std::max(res, x_lod_[i + 1] - x_lod_[i]);
          }
          return res;
        }();

    // Weights come in pairs, with the same dimensions within a pair
    for (int layer = 0; layer < layers_; ++layer) {
      ICs.push_back(vectorize(weights_x_[2 * layer]->dims())[0]);
      OCs.push_back(vectorize(weights_h_[2 * layer]->dims())[0]);
    }

    const std::string unique_name = ctx.OutputName("Hidden");
    // Create memory key without Ti because weights, bias and h0 memories
    // do not depend on Ti size but primitive and input/output memory do
    memory_key_ = platform::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx, CreateKey(dev_ctx, unique_name, MKLDNNGetDataType<T>()));
    key_ = memory_key_;
    key_.append("T").append(std::to_string(Ti_));

    // Is it int8 kernel
    const bool is_int8 = std::is_same<T, uint8_t>::value;

    // Create attributes for each oneDNN gru
    for (int i = 0; i < 2 * layers_; ++i) {
      attrs_.push_back(dnnl::primitive_attr());
    }

    if (is_int8) {
      // Add int8 attributes
      const auto scale_weights = ctx.MultiInput<LoDTensor>("Scale_weights");
      PADDLE_ENFORCE_EQ(
          scale_weights.size(), layers_ * 2,
          platform::errors::InvalidArgument(
              "The number of weight scale inputs does "
              "not match the number of layers. Expected: %d. Actual: %d",
              layers_ * 2, scale_weights.size()));
      const float scale_data = ctx.Attr<float>("Scale_data");
      const float shift_data = ctx.Attr<float>("Shift_data");

      const int weights_scale_mask =
          0 +
          (1 << 3)  // bit, indicating the unique scales for `g` dim in `ldigo`
          +
          (1 << 4);  // bit, indicating the unique scales for `o` dim in `ldigo`

      int w_scale_num = scale_weights.size();
      for (int i = 0; i < w_scale_num; ++i) {
        attrs_[i].set_rnn_data_qparams(scale_data, shift_data);
        const auto scale_weights_data = std::vector<float>(
            scale_weights[i]->data<float>(),
            scale_weights[i]->data<float>() + scale_weights[i]->numel());
        attrs_[i].set_rnn_weights_qparams(weights_scale_mask,
                                          scale_weights_data);
      }
    }

    for (int layer = 0; layer < layers_; ++layer) {
      AcquireGruPrimitiveDescriptor(layer, L2R);
      AcquireGruPrimitiveDescriptor(layer, R2L);
      AcquireConcatPrimitiveDescriptor(layer);
    }
  }

  void AcquireGruPrimitiveDescriptor(int layer, Direction dir) {
    auto pd_key = key_;
    pd_key.append("@gru_pd").append(dir2str(dir)).append(std::to_string(layer));
    auto pd = std::static_pointer_cast<dnnl::gru_forward::primitive_desc>(
        dev_ctx_.GetBlob(pd_key));
    if (pd == nullptr) {
      const bool is_int8 = std::is_same<T, uint8_t>::value;
      // Weights for int8 kernel are of a type s8
      const auto weights_dt =
          is_int8 ? dnnl::memory::data_type::s8 : dnnl::memory::data_type::f32;

      auto x_md = MKLDNNMemDesc({Ti_, N_, ICs[layer]}, MKLDNNGetDataType<T>(),
                                MKLDNNMemoryFormat::ntc);
      auto h0_md = MKLDNNMemDesc({L, D, N_, OCs[layer]}, MKLDNNGetDataType<T>(),
                                 MKLDNNMemoryFormat::ldnc);
      auto wx_md = MKLDNNMemDesc({L, D, ICs[layer], G, OCs[layer]}, weights_dt,
                                 MKLDNNMemoryFormat::any);
      auto wh_md = MKLDNNMemDesc({L, D, OCs[layer], G, OCs[layer]}, weights_dt,
                                 MKLDNNMemoryFormat::any);
      auto b_md =
          MKLDNNMemDesc({L, D, G, OCs[layer]}, MKLDNNGetDataType<float>(),
                        MKLDNNMemoryFormat::ldgo);
      auto h_md =
          MKLDNNMemDesc({Ti_, N_, OCs[layer]},
                        (layer == layers_ - 1) ? MKLDNNGetDataType<T_out>()
                                               : MKLDNNGetDataType<T>(),
                        MKLDNNMemoryFormat::ntc);

      auto desc = std::make_shared<dnnl::gru_forward::desc>(
          dnnl::prop_kind::forward_inference, dir, x_md, h0_md, wx_md, wh_md,
          b_md, h_md, dnnl::memory::desc());
      pd = std::make_shared<dnnl::gru_forward::primitive_desc>(
          *desc, attrs_[2 * layer + (dir == R2L)], engine_);
      PADDLE_ENFORCE_NOT_NULL(
          pd, platform::errors::InvalidArgument(
                  "Primitive descriptor for gru_forward cannot be null."));
      dev_ctx_.SetBlob(pd_key, pd);
    }
    gru_pds_[{layer, dir}] = pd;
  }

  void AcquireConcatPrimitiveDescriptor(int layer) {
    auto pd_key = key_;
    pd_key.append("@c_pd").append(std::to_string(layer));
    auto pd = std::static_pointer_cast<dnnl::concat::primitive_desc>(
        dev_ctx_.GetBlob(pd_key));
    if (pd == nullptr) {
      const int axis = 2;
      auto in_md =
          MKLDNNMemDesc({Ti_, N_, OCs[layer]},
                        (layer == layers_ - 1) ? MKLDNNGetDataType<T_out>()
                                               : MKLDNNGetDataType<T>(),
                        MKLDNNMemoryFormat::ntc);

      std::vector<dnnl::memory::desc> src_mds{in_md, in_md};
      pd = std::make_shared<dnnl::concat::primitive_desc>(axis, src_mds,
                                                          engine_);
      dev_ctx_.SetBlob(pd_key, pd);
    }
    concat_pds_[layer] = pd;
  }

  std::shared_ptr<dnnl::memory> AcquireInputMemoryWithReorder() {
    auto key = key_;
    key.append("@x_m");
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(gru_pds_[{0, L2R}]->src_desc(),
                                                engine_);
      dev_ctx_.SetBlob(key, memory_p);
    }

    auto* x_data = to_void_cast(x_->data<T>());

    auto* x_onednn_data = memory_p->get_data_handle();
    memset(x_onednn_data, 0, sizeof(T) * N_ * Ti_ * ICs[0]);

    if (platform::GetMKLDNNFormat(gru_pds_[{0, L2R}]->src_desc()) ==
        dnnl::memory::format_tag::ntc) {
      reorderPPtoNTC(x_data, x_onednn_data, x_lod_, 0, L2R);
    } else {
      reorderPPtoTNC(x_data, x_onednn_data, x_lod_, 0, L2R);
    }
    return memory_p;
  }

  // Reorder input memory [WORDS, C] + LoD -> [N, T, C]
  void reorderPPtoNTC(void* input_data, void* output_data,
                      std::vector<size_t> lod, int layer, Direction dir) {
    auto* input_data_iter = reinterpret_cast<T*>(input_data);
    auto* output_data_iter = reinterpret_cast<T*>(output_data);
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = (lod[n + 1] - lod[n]) * ICs[layer];
      const auto offset = dir == R2L ? (Ti_ * ICs[layer] - num_elements) : 0;
      memcpy(output_data_iter + n * Ti_ * ICs[layer] + offset, input_data_iter,
             sizeof(T) * num_elements);
      input_data_iter += num_elements;
    }
  }

  // Reorder input memory [WORDS, C] + LoD -> [T, N, C]
  void reorderPPtoTNC(void* input_data, void* output_data,
                      std::vector<size_t> lod, int layer, Direction dir) {
    auto* input_data_iter = reinterpret_cast<T*>(input_data);
    auto* output_data_iter = reinterpret_cast<T*>(output_data);
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = (lod[n + 1] - lod[n]);
      const auto offset = dir == R2L ? (Ti_ - num_elements) : 0;
      for (size_t t = 0; t < num_elements; ++t) {
        memcpy(
            output_data_iter + (t + offset) * N_ * ICs[layer] + n * ICs[layer],
            input_data_iter, sizeof(T) * ICs[layer]);
        input_data_iter += ICs[layer];
      }
    }
  }

  std::shared_ptr<dnnl::memory> executeSingleGru(
      std::shared_ptr<dnnl::memory> input_mem, int layer, Direction dir) {
    auto h0_mem = AcquireH0Memory(layer, dir);
    auto wx_mem = AcquireWeightXMemory(layer, dir);
    auto wh_mem = AcquireWeightHMemory(layer, dir);
    auto b_mem = AcquireBiasMemory(layer, dir);
    auto out_mem = AcquireGruOutputMemory(layer, dir);

    std::unordered_map<int, dnnl::memory> gru_args = {
        {DNNL_ARG_SRC_LAYER, *input_mem},  {DNNL_ARG_SRC_ITER, *h0_mem},
        {DNNL_ARG_WEIGHTS_LAYER, *wx_mem}, {DNNL_ARG_WEIGHTS_ITER, *wh_mem},
        {DNNL_ARG_BIAS, *b_mem},           {DNNL_ARG_DST_LAYER, *out_mem}};

    auto gru_forward_p0 = AcquireGruPrimitive(layer, dir);

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    gru_forward_p0->execute(astream, gru_args);
    astream.wait();
    return out_mem;
  }

  // TODO(grygielski) H0 is for now persistable
  std::shared_ptr<dnnl::memory> AcquireH0Memory(int layer, Direction dir) {
    auto key = memory_key_;
    key.append("@h0").append(dir2str(dir)).append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));
    if (!memory_p) {
      auto user_h0_memory = dnnl::memory();
      user_h0_memory = dnnl::memory({{1, 1, N_, OCs[layer]},
                                     MKLDNNGetDataType<float>(),
                                     MKLDNNMemoryFormat::ldnc},
                                    engine_);
      memset(user_h0_memory.get_data_handle(), 0,
             sizeof(float) * N_ * OCs[layer]);
      memory_p = std::make_shared<dnnl::memory>(
          gru_pds_[{layer, dir}]->src_iter_desc(), engine_);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      dnnl::reorder(user_h0_memory, *memory_p, attrs_[2 * layer + (dir == R2L)])
          .execute(astream, user_h0_memory, *memory_p);

      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireWeightXMemory(int layer, Direction dir) {
    auto key = memory_key_;
    key.append("@wx").append(dir2str(dir)).append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      auto user_md =
          MKLDNNMemDesc({1, 1, ICs[layer], 3, OCs[layer]},
                        MKLDNNGetDataType<float>(), MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, engine_);

      auto* weight_x_data =
          reinterpret_cast<float*>(user_memory.get_data_handle());
      int idx = layer * 2 + (dir == R2L);
      memcpy(weight_x_data, weights_x_[idx]->data<float>(),
             sizeof(float) * ICs[layer] * 3 * OCs[layer]);

      if (origin_mode_ == false) {
        for (int64_t i = 0; i < ICs[layer]; ++i) {
          for (int64_t j = 0; j < OCs[layer]; ++j) {
            weight_x_data[j] *= -1;
          }
          weight_x_data += 3 * OCs[layer];
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          gru_pds_[{layer, dir}]->weights_layer_desc(), engine_);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, attrs_[2 * layer + (dir == R2L)])
          .execute(astream, user_memory, *memory_p);

      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireWeightHMemory(int layer, Direction dir) {
    auto key = memory_key_;
    key.append("@wh").append(dir2str(dir)).append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      auto user_md =
          MKLDNNMemDesc({1, 1, OCs[layer], 3, OCs[layer]},
                        MKLDNNGetDataType<float>(), MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, engine_);

      // Reorder weights_h from PP format [OC, 2OC] + [OC, OC] to
      // oneDNN format [OC, 3OC]
      auto* weight_h_data =
          reinterpret_cast<float*>(user_memory.get_data_handle());

      int idx = layer * 2 + (dir == R2L);
      auto* user_weight_h_data = weights_h_[idx]->data<float>();

      auto src1_iter = user_weight_h_data;
      auto src2_iter = user_weight_h_data + 2 * OCs[layer] * OCs[layer];

      for (int64_t c = 0; c < OCs[layer]; ++c) {
        memcpy(weight_h_data, src1_iter, 2 * OCs[layer] * sizeof(float));
        memcpy(weight_h_data + 2 * OCs[layer], src2_iter,
               OCs[layer] * sizeof(float));

        src1_iter += 2 * OCs[layer];
        src2_iter += OCs[layer];
        weight_h_data += 3 * OCs[layer];
      }

      weight_h_data = reinterpret_cast<float*>(user_memory.get_data_handle());

      if (origin_mode_ == false) {
        for (int64_t i = 0; i < OCs[layer]; ++i) {
          for (int64_t j = 0; j < OCs[layer]; ++j) {
            weight_h_data[j] *= -1;
          }
          weight_h_data += 3 * OCs[layer];
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          gru_pds_[{layer, dir}]->weights_iter_desc(), engine_);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, attrs_[2 * layer + (dir == R2L)])
          .execute(astream, user_memory, *memory_p);

      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(int layer, Direction dir) {
    auto key = memory_key_;
    key.append("@b").append(dir2str(dir)).append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(
          gru_pds_[{layer, dir}]->bias_desc(), engine_);
      auto* bias_data = reinterpret_cast<float*>(memory_p->get_data_handle());

      int idx = layer * 2 + (dir == R2L);
      if (biases_.size() > 0 && biases_[idx]) {
        const float* user_bias_data =
            biases_[idx]->data<float>();  // Bias in oneDNN is always float
        memcpy(bias_data, user_bias_data, sizeof(float) * 3 * OCs[layer]);
      } else {
        // oneDNN always need bias memory, if it's not provided in PP, let
        // oneDNN allocate memory and set it to 0
        memset(bias_data, 0, sizeof(float) * 3 * OCs[layer]);
      }

      if (origin_mode_ == false && biases_.size() && biases_[idx]) {
        for (int64_t i = 0; i < OCs[layer]; ++i) {
          bias_data[i] *= -1;
        }
      }
      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireGruOutputMemory(int layer,
                                                       Direction dir) {
    auto key = key_;
    key.append("@h_m").append(dir2str(dir)).append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(
          gru_pds_[{layer, dir}]->dst_desc(), engine_);
      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::gru_forward> AcquireGruPrimitive(int layer,
                                                         Direction dir) {
    auto key = key_;
    key.append("@gru_p").append(dir2str(dir)).append(std::to_string(layer));
    auto prim =
        std::static_pointer_cast<dnnl::gru_forward>(dev_ctx_.GetBlob(key));
    if (prim == nullptr) {
      prim = std::make_shared<dnnl::gru_forward>(*gru_pds_[{layer, dir}]);
      dev_ctx_.SetBlob(key, prim);
    }
    return prim;
  }

  void reorderInputL2RtoR2L(std::shared_ptr<dnnl::memory> mem, int layer) {
    auto* data = mem->get_data_handle();
    auto* data_iter = reinterpret_cast<T*>(data);
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = (x_lod_[n + 1] - x_lod_[n]) * ICs[layer];
      const auto offset = Ti_ * ICs[layer] - num_elements;
      memmove(data_iter + offset, data_iter, sizeof(T) * num_elements);
      memset(data_iter, 0, sizeof(T) * offset);
      data_iter += Ti_ * ICs[layer];
    }
  }

  template <typename K>
  void reorderOutputR2LtoL2R(std::shared_ptr<dnnl::memory> mem, int layer) {
    auto* data = mem->get_data_handle();
    auto* data_iter = reinterpret_cast<K*>(data);
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = (x_lod_[n + 1] - x_lod_[n]) * OCs[layer];
      const auto offset = Ti_ * OCs[layer] - num_elements;
      memmove(data_iter, data_iter + offset, sizeof(K) * num_elements);
      memset(data_iter + num_elements, 0, sizeof(K) * offset);
      data_iter += Ti_ * OCs[layer];
    }
  }

  std::shared_ptr<dnnl::memory> executeConcat(
      std::shared_ptr<dnnl::memory> mem1, std::shared_ptr<dnnl::memory> mem2,
      int layer) {
    auto out_mem = AcquireConcatOutputMemory(layer);

    std::unordered_map<int, dnnl::memory> concat_args{
        {DNNL_ARG_MULTIPLE_SRC, *mem1},
        {DNNL_ARG_MULTIPLE_SRC + 1, *mem2},
        {DNNL_ARG_DST, *out_mem}};

    auto concat_p = AcquireConcatPrimitive(layer);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    concat_p->execute(astream, concat_args);
    astream.wait();
    return out_mem;
  }

  std::shared_ptr<std::vector<dnnl::memory>> AcquireConcatInputMemories(
      int layer) {
    auto key = key_;
    key.append("@ci_m").append(std::to_string(layer));
    auto memory_p = std::static_pointer_cast<std::vector<dnnl::memory>>(
        dev_ctx_.GetBlob(key));

    if (!memory_p) {
      std::vector<dnnl::memory> src_mems{
          dnnl::memory(concat_pds_[layer]->src_desc(0), engine_),
          dnnl::memory(concat_pds_[layer]->src_desc(1), engine_)};
      memory_p = std::make_shared<std::vector<dnnl::memory>>(src_mems);
      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireConcatOutputMemory(int layer) {
    auto key = key_;
    key.append("@co_m").append(std::to_string(layer));
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(concat_pds_[layer]->dst_desc(),
                                                engine_);
      dev_ctx_.SetBlob(key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::concat> AcquireConcatPrimitive(int layer) {
    auto key = key_;
    key.append("@c_p").append(std::to_string(layer));
    auto prim = std::static_pointer_cast<dnnl::concat>(dev_ctx_.GetBlob(key));
    if (prim == nullptr) {
      prim = std::make_shared<dnnl::concat>(*concat_pds_[layer]);
      dev_ctx_.SetBlob(key, prim);
    }
    return prim;
  }

  template <typename Tout>
  void reorderOutput(std::shared_ptr<dnnl::memory> mem, int layer) {
    auto* data = mem->get_data_handle();
    auto* hidden_data = to_void_cast(hidden_->mutable_data<Tout>(place_));
    if (isNTC(layers_ - 1)) {
      reorderNTCtoPP(data, hidden_data, layers_ - 1);
    } else {
      reorderTNCtoPP(data, hidden_data, layers_ - 1);
    }
  }

  bool isNTC(int layer) {
    return (platform::GetMKLDNNFormat(gru_pds_[{layer, L2R}]->dst_desc()) ==
            dnnl::memory::format_tag::ntc);
  }

  int getLayers() const { return layers_; }

  // Reorder output values to PP format [N, T, C] -> [WORDS, C]
  void reorderNTCtoPP(void* input_data, void* output_data, int layer) {
    auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
    auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
    auto oc = OCs[layer] * 2;
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = (x_lod_[n + 1] - x_lod_[n]) * oc;
      memcpy(output_data_iter, input_data_iter + n * Ti_ * oc,
             sizeof(T_out) * num_elements);
      output_data_iter += num_elements;
    }
  }

  // Reorder output values to PP format [T, N, C] -> [WORDS, C]
  void reorderTNCtoPP(void* input_data, void* output_data, int layer) {
    auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
    auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
    for (int n = 0; n < N_; ++n) {
      const auto num_elements = x_lod_[n + 1] - x_lod_[n];
      for (size_t t = 0; t < num_elements; ++t) {
        memcpy(output_data_iter,
               input_data_iter + t * N_ * OCs[layer] + n * OCs[layer],
               sizeof(T_out) * OCs[layer]);
        output_data_iter += OCs[layer];
      }
    }
  }

 private:
  // RNN dimensions
  // N - Batch Size
  // Ti - Max sentence length
  // ICs - Input Channels
  // OCs - Output Channels
  int64_t N_, Ti_;
  std::vector<int64_t> ICs, OCs;

  const platform::MKLDNNDeviceContext& dev_ctx_;
  const dnnl::engine engine_;
  const platform::Place place_;
  const bool origin_mode_;
  const int layers_;

  std::map<std::pair<int, Direction>,
           std::shared_ptr<dnnl::gru_forward::primitive_desc>>
      gru_pds_;
  std::vector<std::shared_ptr<dnnl::concat::primitive_desc>> concat_pds_;

  std::string key_;
  // Memory size of weights, bias and h0 does not depend
  // on Ti size, thus we need another key to cache them
  std::string memory_key_;

  const LoDTensor* x_;
  const std::vector<const Tensor*> weights_x_;
  const std::vector<const Tensor*> weights_h_;
  const std::vector<const Tensor*> biases_;
  LoDTensor* hidden_;
  std::vector<dnnl::primitive_attr> attrs_;
  const paddle::framework::Vector<size_t>& x_lod_;
};

template <typename T>
class MultiGRUMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool force_fp32_output =
        ctx.HasAttr("force_fp32_output") && ctx.Attr<bool>("force_fp32_output");

    if (force_fp32_output) {
      RunKernel<float>(ctx);
    } else {
      RunKernel<T>(ctx);
    }
  }

  template <typename Tout = T>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    MultiGRUHandler<T, Tout> handler(ctx, dev_ctx);

    int layers = handler.getLayers();
    auto input_mem = handler.AcquireInputMemoryWithReorder();
    for (int layer = 0; layer < layers; ++layer) {
      auto gru_out_L2R = handler.executeSingleGru(input_mem, layer, L2R);
      handler.reorderInputL2RtoR2L(input_mem, layer);
      auto gru_out_R2L = handler.executeSingleGru(input_mem, layer, R2L);
      if (layer < layers - 1)
        handler.template reorderOutputR2LtoL2R<T>(gru_out_R2L, layer);
      else
        handler.template reorderOutputR2LtoL2R<Tout>(gru_out_R2L, layer);
      input_mem = handler.executeConcat(gru_out_L2R, gru_out_R2L, layer);
    }
    handler.template reorderOutput<Tout>(input_mem, layers - 1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(multi_gru, MKLDNN, paddle::platform::CPUPlace,
                   ops::MultiGRUMKLDNNKernel<float>,
                   ops::MultiGRUMKLDNNKernel<uint8_t>);
