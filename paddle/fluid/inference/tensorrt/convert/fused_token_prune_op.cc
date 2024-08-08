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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"

namespace paddle::inference::tensorrt {

class FusedTokenPruneOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;
    auto* Attn = engine_->GetITensor(op_desc.Input("Attn").front());
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Mask = engine_->GetITensor(op_desc.Input("Mask").front());
    auto* NewMask = engine_->GetITensor(op_desc.Input("NewMask").front());
    bool keep_first_token =
        op_desc.HasAttr("keep_first_token")
            ? PADDLE_GET_CONST(bool, op_desc.GetAttr("keep_first_token"))
            : true;
    bool keep_order =
        op_desc.HasAttr("keep_order")
            ? PADDLE_GET_CONST(bool, op_desc.GetAttr("keep_order"))
            : false;
    auto output_name = op_desc.Output("SlimmedX")[0];
    auto out_inds_name = op_desc.Output("CLSInds")[0];
    if (engine_->with_dynamic_shape()) {
      // reduce_sum: (-1,headsize,token_length,token_length) ->
      // (-1,token_length)
      uint32_t reduce_dim = 0;
      reduce_dim |= 1 << 1;  // 00000000000000000000000000000010
      reduce_dim |= 1 << 2;  // 00000000000000000000000000000110
      bool keep_dim = false;
      nvinfer1::ReduceOperation reduce_type = nvinfer1::ReduceOperation::kSUM;
      auto* reduce_sum_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Reduce, *Attn, reduce_type, reduce_dim, keep_dim);
      auto* Reduced = reduce_sum_layer->getOutput(0);

      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

      if (engine_->precision() == phi::DataType::INT8) {
        with_fp16 = true;
      }
      bool flag_varseqlen = engine_->use_varseqlen();
      plugin::FusedTokenPrunePluginDynamic* plugin =
          new plugin::FusedTokenPrunePluginDynamic(
              with_fp16, keep_first_token, keep_order, flag_varseqlen);
      if (flag_varseqlen) {
        auto* word_id = engine_->GetITensor("word_id");
        auto* pos_id = engine_->GetITensor("pos_id");
        auto* mask_id = engine_->GetITensor("mask_id");

        std::vector<nvinfer1::ITensor*> itensors = {
            Reduced, X, Mask, NewMask, word_id, pos_id, mask_id};
        layer = engine_->AddDynamicPlugin(
            itensors.data(), itensors.size(), plugin);  // inputs'number: 7

        layer->getOutput(0)->setName(output_name.c_str());
        engine_->SetITensor(output_name, layer->getOutput(0));

        layer->getOutput(1)->setName(out_inds_name.c_str());
        engine_->SetITensor(out_inds_name, layer->getOutput(1));

        engine_->DeleteITensor("word_id", word_id);
        layer->getOutput(2)->setName("word_id_after_token_prune");
        engine_->SetITensor("word_id", layer->getOutput(2));

        engine_->DeleteITensor("pos_id", pos_id);
        layer->getOutput(3)->setName("pos_id_after_token_prune");
        engine_->SetITensor("pos_id", layer->getOutput(3));

        engine_->DeleteITensor("mask_id", mask_id);
        layer->getOutput(4)->setName("mask_id_after_token_prune");
        engine_->SetITensor("mask_id", layer->getOutput(4));
      } else {
        std::vector<nvinfer1::ITensor*> itensors = {Reduced, X, Mask, NewMask};
        layer = engine_->AddDynamicPlugin(
            itensors.data(), itensors.size(), plugin);  // inputs'number: 4

        layer->getOutput(0)->setName(output_name.c_str());
        engine_->SetITensor(output_name, layer->getOutput(0));

        layer->getOutput(1)->setName(out_inds_name.c_str());
        engine_->SetITensor(out_inds_name, layer->getOutput(1));
      }
      layer->setName(
          ("fused_token_prune(Output: " + output_name + ")").c_str());
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(fused_token_prune, FusedTokenPruneOpConverter);
