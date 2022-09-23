/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/inference/tensorrt/convert/utils.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class EmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid EmbEltwiseLayerNorm op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    auto word_id_name = op_desc.Input("WordId").front();
    auto pos_id_name = engine_->tensorrt_transformer_posid();
    engine_->Set("ernie_pos_name", new std::string(pos_id_name));

    auto sent_id_name = op_desc.Input("SentId").front();
    auto mask_id_name = engine_->tensorrt_transformer_maskid();
    auto word_emb_name = op_desc.Input("WordEmbedding").front();
    auto pos_emb_name = op_desc.Input("PosEmbedding").front();
    auto sent_emb_name = op_desc.Input("SentEmbedding").front();

    std::vector<std::string> id_names;
    std::vector<std::string> emb_names;
    bool flag_varseqlen =
        engine_->use_varseqlen() && pos_id_name != "" && mask_id_name != "";

    if (flag_varseqlen) {
      engine_->SetITensor("word_id", engine_->GetITensor(word_id_name));
      engine_->SetITensor("pos_id", engine_->GetITensor(pos_id_name));
      engine_->SetITensor("mask_id", engine_->GetITensor(mask_id_name));
      id_names =
          std::vector<std::string>{word_id_name, pos_id_name, sent_id_name};
      emb_names =
          std::vector<std::string>{word_emb_name, pos_emb_name, sent_emb_name};

      auto mask_id_tensor = engine_->GetITensor("mask_id");
      auto mask_dims = mask_id_tensor->getDimensions();
      auto slice_start_dims = mask_dims;
      auto slice_stride_dims = mask_dims;

      for (int i = 0; i < mask_dims.nbDims; i++) {
        slice_start_dims.d[i] = 0;
        slice_stride_dims.d[i] = 1;
      }

      auto* shape_tensor = Shape(mask_id_tensor);
      std::vector<nvinfer1::ITensor*> start_vec_tensor;
      std::vector<nvinfer1::ITensor*> size_vec_tensor;
      for (int i = 0; i < mask_dims.nbDims; i++) {
        start_vec_tensor.push_back(Add1DConstantLayer(0));
        size_vec_tensor.push_back(Add1DConstantLayer(1));
      }
      size_vec_tensor[1] = GetEleTensorOfShape(shape_tensor, 1);

      auto start_tensor = Concat(start_vec_tensor);
      auto size_tensor = Concat(size_vec_tensor);

      auto slice_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               Slice,
                               *mask_id_tensor,
                               slice_start_dims,
                               slice_start_dims,
                               slice_stride_dims);  // unuseful slice_start_dims
      slice_layer->setInput(1, *start_tensor);
      slice_layer->setInput(2, *size_tensor);
      slice_layer->setName(
          ("Embeltwise_slice_layer (Output: slice_max_seqlen " +
           op_desc.Output("Out")[0] + ")")
              .c_str());
      engine_->SetTensorDynamicRange(slice_layer->getOutput(0), 1.0f);

      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *slice_layer->getOutput(0));
      nvinfer1::Dims shape_dim;
      shape_dim.nbDims = 1;
      shape_dim.d[0] = -1;
      reshape_layer->setReshapeDimensions(shape_dim);
      reshape_layer->setName(("Embeltwise_reshape_layer (Output: max_seqlen " +
                              op_desc.Output("Out")[0] + ")")
                                 .c_str());
      engine_->SetTensorDynamicRange(reshape_layer->getOutput(0), 1.0f);
      engine_->SetITensor("max_seqlen_tensor", reshape_layer->getOutput(0));
    } else {
      id_names = op_desc.Input("Ids");
      emb_names = op_desc.Input("Embs");
    }

    int input_num = id_names.size();

    // Declare inputs
    std::vector<nvinfer1::ITensor*> input_ids;
    for (int i = 0; i < input_num; i++) {
      input_ids.push_back(engine_->GetITensor(id_names[i]));
    }

    // input_embs[0]: word_embedding
    // input_embs[1]: pos_embedding
    // input_embs[2]: sent_embedding
    std::vector<nvinfer1::Weights> input_embs;
    std::vector<int> emb_sizes;

    // get the presistable var's data
    auto GetWeight = [&](const std::string& var_name,
                         framework::DDim* dim) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      *dim = temp_tensor->dims();
      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };

    auto GetFp16Weight = [&](const std::string& var_name,
                             framework::DDim* dim) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      *dim = temp_tensor->dims();
      auto weight = engine_->GetFp16TrtWeight(var_name, *temp_tensor);
      return weight;
    };

    auto GetFp32Weight = [&](const std::string& var_name,
                             framework::DDim* dim) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      *dim = temp_tensor->dims();
      auto weight = engine_->GetFp32TrtWeight(var_name, *temp_tensor);
      return weight;
    };
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    int hidden = 0;
    for (int i = 0; i < input_num; i++) {
      framework::DDim emb_dims;
      TensorRTEngine::Weight weight;
      if (flag_varseqlen) {
        weight = GetWeight(emb_names[i], &emb_dims);
      } else {
        if (with_fp16) {
          weight = GetFp16Weight(emb_names[i], &emb_dims);
        } else {
          weight = GetFp32Weight(emb_names[i], &emb_dims);
        }
      }
      input_embs.push_back(weight.get());
      emb_sizes.push_back(weight.get().count);
      PADDLE_ENFORCE_EQ(
          emb_dims.size(),
          2,
          platform::errors::InvalidArgument(
              "The fused EmbEltwiseLayerNorm's emb should be 2 dims."));
      hidden = emb_dims[1];
    }

    framework::DDim bias_dims, scale_dims;
    TensorRTEngine::Weight bias_weight, scale_weight;
    if (flag_varseqlen) {
      bias_weight = GetWeight(op_desc.Input("Bias").front(), &bias_dims);
      scale_weight = GetWeight(op_desc.Input("Scale").front(), &scale_dims);
    } else {
      if (with_fp16) {
        bias_weight = GetFp16Weight(op_desc.Input("Bias").front(), &bias_dims);
        scale_weight =
            GetFp16Weight(op_desc.Input("Scale").front(), &scale_dims);
      } else {
        bias_weight = GetFp32Weight(op_desc.Input("Bias").front(), &bias_dims);
        scale_weight =
            GetFp32Weight(op_desc.Input("Scale").front(), &scale_dims);
      }
    }

    int64_t bias_size = phi::product(bias_dims);
    int64_t scale_size = phi::product(scale_dims);
    nvinfer1::ILayer* layer = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");

    if (flag_varseqlen) {
      int output_fp16 = static_cast<int>((engine_->WithFp16() == 1) ? 1 : 0);
      if (enable_int8) {
        output_fp16 = 1;
      }
      PADDLE_ENFORCE_EQ(
          input_num,
          3,
          platform::errors::InvalidArgument(
              "When using oss and var-len, embedding_eltwise_layernorm op"
              "should have 3 inputs only, but got %d.",
              input_num));
      PADDLE_ENFORCE_EQ(
          output_fp16,
          1,
          platform::errors::InvalidArgument(
              "Only Precision::KHalf(fp16) is supported when infering "
              "ernie(bert) model with config.EnableVarseqlen(). "
              "But Precision::KFloat32 is setted."));
      const std::vector<nvinfer1::PluginField> fields{
          {"bert_embeddings_layernorm_beta",
           bias_weight.get().values,
           GetPluginFieldType(bias_weight.get().type),
           static_cast<int32_t>(bias_size)},
          {"bert_embeddings_layernorm_gamma",
           scale_weight.get().values,
           GetPluginFieldType(scale_weight.get().type),
           static_cast<int32_t>(scale_size)},
          {"bert_embeddings_word_embeddings",
           input_embs[0].values,
           GetPluginFieldType(input_embs[0].type),
           static_cast<int32_t>(emb_sizes[0])},
          {"bert_embeddings_token_type_embeddings",
           input_embs[2].values,
           GetPluginFieldType(input_embs[2].type),
           static_cast<int32_t>(emb_sizes[2])},
          {"bert_embeddings_position_embeddings",
           input_embs[1].values,
           GetPluginFieldType(input_embs[1].type),
           static_cast<int32_t>(emb_sizes[1])},
          {"output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1},
      };

      nvinfer1::PluginFieldCollection* plugin_ptr =
          static_cast<nvinfer1::PluginFieldCollection*>(
              malloc(sizeof(*plugin_ptr) +
                     fields.size() * sizeof(nvinfer1::PluginField)));
      plugin_ptr->nbFields = static_cast<int>(fields.size());
      plugin_ptr->fields = fields.data();

      std::vector<nvinfer1::ITensor*> plugin_inputs;
      plugin_inputs.emplace_back(
          engine_->GetITensor(word_id_name));  // word_embedding,
                                               // eval_placeholder_0
      plugin_inputs.emplace_back(
          engine_->GetITensor(sent_id_name));  // sent_embedding,
                                               // eval_placeholder_1
      plugin_inputs.emplace_back(
          engine_->GetITensor(pos_id_name));  // cu_seqlens,
                                              // eval_placeholder_2
      plugin_inputs.emplace_back(engine_->GetITensor(
          "max_seqlen_tensor"));  // max_seqlen, eval_placeholder_3

      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomEmbLayerNormPluginDynamic", "2");

      auto plugin_obj =
          creator->createPlugin("CustomEmbLayerNormPluginDynamic", plugin_ptr);
      auto plugin_layer = engine_->network()->addPluginV2(
          plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);
      plugin_layer->setName(("CustomEmbLayerNormPluginDynamic_V2(Output: " +
                             op_desc.Output("Out")[0] + ")")
                                .c_str());
      free(plugin_ptr);
      if (enable_int8) {
        float out_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(0), out_scale);
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(1), out_scale);
      }
      if (engine_->with_interleaved()) {
        VLOG(4) << "fused emb_eltwise_layernorm op: use_varseqlen and "
                   "with_interleaved";
        if (!enable_int8) {
          PADDLE_THROW(
              platform::errors::Fatal("use with_interleaved must be int8."));
        }
        auto* shuffler_embed = TRT_ENGINE_ADD_LAYER(
            engine_, Shuffle, *(plugin_layer->getOutput(0)));
        nvinfer1::Permutation transpose_embed{2, 1, 0, 3};
        shuffler_embed->setSecondTranspose(transpose_embed);
        engine_->SetITensor(op_desc.Output("Out")[0],
                            shuffler_embed->getOutput(0));
        shuffler_embed->setName(
            ("Emb_eltwise_out_shuffler_transpose (Output: " +
             op_desc.Output("Out")[0] + ")")
                .c_str());
      } else {
        layer = plugin_layer;
        auto output_name = op_desc.Output("Out")[0];
        RreplenishLayerAndOutput(layer,
                                 "CustomEmbLayerNormPluginDynamic_V2",
                                 {output_name, std::string("qkv_plugin_mask")},
                                 test_mode);
      }
    } else {
      float eps = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));
      plugin::DynamicPluginTensorRT* plugin = nullptr;
      std::vector<void*> input_embs_data;
      for (size_t i = 0; i < input_embs.size(); ++i) {
        input_embs_data.push_back(const_cast<void*>(
            reinterpret_cast<const void*>(input_embs[i].values)));
      }
      plugin = new plugin::EmbEltwiseLayernormPluginDynamic(
          input_embs_data,
          const_cast<void*>(static_cast<const void*>(bias_weight.get().values)),
          const_cast<void*>(
              static_cast<const void*>(scale_weight.get().values)),
          emb_sizes,
          bias_size,
          scale_size,
          hidden,
          eps,
          with_fp16);
      layer = engine_->AddDynamicPlugin(input_ids.data(), input_num, plugin);
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(
          layer, "emb_eltwise_layernorm", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_embedding_eltwise_layernorm,
                          EmbEltwiseLayerNormOpConverter);
