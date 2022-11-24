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
#include "paddle/fluid/inference/tensorrt/convert/utils.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_varseqlen_plugin.h"

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

class PrelnEmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert fluid PrelnEmbEltwiseLayerNorm op to tensorrt layer";
    // get the presistable var's data
    auto GetWeight = [&](const std::string& var_name,
                         framework::DDim* dim) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<phi::DenseTensor>();
      *dim = temp_tensor->dims();
      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };

    auto pos_id_name = engine_->tensorrt_transformer_posid();
    auto mask_id_name = engine_->tensorrt_transformer_maskid();
    bool flag_prelayernorm = engine_->with_interleaved() &&
                             engine_->use_varseqlen() && pos_id_name != "" &&
                             mask_id_name != "";

    if (!flag_prelayernorm) {
      PADDLE_THROW(platform::errors::Fatal(
          "PrelnErnie: If you want to use varseqlen, must be with interleaved, "
          "set pos_id_name, set mask_id_name."));
    }
    framework::OpDesc op_desc(op, nullptr);
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    if (!enable_int8) {
      PADDLE_THROW(
          platform::errors::Fatal("use with_interleaved must be int8."));
    }
<<<<<<< HEAD
    auto word_id_name = op_desc.Input("WordId").front();
    engine_->Set("ernie_pos_name", new std::string(pos_id_name));
=======
    // Declare inputs
    std::vector<nvinfer1::ITensor*> input_ids;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    // Declare inputs_weight
    std::vector<nvinfer1::Weights> input_embs;
    std::vector<int> emb_sizes;
    TensorRTEngine::Weight weight;
    framework::DDim emb_dims;
    framework::DDim bias_dims, scale_dims;
    TensorRTEngine::Weight bias_weight, scale_weight;

<<<<<<< HEAD
    engine_->SetITensor("word_id", engine_->GetITensor(word_id_name));
    engine_->SetITensor("pos_id", engine_->GetITensor(pos_id_name));
    engine_->SetITensor("mask_id", engine_->GetITensor(mask_id_name));

    std::vector<std::string> emb_names;
    emb_names =
        std::vector<std::string>{word_emb_name, pos_emb_name, sent_emb_name};
=======
    int64_t bias_size = phi::product(bias_dims);
    int64_t scale_size = phi::product(scale_dims);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    std::vector<std::string> id_names = op_desc.Input("Ids");
    std::vector<std::string> emb_names = op_desc.Input("Embs");
    int input_num = id_names.size();

    engine_->SetITensor("pos_id", engine_->GetITensor(pos_id_name));
    engine_->SetITensor("mask_id", engine_->GetITensor(mask_id_name));

    auto mask_id_tensor = engine_->GetITensor("mask_id");
    auto mask_dims = mask_id_tensor->getDimensions();
    auto slice_start_dims = mask_dims;
    auto slice_stride_dims = mask_dims;

<<<<<<< HEAD
      auto* temp_data = const_cast<float*>(static_cast<const float*>(
          engine_->GetFp32TrtWeight(var_name, *temp_tensor).get().values));
      return temp_data;
    };

    for (int i = 0; i < input_num; i++) {
      framework::DDim emb_dims;
      float* emb_data = get_persistable_data(emb_names[i], &emb_dims);
      int64_t emb_size = phi::product(emb_dims);
      input_embs.push_back(emb_data);
      emb_sizes.push_back(emb_size);
      PADDLE_ENFORCE_EQ(
          emb_dims.size(),
          2,
          platform::errors::InvalidArgument(
              "The fused PrelnEmbEltwiseLayerNorm's emb should be 2 dims."));
=======
    for (int i = 0; i < mask_dims.nbDims; i++) {
      slice_start_dims.d[i] = 0;
      slice_stride_dims.d[i] = 1;
    }

    auto* shape_tensor = Shape(mask_id_tensor);
    std::vector<nvinfer1::ITensor*> size_vec_tensor;
    std::vector<nvinfer1::ITensor*> start_vec_tensor;
    for (int i = 0; i < mask_dims.nbDims; i++) {
      size_vec_tensor.push_back(Add1DConstantLayer(1));
      start_vec_tensor.push_back(Add1DConstantLayer(0));
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }
    size_vec_tensor[1] = GetEleTensorOfShape(shape_tensor, 1);
    auto size_tensor = Concat(size_vec_tensor);
    auto start_tensor = Concat(start_vec_tensor);

    auto slice_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Slice,
                             *mask_id_tensor,
                             slice_start_dims,
                             slice_start_dims,
                             slice_stride_dims);  // unuseful slice_start_dims
    slice_layer->setInput(1, *start_tensor);
    slice_layer->setInput(2, *size_tensor);
    slice_layer->setName(("Embeltwise_slice_layer (Output: slice_max_seqlen " +
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

    for (int i = 0; i < input_num; i++) {
      auto input_tensor = engine_->GetITensor(id_names[i]);
      weight = GetWeight(emb_names[i], &emb_dims);
      if (id_names[i] == pos_id_name) {
        input_ids.insert(input_ids.begin(), input_tensor);
        input_embs.insert(input_embs.begin(), weight.get());
        emb_sizes.insert(emb_sizes.begin(), weight.get().count);
      } else {
        input_ids.push_back(input_tensor);
        input_embs.push_back(weight.get());
        emb_sizes.push_back(weight.get().count);
      }
    }
    bias_weight = GetWeight(op_desc.Input("Bias").front(), &bias_dims);
    scale_weight = GetWeight(op_desc.Input("Scale").front(), &scale_dims);
    bias_size = phi::product(bias_dims);
    scale_size = phi::product(scale_dims);
    // other_id(except pos_id)
    engine_->SetITensor("word_id", input_ids[1]);

    int output_fp16 = static_cast<int>((engine_->WithFp16() == 1) ? 1 : 0);
    if (enable_int8) {
      output_fp16 = 1;
    }
    PADDLE_ENFORCE_EQ(
<<<<<<< HEAD
        input_num,
        3,
        platform::errors::InvalidArgument(
            "When using oss and var-len, embedding_eltwise_layernorm op"
            "should have 3 inputs only, but got %d.",
            input_num));
    const std::vector<nvinfer1::PluginField> fields{
        {"bert_embeddings_layernorm_beta",
         bias,
         nvinfer1::PluginFieldType::kFLOAT32,
         static_cast<int32_t>(bias_size)},
        {"bert_embeddings_layernorm_gamma",
         scale,
         nvinfer1::PluginFieldType::kFLOAT32,
         static_cast<int32_t>(scale_size)},
        {"bert_embeddings_word_embeddings",
         input_embs[0],
         nvinfer1::PluginFieldType::kFLOAT32,
         static_cast<int32_t>(emb_sizes[0])},
        {"bert_embeddings_token_type_embeddings",
         input_embs[2],
         nvinfer1::PluginFieldType::kFLOAT32,
         static_cast<int32_t>(emb_sizes[2])},
        {"bert_embeddings_position_embeddings",
         input_embs[1],
         nvinfer1::PluginFieldType::kFLOAT32,
         static_cast<int32_t>(emb_sizes[1])},
        {"output_fp16", &output_int8, nvinfer1::PluginFieldType::kINT32, 1},
    };
=======
        output_fp16,
        1,
        platform::errors::InvalidArgument(
            "Only Precision::KHalf(fp16) is supported when infering "
            "ernie(bert) model with config.EnableVarseqlen(). "
            "But Precision::KFloat32 is setted."));

    std::vector<nvinfer1::PluginField> fields;
    std::vector<std::string> temp_fields_keys;
    fields.emplace_back("bert_embeddings_layernorm_beta",
                        bias_weight.get().values,
                        GetPluginFieldType(bias_weight.get().type),
                        static_cast<int32_t>(bias_size));
    fields.emplace_back("bert_embeddings_layernorm_gamma",
                        scale_weight.get().values,
                        GetPluginFieldType(scale_weight.get().type),
                        static_cast<int32_t>(scale_size));
    fields.emplace_back(
        "output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1);
    for (int i = 0; i < input_num; ++i) {
      temp_fields_keys.push_back("bert_embeddings_word_embeddings_" +
                                 std::to_string(i));
      fields.emplace_back(temp_fields_keys.rbegin()->c_str(),
                          input_embs[i].values,
                          GetPluginFieldType(input_embs[i].type),
                          static_cast<int32_t>(emb_sizes[i]));
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    nvinfer1::PluginFieldCollection* plugin_ptr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_ptr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_ptr->nbFields = static_cast<int>(fields.size());
    plugin_ptr->fields = fields.data();

<<<<<<< HEAD
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
    auto mask_id_tensor = engine_->GetITensor("mask_id");
    auto mask_dims = mask_id_tensor->getDimensions();
    auto slice_start_dims = mask_dims;
    auto slice_size_dims = mask_dims;
    auto slice_stride_dims = mask_dims;

    for (int i = 0; i < mask_dims.nbDims; i++) {
      slice_start_dims.d[i] = 0;
      slice_size_dims.d[i] = 1;
      slice_stride_dims.d[i] = 1;
    }
    slice_size_dims.d[1] = mask_dims.d[1];
    auto* slice_size_tensor = Add1DConstantLayer(slice_size_dims);
    auto slice_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Slice,
                             *mask_id_tensor,
                             slice_start_dims,
                             slice_start_dims,
                             slice_stride_dims);  // unuseful slice_start_dims
    slice_layer->setInput(2, *slice_size_tensor);
    slice_layer->setName(
        ("PrelnEmbeltwise_slice_layer (Output: slice_max_seqlen " +
         op_desc.Output("Out")[0] + ")")
            .c_str());
    engine_->SetTensorDynamicRange(slice_layer->getOutput(0), 1.0f);

    auto* reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *slice_layer->getOutput(0));
    nvinfer1::Dims shape_dim;
    shape_dim.nbDims = 1;
    shape_dim.d[0] = -1;
    reshape_layer->setReshapeDimensions(shape_dim);
    reshape_layer->setName(
        ("PrelnEmbeltwise_reshape_layer (Output: max_seqlen " +
         op_desc.Output("Out")[0] + ")")
            .c_str());
    engine_->SetTensorDynamicRange(reshape_layer->getOutput(0), 1.0f);
    engine_->SetITensor("max_seqlen_tensor", reshape_layer->getOutput(0));
    plugin_inputs.emplace_back(
        reshape_layer->getOutput(0));  // max_seqlen, eval_placeholder_3
=======
    std::vector<nvinfer1::ITensor*> plugin_inputs = input_ids;
    plugin_inputs.emplace_back(engine_->GetITensor(
        "max_seqlen_tensor"));  // max_seqlen, eval_placeholder_3
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    auto creator = GetPluginRegistry()->getPluginCreator(
        "ManyEmbLayerNormPluginDynamic", "2");

    auto plugin_obj =
        creator->createPlugin("ManyEmbLayerNormPluginDynamic", plugin_ptr);

    auto plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

    plugin_layer->setName(("ManyEmbLayerNormPluginDynamic_V3(Output: " +
                           op_desc.Output("Out")[0] + ")")
                              .c_str());
    free(plugin_ptr);
    float out_0_scale =
        PADDLE_GET_CONST(float, op_desc.GetAttr("out_0_threshold"));
    float out_1_scale =
        PADDLE_GET_CONST(float, op_desc.GetAttr("out_1_threshold"));
    engine_->SetTensorDynamicRange(plugin_layer->getOutput(0), out_0_scale);
    engine_->SetTensorDynamicRange(plugin_layer->getOutput(1), out_1_scale);

    auto* shuffler_embed_out0 =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(plugin_layer->getOutput(0)));
    nvinfer1::Permutation transpose_0{2, 1, 0, 3};
    shuffler_embed_out0->setSecondTranspose(transpose_0);
    shuffler_embed_out0->getOutput(0)->setName(
        op_desc.Output("Out_0")[0].c_str());
    engine_->SetITensor(op_desc.Output("Out_0")[0],
                        shuffler_embed_out0->getOutput(0));
    shuffler_embed_out0->setName(
        ("shuffler_after_ManyEmbLayerNormPluginDynamic_V3(Output_0: " +
         op_desc.Output("Out_0")[0] + ")")
            .c_str());

    auto* shuffler_embed_out1 =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(plugin_layer->getOutput(1)));
    nvinfer1::Permutation transpose_1{2, 1, 0, 3};
    shuffler_embed_out1->setSecondTranspose(transpose_1);
    shuffler_embed_out1->getOutput(0)->setName(
        op_desc.Output("Out_1")[0].c_str());

    engine_->SetITensor(op_desc.Output("Out_1")[0],
                        shuffler_embed_out1->getOutput(0));
    shuffler_embed_out1->setName(
        ("shuffler_after_ManyEmbLayerNormPluginDynamic_V3(Output_1: " +
         op_desc.Output("Out_1")[0] + ")")
            .c_str());

#else
    PADDLE_THROW(platform::errors::Fatal(
        "PreInErnie want to use oss, must be with interleaved, "
        "your TRT version is no less than 7.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_preln_embedding_eltwise_layernorm,
                          PrelnEmbEltwiseLayerNormOpConverter);
