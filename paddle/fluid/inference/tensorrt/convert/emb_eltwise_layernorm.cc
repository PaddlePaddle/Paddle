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
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class EmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(4) << "convert fluid swish op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    auto id_names = op_desc.Input("Ids");
    auto emb_names = op_desc.Input("Embs");

    PADDLE_ENFORCE_EQ(id_names.size(), emb_names.size(),
                      platform::errors::InvalidArgument(
                          "The id and emb size of fused EmbEltwiseLayerNormOp "
                          "should be same "));
    int input_num = id_names.size();

    // Declare inputs
    std::vector<nvinfer1::ITensor*> input_ids;
    for (int i = 0; i < input_num; i++) {
      input_ids.push_back(engine_->GetITensor(id_names[i]));
    }

    std::vector<float*> input_embs;
    std::vector<int> emb_sizes;

    // get the presistable var's data
    auto get_persistable_data = [&](const std::string& var_name,
                                    framework::DDim* dims) -> float* {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor, false);
      return temp_data;
    };

    int hidden = 0;
    for (int i = 0; i < input_num; i++) {
      framework::DDim emb_dims;
      float* emb_data = get_persistable_data(emb_names[i], &emb_dims);
      int64_t emb_size = framework::product(emb_dims);
      input_embs.push_back(emb_data);
      emb_sizes.push_back(emb_size);
      PADDLE_ENFORCE_EQ(
          emb_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The fused EmbEltwiseLayerNorm's emb should be 2 dims."));
      hidden = emb_dims[1];
    }

    framework::DDim bias_dims, scale_dims;

    auto* bias =
        get_persistable_data(op_desc.Input("Bias").front(), &bias_dims);
    auto* scale =
        get_persistable_data(op_desc.Input("Scale").front(), &scale_dims);
    int64_t bias_size = framework::product(bias_dims);
    int64_t scale_size = framework::product(scale_dims);
    float eps = BOOST_GET_CONST(float, op_desc.GetAttr("epsilon"));
    nvinfer1::ILayer* layer = nullptr;

    if (engine_->with_dynamic_shape()) {
      auto use_fp16 = engine_->WithFp16();
      plugin::DynamicPluginTensorRT* plugin = nullptr;
      if (use_fp16) {
#ifdef SUPPORTS_CUDA_FP16
        plugin = new plugin::EmbEltwiseLayernormPluginDynamic<half>(
            input_embs, bias, scale, emb_sizes, bias_size, scale_size, hidden,
            eps);
#else
        plugin = new plugin::EmbEltwiseLayernormPluginDynamic<float>(
            input_embs, bias, scale, emb_sizes, bias_size, scale_size, hidden,
            eps);
#endif
      } else {
        plugin = new plugin::EmbEltwiseLayernormPluginDynamic<float>(
            input_embs, bias, scale, emb_sizes, bias_size, scale_size, hidden,
            eps);
      }
      layer = engine_->AddPluginV2(input_ids.data(), input_num, plugin);
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static"
          "shape mode, which is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface"
          " to set the shape information to run the dynamic shape mode."));
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "emb_eltwise_layernorm", {output_name},
                             test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_embedding_eltwise_layernorm,
                          EmbEltwiseLayerNormOpConverter);
