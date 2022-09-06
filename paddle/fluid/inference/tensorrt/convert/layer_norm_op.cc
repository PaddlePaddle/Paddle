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
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a fluid layer_norm op to tensorrt layer_norm plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    PADDLE_ENFORCE_NOT_NULL(
        Bias_v,
        platform::errors::InvalidArgument(
            "Input(Bias) of layer_norm should not be null."));
    PADDLE_ENFORCE_NOT_NULL(
        Scale_v,
        platform::errors::InvalidArgument(
            "Input(Scale) of layer_norm should not be null."));

    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();

    auto bias_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
    auto scale_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);

    nvinfer1::ILayer* layernorm_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      // For dynamic shape,
      // the shape of mean and variance will be determine in configuPlugin.
      std::vector<int64_t> mean_shape{1};
      std::vector<int64_t> variance_shape{1};
      plugin::LayerNormPluginDynamic* plugin =
          new plugin::LayerNormPluginDynamic(
              static_cast<const float*>(bias_weight.get().values),
              bias_weight.get().count,
              static_cast<const float*>(scale_weight.get().values),
              scale_weight.get().count,
              begin_norm_axis,
              eps,
              mean_shape,
              variance_shape);
      layernorm_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
      int statis_num = 1;
      for (int i = 1; i < begin_norm_axis; i++) {
        statis_num *= X->getDimensions().d[i];
      }
      std::vector<int64_t> mean_shape{statis_num};
      std::vector<int64_t> variance_shape{statis_num};
      plugin::LayerNormPlugin* plugin = new plugin::LayerNormPlugin(
          static_cast<const float*>(bias_weight.get().values),
          bias_weight.get().count,
          static_cast<const float*>(scale_weight.get().values),
          scale_weight.get().count,
          begin_norm_axis,
          eps,
          mean_shape,
          variance_shape);
      layernorm_layer = engine_->AddPlugin(
          &X, 1, reinterpret_cast<plugin::PluginTensorRT*>(plugin));
    }

    auto output_name = op_desc.Output("Y").front();
    RreplenishLayerAndOutput(
        layernorm_layer, "layer_norm", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
