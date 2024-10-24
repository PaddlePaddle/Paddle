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
#include "paddle/fluid/inference/tensorrt/plugin/trans_layernorm_op_plugin.h"
namespace paddle::inference::tensorrt {

class TransLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a trans_layer_norm fused op to tensorrt "
               "trans_layernorm plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    // we already check the begin_norm_axis in pass action.
    // here we set begin_norm_axis as 3 to fit the calculation in trt plugin.
    const int begin_norm_axis = 3;
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    PADDLE_ENFORCE_NOT_NULL(
        Bias_v,
        common::errors::InvalidArgument(
            "Input(Bias) of layer_norm should not be null."));
    PADDLE_ENFORCE_NOT_NULL(
        Scale_v,
        common::errors::InvalidArgument(
            "Input(Scale) of layer_norm should not be null."));

    auto* Bias_t = Bias_v->GetMutable<phi::DenseTensor>();
    auto* Scale_t = Scale_v->GetMutable<phi::DenseTensor>();

    auto bias_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
    auto scale_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);

    nvinfer1::ILayer* layernorm_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      // For dynamic shape,
      // the shape of mean and variance will be determine in configurePlugin.
      std::vector<int64_t> mean_shape{1};
      std::vector<int64_t> variance_shape{1};
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::TransLayerNormPluginDynamic* plugin =
          new plugin::TransLayerNormPluginDynamic(
              static_cast<const float*>(bias_weight.get().values),
              bias_weight.get().count,
              static_cast<const float*>(scale_weight.get().values),
              scale_weight.get().count,
              begin_norm_axis,
              eps,
              mean_shape,
              variance_shape,
              with_fp16);
      layernorm_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "trans_layernorm do not support static shape mode yet"));
    }

    auto output_layernorm_name = op_desc.Output("Out_layernorm").front();
    auto output_reshape_name = op_desc.Output("Out_reshape").front();
    ReplenishLayerAndOutput(layernorm_layer,
                            "trans_layernorm",
                            {output_layernorm_name, output_reshape_name},
                            test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(trans_layernorm, TransLayerNormOpConverter);
