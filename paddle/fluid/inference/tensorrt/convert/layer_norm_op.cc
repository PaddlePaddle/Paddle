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

namespace paddle::inference::tensorrt {

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a layer_norm op  to  INormalization layer or  "
               "layer_norm plugin";
    framework::OpDesc op_desc(op, nullptr);
    auto* X = engine_->GetITensor(op_desc.Input("X")[0]);
    std::string output_name = op_desc.Output("Y")[0];
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
#if IS_TRT_VERSION_GE(8600)
    auto* Scale = engine_->GetITensor(op_desc.Input("Scale")[0]);
    auto* Bias = engine_->GetITensor(op_desc.Input("Bias")[0]);
    auto rank = X->getDimensions().nbDims;
    int32_t begin_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
    uint32_t axisMask{0};
    for (int32_t i = begin_axis; i < rank; i++) {
      axisMask |= 1 << i;
    }
    std::vector<int32_t> indice_dim_vec(rank);
    std::iota(indice_dim_vec.begin(), indice_dim_vec.end(), 0);
    auto p = std::remove_if(indice_dim_vec.begin(),
                            indice_dim_vec.end(),
                            [begin_axis](int x) { return x < begin_axis; });
    indice_dim_vec.resize(p - indice_dim_vec.begin());
    auto newDims = Gather(Shape(X), indice_dim_vec);
    auto newrank = indice_dim_vec.size();
    auto* one_rank_tensor =
        Add1DConstantLayer(std::vector<int32_t>(rank - newrank, 1));
    std::vector<nvinfer1::ITensor*> itensors;
    itensors.push_back(one_rank_tensor);
    itensors.push_back(newDims);
    nvinfer1::ITensor* concat_shape_tensor = Concat(itensors);
    auto Bias_reshape = Reshape(
        Bias,
        concat_shape_tensor,
        ("layer_norm Bias: reshape: (Output(" + output_name + ")").c_str());
    auto Scale_reshape = Reshape(
        Scale,
        concat_shape_tensor,
        ("layer_norm Scale: reshape: (Output(" + output_name + ")").c_str());
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Normalization, *X, *Scale_reshape, *Bias_reshape, axisMask);
    SupportFP32MixPrecision(output_name, op_desc.Type(), layer);
    layer->setEpsilon(eps);
    ReplenishLayerAndOutput(layer, "layer_norm", {output_name}, test_mode);
#endif
#if IS_TRT_VERSION_LT(8600)
    // For dynamic shape & trt<8.6,
    // the shape of mean and variance will be determine in configurePlugin.
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
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
    std::vector<int64_t> mean_shape{1};
    std::vector<int64_t> variance_shape{1};
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    plugin::LayerNormPluginDynamic* plugin = new plugin::LayerNormPluginDynamic(
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
    ReplenishLayerAndOutput(
        layernorm_layer, "layer_norm", {output_name}, test_mode);
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
