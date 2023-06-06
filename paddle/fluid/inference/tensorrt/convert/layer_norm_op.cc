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
    VLOG(4) << "convert a layer_norm op with dynamic shape to  Normalization "
               "layer or  Static shape  tensorrt layer_norm plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X")[0]);
    auto rank = X->getDimensions().nbDims;
    std::string output_name = op_desc.Output("Y")[0];
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    if (engine_->with_dynamic_shape()) {
      auto* Scale = engine_->GetITensor(op_desc.Input("Scale")[0]);
      auto* Bias = engine_->GetITensor(op_desc.Input("Bias")[0]);
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
#if IS_TRT_VERSION_GE(8600)
      auto layer = TRT_ENGINE_ADD_LAYER(
          engine_, Normalization, *X, *Scale_reshape, *Bias_reshape, axisMask);
      layer->setEpsilon(eps);
      RreplenishLayerAndOutput(layer, "layer_norm", {output_name}, test_mode);
#else
      // μ
      auto miu_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Reduce, *X, nvinfer1::ReduceOperation::kAVG, axisMask, true);
      miu_layer->setName((output_name + "_miu").c_str());
      auto miu_output = mean_expected_layer->getOutput(0);
      // x−μ
      xsubmiu_output = Sub(*X, *miu_output);
      // σ
      // pow(x−μ,2)
      auto pow_tensor = Add1DConstantLayer(2);
      auto xsubmiu_pow_out = Pow(*xsubmiu_output, *pow_tensor);
      // mean_var
      auto mean_var_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               Reduce,
                               *xsubmiu_pow_out,
                               nvinfer1::ReduceOperation::kAVG,
                               axisMask,
                               true);
      mean_var_layer->setName((output_name + "_sigma").c_str());
      auto mean_var_out = sigma_layer->getOutput(0);
      // sigma
      auto eps_tensor = Add1DConstantLayer(eps);
      auto sum_out = Sum(mean_var_out, eps_tensor);
      auto sigma_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Unary, *sum_out, nvinfer1::UnaryOperation::kSQRT);
      auto sigma_output = sigma_layer->getOutput(0);
      // σ/sigma
      auto div_out = Div(xsubmiu_output, sigma_output);
      // (σ/sigma)*g+b
      auto scale_out =
          Prod(div_out,
               BroadcastTensors(
                   X,
                   Scale_reshape,
                   ("layer_norm Scale: reshape_for_broadcast: (Output(" +
                    output_name + ")")
                       .c_str()));

      auto layer = TRT_ENGINE_ADD_LAYER(
          engine_,
          ElementWise,
          *scale_out,
          *BroadcastTensors(
              X,
              Bias_reshape,
              ("layer_norm Bias: reshape_for_broadcast: (Output(" +
               output_name + ")")
                  .c_str()),
          nvinfer1::ElementWiseOperation::kSUM);
      RreplenishLayerAndOutput(layer, "layer_norm", {output_name}, test_mode);
#endif
    } else {
      auto* Bias_v = scope.FindVar(op_desc.Input("Bias")[0]);
      auto* Scale_v = scope.FindVar(op_desc.Input("Scale")[0]);
      PADDLE_ENFORCE_NOT_NULL(
          Bias_v,
          platform::errors::InvalidArgument(
              "Input(Bias) of layer_norm should not be null."));
      PADDLE_ENFORCE_NOT_NULL(
          Scale_v,
          platform::errors::InvalidArgument(
              "Input(Scale) of layer_norm should not be null."));
      auto* Bias_t = Bias_v->GetMutable<phi::DenseTensor>();
      auto* Scale_t = Scale_v->GetMutable<phi::DenseTensor>();

      auto bias_weight =
          engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
      auto scale_weight =
          engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);

      const int begin_norm_axis =
          op_desc.HasAttr("begin_norm_axis")
              ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
              : 1;

      int statis_num = 1;
      for (int i = 1; i < begin_norm_axis; i++) {
        statis_num *= X->getDimensions().d[i];
      }
      std::vector<int64_t> mean_shape{statis_num};
      std::vector<int64_t> variance_shape{statis_num};
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::LayerNormPlugin* plugin = new plugin::LayerNormPlugin(
          static_cast<const float*>(bias_weight.get().values),
          bias_weight.get().count,
          static_cast<const float*>(scale_weight.get().values),
          scale_weight.get().count,
          begin_norm_axis,
          eps,
          mean_shape,
          variance_shape,
          with_fp16);
      auto* layernorm_layer = engine_->AddPlugin(
          &X, 1, reinterpret_cast<plugin::PluginTensorRT*>(plugin));
      RreplenishLayerAndOutput(
          layernorm_layer, "layer_norm", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
