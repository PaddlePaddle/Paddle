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
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid layer_norm op to tensorrt layer_norm plugin";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Bias").size(), 1);   // Bias is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Scale").size(), 1);  // Scale is a weight

    PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    // Declare weights
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    auto* Mean_v = scope.FindVar(op_desc.Output("Mean").front());
    auto* Variance_v = scope.FindVar(op_desc.Output("Variance").front());
    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? boost::get<int>(op_desc.GetAttr("begin_norm_axis"))
            : 1;
    const float eps = op_desc.HasAttr("epsilon")
                          ? boost::get<float>(op_desc.GetAttr("epsilon"))
                          : 1e-5f;

    nvinfer1::Dims input_shape = X->getDimensions();
    int input_dims = input_shape.nbDims;

    PADDLE_ENFORCE_NOT_NULL(Bias_v,
                            "Input(Bias) of layer_norm should not be null.");
    PADDLE_ENFORCE_NOT_NULL(Scale_v,
                            "Input(Scale) of layer_norm should not be null.");

    // get tensor
    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();
    auto* Mean_t = Mean_v->GetMutable<framework::LoDTensor>();
    auto* Variance_t = Variance_v->GetMutable<framework::LoDTensor>();

    std::vector<int64_t> mean_shape =
        framework::vectorize<int64_t>(Mean_t->dims());
    std::vector<int64_t> variance_shape =
        framework::vectorize<int64_t>(Variance_t->dims());

    // create temp tensor for weights
    std::unique_ptr<framework::LoDTensor> bias_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> scale_tensor(
        new framework::LoDTensor());

    bias_tensor->Resize(Bias_t->dims());
    scale_tensor->Resize(Scale_t->dims());

    platform::CPUPlace cpu_place;
    // copy data from gpu to cpu
    TensorCopySync((*Bias_t), cpu_place, &(*bias_tensor));
    TensorCopySync((*Scale_t), cpu_place, &(*scale_tensor));

    auto* bias_data = bias_tensor->mutable_data<float>(platform::CPUPlace());
    auto* scale_data = scale_tensor->mutable_data<float>(platform::CPUPlace());

    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    if (input_dims == 3) {
      d0 = input_shape.d[0];
      d1 = input_shape.d[1];
      d2 = input_shape.d[2];
      d3 = 1;
    }
    nvinfer1::Dims4 reshape_dim(d0, d1, d2, d3);

    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *const_cast<nvinfer1::ITensor*>(X));
    reshape_layer->setReshapeDimensions(reshape_dim);
    auto* input_after_reshape = reshape_layer->getOutput(0);

    plugin::LayerNormPlugin* plugin = new plugin::LayerNormPlugin(
        bias_data, bias_tensor->numel(), scale_data, scale_tensor->numel(),
        begin_norm_axis, eps, mean_shape, variance_shape);
    nvinfer1::IPluginLayer* layernorm_layer =
        engine_->AddPlugin(&input_after_reshape, 1, plugin);

    auto output_name = op_desc.Output("Y").front();
    engine_->SetWeights(op_desc.Input("Bias").front(), std::move(bias_tensor));
    engine_->SetWeights(op_desc.Input("Scale").front(),
                        std::move(scale_tensor));
    RreplenishLayerAndOutput(layernorm_layer, "layer_norm", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(layer_norm);
REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
