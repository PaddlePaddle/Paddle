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
#include "paddle/fluid/inference/tensorrt/plugin/multihead_matmul_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* Input = engine_->GetITensor(op_desc.Input("Input").front());
    auto* W = scope.FindVar(op_desc.Input("W").front());
    auto* Bias = scope.FindVar(op_desc.Input("Bias").front());
    auto* BiasQK = engine_->GetITensor(op_desc.Input("BiasQK").front());
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Input").size(), 1,
        platform::errors::InvalidArgument(
            "size of input Input of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(
        op_desc.Input("BiasQK").size(), 1,
        platform::errors::InvalidArgument(
            "size of input BiasQK of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1,
                      platform::errors::InvalidArgument(
                          "size of output of multihead_matmul should be 1"));
    PADDLE_ENFORCE_NOT_NULL(
        Bias, platform::errors::InvalidArgument(
                  "param Bias of multihead_matmul should not be null"));
    PADDLE_ENFORCE_EQ(
        BiasQK->getDimensions().nbDims, 3,
        platform::errors::InvalidArgument(
            "dims size of input BiasQK of multihead_matmul should be 3"));
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("alpha"), true,
        platform::errors::PreconditionNotMet(
            "attribute alpha of multihead_matmul should not be empty"));
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("head_number"), true,
        platform::errors::PreconditionNotMet(
            "attribute head_number of multihead_matmul should not be empty"));

    // Declare attributes
    const bool transpose_q =
        op_desc.HasAttr("transpose_Q")
            ? boost::get<bool>(op_desc.GetAttr("transpose_Q"))
            : false;
    const bool transpose_k =
        op_desc.HasAttr("transpose_K")
            ? boost::get<bool>(op_desc.GetAttr("transpose_K"))
            : true;
    const bool transpose_v =
        op_desc.HasAttr("transpose_V")
            ? boost::get<bool>(op_desc.GetAttr("transpose_V"))
            : false;
    const float alpha = boost::get<float>(op_desc.GetAttr("alpha"));
    const int head_number = boost::get<int>(op_desc.GetAttr("head_number"));

    nvinfer1::Dims input_shape = Input->getDimensions();
    int seq_len = input_shape.d[0];
    int size_per_head = input_shape.d[1] / head_number;
    std::string alpha_name = op_desc.Output("Out")[0] + "_alpha";
    framework::DDim alpha_dim = framework::make_ddim({1});
    std::unique_ptr<framework::LoDTensor> alpha_t(new framework::LoDTensor());
    alpha_t->Resize(alpha_dim);
    float* alpha_data = alpha_t->mutable_data<float>(platform::CPUPlace());
    alpha_data[0] = alpha;

    TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(alpha_data), 1};
    TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto* w_t = W->GetMutable<framework::LoDTensor>();
    float* w_cpu_data =
        engine_->GetWeightCPUData(op_desc.Input("W").front(), w_t, false);
    std::unique_ptr<framework::LoDTensor> w_tensor(new framework::LoDTensor());
    w_tensor->Resize(w_t->dims());
    platform::CPUPlace cpu_place;
    TensorCopySync((*w_t), cpu_place, w_tensor.get());
    TensorRTEngine::Weight w_w{nvinfer1::DataType::kFLOAT,
                               static_cast<void*>(w_cpu_data),
                               w_tensor->memory_size() / sizeof(float)};

    auto* bias_t = Bias->GetMutable<framework::LoDTensor>();
    float* bias_cpu_data =
        engine_->GetWeightCPUData(op_desc.Input("Bias").front(), bias_t, false);
    std::unique_ptr<framework::LoDTensor> bias_tensor(
        new framework::LoDTensor());
    bias_tensor->Resize(bias_t->dims());
    TensorCopySync((*bias_t), cpu_place, bias_tensor.get());
    TensorRTEngine::Weight bias_w{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(bias_cpu_data),
                                  bias_tensor->memory_size() / sizeof(float)};

    auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *Input,
                                          3 * head_number * size_per_head,
                                          w_w.get(), bias_w.get());

    plugin::MultiheadMatmulPlugin* plugin = new plugin::MultiheadMatmulPlugin(
        transpose_q, transpose_k, transpose_v, alpha, head_number, seq_len,
        size_per_head);
    plugin->AddInput(fc_layer->getOutput(0));
    plugin->AddInput(BiasQK);
    nvinfer1::IPluginLayer* plugin_layer = engine_->AddPlugin(
        const_cast<nvinfer1::ITensor* const*>(plugin->GetInputs().data()), 2,
        reinterpret_cast<plugin::PluginTensorRT*>(plugin));

    auto output_name = op_desc.Output("Out").front();
    engine_->SetWeights(alpha_name, std::move(alpha_t));
    engine_->SetWeights(op_desc.Input("W").front(), std::move(w_tensor));
    engine_->SetWeights(op_desc.Input("Bias").front(), std::move(bias_tensor));
    RreplenishLayerAndOutput(plugin_layer, "multihead_matmul", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
