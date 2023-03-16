/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class BevCrossMultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3)
        << "convert a bev_cross_multihead_mamul op to a corresponding tensorrt "
           "network structure";
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
      with_fp16 = true;
    }
    PADDLE_ENFORCE_EQ(
        with_fp16,
        true,
        platform::errors::Unimplemented(
            "Trt cross attention oss plugin only support fp16 mode yet."));

    framework::OpDesc op_desc(op, nullptr);
    auto* input_q = engine_->GetITensor(op_desc.Input("InputQ").front());
    auto* input_k = engine_->GetITensor(op_desc.Input("InputK").front());
    auto* input_v = engine_->GetITensor(op_desc.Input("InputV").front());
#define DEBUG 1
#ifdef DEBUG
    auto in_dim = input_q->getDimensions();
    std::cout << "input_q=";
    for (int i = 0; i < in_dim.nbDims; i++) {
      std::cout << in_dim.d[i] << ":";
    }
    std::cout << std::endl;

    in_dim = input_k->getDimensions();
    std::cout << "input_k=";
    for (int i = 0; i < in_dim.nbDims; i++) {
      std::cout << in_dim.d[i] << ":";
    }
    std::cout << std::endl;
#endif
    // getDimensions();
    nvinfer1::ITensor* input_q_shape_tensor = Shape(input_q);
    auto output_name = op_desc.Output("Out")[0];
    std::cout << "#######=" << output_name << std::endl;
    // add  concat layer, refer to concat_op.cc
    std::vector<nvinfer1::ITensor*> itensors;
    itensors.push_back(input_k);
    itensors.push_back(input_v);
    auto* kv_layer_before = TRT_ENGINE_ADD_LAYER(
        engine_, Concatenation, itensors.data(), itensors.size());
    kv_layer_before->setAxis(2);
    std::cout << "#######1" << std::endl;
    // add shuffle
    auto* kv_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *kv_layer_before->getOutput(0));
    std::vector<nvinfer1::ITensor*> reshape;
    reshape.push_back(Add1DConstantLayer(1));     //  (batch_tensor);
    reshape.push_back(Add1DConstantLayer(6000));  //  (length_tensor);
    reshape.push_back(Add1DConstantLayer(8));
    reshape.push_back(Add1DConstantLayer(2));
    reshape.push_back(Add1DConstantLayer(32));
    kv_layer->setInput(1, *Concat(reshape));
    kv_layer->setName(
        ("bev_multihead_matmul(Output: " + output_name + ")").c_str());

    // add cross_attention_plugin
    auto creator = GetPluginRegistry()->getPluginCreator("fMHCA", "1");
    assert(creator != nullptr);
    std::vector<nvinfer1::PluginField> fields{};
    nvinfer1::PluginFieldCollection* plugin_collection =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(nvinfer1::PluginFieldCollection) +
                   fields.size() *
                       sizeof(nvinfer1::PluginField)));  // remember to free

    plugin_collection->nbFields = static_cast<int>(fields.size());
    plugin_collection->fields = fields.data();
    auto plugin = creator->createPlugin("fMHA_V2", plugin_collection);
    free(plugin_collection);
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(input_q);
    plugin_inputs.emplace_back(kv_layer->getOutput(0));
    auto plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin);

    // add shuffle
    auto* reshape_after_mha_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *plugin_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> reshape_tensor;
    reshape_tensor.push_back(Add1DConstantLayer(1));    //  (batch_tensor);
    reshape_tensor.push_back(Add1DConstantLayer(900));  //  (length_tensor);
    reshape_tensor.push_back(Add1DConstantLayer(256));
    reshape_after_mha_layer->setInput(1, *Concat(reshape_tensor));
    reshape_after_mha_layer->setName(
        ("shuffle_bev_multihead_matmul(Output: " + output_name + ")").c_str());
    auto layer = reshape_after_mha_layer;
    // return
    RreplenishLayerAndOutput(
        layer, "bev_cross_multihead_matmul", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(bev_cross_multihead_matmul,
                          BevCrossMultiheadMatMulOpConverter);
