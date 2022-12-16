/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class MultiheadMatMulMemEffOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
        "network structure";
    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    // auto input_dims = input->getDimensions();
    auto weight_name = op_desc.Input("W").front();
    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<phi::DenseTensor>();
    float* weight_data = nullptr;
    weight_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(weight_name, *weight_t).get().values));
        // (hidden_in, 3, hidden_out)
    const auto& weight_dims = weight_t->dims();

    int hidden_in = weight_dims[0];   // channels_in
    int three = weight_dims[1];       // three
    int hidden_out = weight_dims[2];  // channels_out
    int head_number = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));
    int head_size = hidden_out / head_number;

    // VLOG(0)<<"@@@ hidden_in:"<<hidden_in
    //        << "hidden_out:"<<hidden_out
    //        <<" head_number:"<<head_number
    //        <<" head_size:"<<head_size;

    // float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));
    // int m = hidden_in;
    int n = three * hidden_out;
    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];
    // [hidden_in, 3, head_number, head_size] 
    // -> [head_number, 3, head_size, hidden_in]

    // VLOG(0)<<"@@ before transpose_weight";
    
    auto transpose_weight = [](const float* src,
                               float* dst,
                               int three,
                               int head_number,
                               int head_size,
                               int hidden_in) {
            for(int hn =0;hn<head_number;hn++){
              for(int t=0;t<three;t++){
                for(int hs=0;hs<head_size;hs++){
                  for (int hi=0;hi<hidden_in;hi++){
                    int out_index=hn*three*head_size*hidden_in+
                                  t*head_size*hidden_in+
                                  hs*hidden_in+
                                  hi;
                    // int out_index = hi*three*head_number*head_size+
                    //                t*head_number*head_size+
                    //                hn*head_size+
                    //                hs;
                    int in_index = hi*three*head_number*head_size+
                                   t*head_number*head_size+
                                   hn*head_size+
                                   hs;
                    dst[out_index]=src[in_index];
                  }
                }
              }
            }
          };
    std::vector<float> weight_data_tmp;
    // VLOG(0)<<"@@ pin 2";

    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(),
           weight_data,
           weight_t->numel() * sizeof(float));
    transpose_weight(weight_data_tmp.data(),
                    weight_data,
                    three,
                    head_number,
                    head_size,
                    hidden_in);
    // printf("print qkv fc weight : \n ");
    // for(int i =0;i<3*320*320;i++){
    //   printf("%f ",weight_data[i]);
    //   if(i%320==319){
    //     printf("\n");
    //   }
    // }
    // printf("\n");
    nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                             static_cast<void*>(weight_data),
                             static_cast<int32_t>(weight_t->numel())};
    nvinfer1::Weights bias{};
    // add shuffle for FullyConnected layer
    // VLOG(0)<<"@@ pin 3";

    std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    for (int i = 0; i < 5; i++) {
      reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
    }
    for (int i = 0; i < 3; i++) {
      reshape_before_fc_shape_tensor[i] =
          GetEleTensorOfShape(input_shape_tensor, i);
    }
    // VLOG(0)<<"@@ pin 3.5";
    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    reshape_before_fc_layer->setInput(
        1, *Concat(reshape_before_fc_shape_tensor));
    reshape_before_fc_layer->setName(
        ("shuffle_before_fc_multihead_matmul(Output: " + output_name +
          ")")
            .c_str());
    // VLOG(0)<<"@@ pin 4";

    nvinfer1::ILayer* fc_layer = nullptr;
    fc_layer = TRT_ENGINE_ADD_LAYER(
                engine_, FullyConnected, *reshape_before_fc_layer->getOutput(0),
                n,
                weight, 
                bias);
    fc_layer->setName(
            ("multihead_mamul_fc(Output: " + output_name + ")").c_str());

    // add shuffle for fc layer
    auto* reshape_after_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *fc_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> mha_input_tensor_shape;
    for (int i = 0; i < 5; i++) {
      mha_input_tensor_shape.push_back(Add1DConstantLayer(1));
    }
    mha_input_tensor_shape[0] =
          GetEleTensorOfShape(input_shape_tensor, 0);
    mha_input_tensor_shape[1] =
          GetEleTensorOfShape(input_shape_tensor, 1);
    mha_input_tensor_shape[2] = Add1DConstantLayer(head_number);
    mha_input_tensor_shape[3] = Add1DConstantLayer(3);
    mha_input_tensor_shape[4] = Add1DConstantLayer(head_size);
    reshape_after_fc_layer->setInput(1, *Concat(mha_input_tensor_shape));
    reshape_after_fc_layer->setName(
        ("shuffle_after_fc_multihead_matmul(Output: " + output_name + ")")
            .c_str());

    auto creator = GetPluginRegistry()->getPluginCreator(
      "fMHA_V2", "1");
    assert(creator != nullptr);
    std::vector<nvinfer1::PluginField> fields{};
    nvinfer1::PluginFieldCollection* plugin_collection =
      static_cast<nvinfer1::PluginFieldCollection*>(malloc(
        sizeof(*plugin_collection) +
          fields.size() *
          sizeof(nvinfer1::PluginField)));  // remember to free

    plugin_collection->nbFields = static_cast<int>(fields.size());
    plugin_collection->fields = fields.data();
    auto plugin = creator->createPlugin("fMHA_V2",
                                        plugin_collection);
    free(plugin_collection);
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(reshape_after_fc_layer->getOutput(0));
    auto plugin_layer = engine_->network()->addPluginV2(
    plugin_inputs.data(), plugin_inputs.size(), *plugin);
    
    // add shuffle
    nvinfer1::ITensor* batch_tensor =
      GetEleTensorOfShape(input_shape_tensor, 0);
    nvinfer1::ITensor* length_tensor =
      GetEleTensorOfShape(input_shape_tensor, 1);
    auto* reshape_after_mha_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *plugin_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> reshape_tensor;
    reshape_tensor.push_back(batch_tensor);
    reshape_tensor.push_back(length_tensor);
    reshape_tensor.push_back(Add1DConstantLayer(-1));
    reshape_after_mha_layer->setInput(1, *Concat(reshape_tensor));
    reshape_after_mha_layer->setName(
        ("shuffle_last_multihead_matmul(Output: " + output_name + ")")
            .c_str());
    // return
    layer = reshape_after_mha_layer;
    RreplenishLayerAndOutput(
        layer, "multihead_matmul_memeff", {output_name}, test_mode);
  }
};

}
}
}

REGISTER_TRT_OP_CONVERTER(multihead_matmul_v4, MultiheadMatMulMemEffOpConverter);
