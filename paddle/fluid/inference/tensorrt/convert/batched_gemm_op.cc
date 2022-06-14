/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/plugin/batched_gemm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class BatchedGemmOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a BatchedGemm op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    // input is combined
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    auto t = input->getType();
    if(t == nvinfer1::DataType::kFLOAT)
      std::cout<< "***converter::float***" << std::endl;
    if(t == nvinfer1::DataType::kHALF)
      std::cout<< "***converter::half***" << std::endl;
    // combined fc weights and combined fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

    float* weight_data = nullptr;

    //weight_data = engine_->GetWeightCPUData(weight_name, weight_t, false); //false for not int8 
    weight_data = engine_->GetWeightCPUData(weight_name, weight_t); //false for not int8 

    float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t); 
    //float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t, false); 
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(), weight_data,
           weight_t->numel() * sizeof(float));

    // (batchcount, k, n) k:#rows n:#colums
    auto weight_dims = weight_t->dims();

    int batchcount = weight_dims[0];   
    int k = weight_dims[1];       // rows
    int n = weight_dims[2];  // columns


    
    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::IShuffleLayer* expand_layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];

    auto x_dim = input->getDimensions();
    //v1
    /*
    if (x_dim.nbDims == 2) {
      nvinfer1::Dims expand_shape;
      expand_shape.nbDims = 1 + x_dim.nbDims;//bc, m, k
      expand_shape.d[0] = batchcount;
      expand_shape.d[1] = x_dim.d[0];// < 0 ? 0 : x_dim.d[0];
      int dim3 = int(x_dim.d[1] / batchcount);
      expand_shape.d[2] = dim3;
      
      expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      expand_layer->setReshapeDimensions(expand_shape);
      input = expand_layer->getOutput(0);
      expand_layer->getOutput(0)->setName(
          ("reshape_before_batchedgemm_out: " + output_name).c_str());
      expand_layer->setName(
          ("batchedgemm_Shuffle: (Output: " + output_name + ")").c_str());
    }*/
    if (x_dim.nbDims == 2) {
      nvinfer1::Dims expand_shape;
      expand_shape.nbDims = 1 + x_dim.nbDims;//bc, m, k
      expand_shape.d[0] = x_dim.d[0];
      expand_shape.d[1] = batchcount;// < 0 ? 0 : x_dim.d[0];
      int dim3 = int(x_dim.d[1] / batchcount);
      expand_shape.d[2] = dim3;
      
      std::vector<int> axis = {1, 0, 2};
      nvinfer1::Permutation perm;
      for (int i = 0; i < 3; i++) {
        perm.order[i] = axis[i];
      }

      expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      expand_layer->setReshapeDimensions(expand_shape);
      expand_layer->setSecondTranspose(perm);
      input = expand_layer->getOutput(0);
      expand_layer->getOutput(0)->setName(
          ("reshape_before_batchedgemm_out: " + output_name).c_str());
      expand_layer->setName(
          ("batchedgemm_Shuffle: (Output: " + output_name + ")").c_str());
    }


    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? BOOST_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";
    //only support relu now
    bool has_relu = false;
    if(activation_type.compare("relu") == 0)
        has_relu = true;

    if (engine_->with_dynamic_shape()) {
      //no oss inplement
      /*
      PADDLE_ENFORCE_EQ(
          input->getDimensions().nbDims, 2,
          platform::errors::InvalidArgument(
              "The Input dim of the BatchedGemm should be 4, "
              "but it's (%d) now.",
              input->getDimensions().nbDims));
      */
      //TensorRTEngine::Weight weight;
      nvinfer1::Weights weight;
      weight.count = weight_t->numel();
      //{nvinfer1::DataType::kFLOAT,
      //                              static_cast<void*>(weight_data),
      //                              static_cast<size_t>(weight_t->numel())};
      //weight.dims.assign({batchcount, k, n});
      nvinfer1::Weights bias;
      //TensorRTEngine::Weight bias;//{nvinfer1::DataType::kFLOAT,
                                 // static_cast<void*>(bias_data),
                                //static_cast<size_t>(bias_t->numel())};
      bias.count = bias_t->numel();
      
      std::vector<nvinfer1::ITensor*> plugin_inputs;
      plugin_inputs.push_back(input);
      //reserve for laster support
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

      if (engine_->precision() == AnalysisConfig::Precision::kHalf) {
        with_fp16 = true;
      }
      if (with_fp16) {
        auto half_weight_data = new half[weight_t->numel()];
        for (int i = 0; i < weight_t->numel(); i++) {
          half_weight_data[i] = static_cast<half>(weight_data[i]);
        }
        weight.type = nvinfer1::DataType::kHALF;
        weight.values = half_weight_data;
      } else {
        weight.type = nvinfer1::DataType::kFLOAT;
        weight.values = weight_data;
      }
      //weight.dims.assign({batchcount, k, n});

      if (with_fp16) {
        auto half_bias_data = new half[bias_t->numel()];
        for (int i = 0; i < bias_t->numel(); i++) {
          half_bias_data[i] = static_cast<half>(bias_data[i]);
        }
        bias.type = nvinfer1::DataType::kHALF;
        bias.values = half_bias_data;
      } else {
        bias.type = nvinfer1::DataType::kFLOAT;
        bias.values = bias_data;
      }

      plugin::DynamicPluginTensorRT* plugin =
          new plugin::CuBLASBatchedGemmPlugin("batchedgemmplugin", weight, bias, k, n, batchcount, has_relu, with_fp16);
      layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 1, plugin);
      RreplenishLayerAndOutput(layer, "batchedgemm", {output_name},
                             test_mode);
  
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(batchedgemm, BatchedGemmOpConverter);
