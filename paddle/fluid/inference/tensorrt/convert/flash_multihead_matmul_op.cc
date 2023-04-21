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

class FlashMultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a flash_multihead_mamul op to a corresponding tensorrt "
               "network structure";

    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
      with_fp16 = true;
    }
    PADDLE_ENFORCE_EQ(
        with_fp16,
        true,
        platform::errors::Unimplemented(
            "Trt flash attention oss plugin only support fp16 mode yet."));

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());

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

    int n = three * hidden_out;
    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];

    // [hidden_in, 3, head_number, head_size]
    // -> [head_number, 3, head_size, hidden_in]
    auto transpose_weight = [](const float* src,
                               float* dst,
                               int three,
                               int head_number,
                               int head_size,
                               int hidden_in) {
      for (int hn = 0; hn < head_number; hn++) {
        for (int t = 0; t < three; t++) {
          for (int hs = 0; hs < head_size; hs++) {
            for (int hi = 0; hi < hidden_in; hi++) {
              int out_index = hn * three * head_size * hidden_in +
                              t * head_size * hidden_in + hs * hidden_in + hi;
              int in_index = hi * three * head_number * head_size +
                             t * head_number * head_size + hn * head_size + hs;
              dst[out_index] = src[in_index];
            }
          }
        }
      }
    };
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(
        weight_data_tmp.data(), weight_data, weight_t->numel() * sizeof(float));

    transpose_weight(weight_data_tmp.data(),
                     weight_data,
                     three,
                     head_number,
                     head_size,
                     hidden_in);
    nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                             static_cast<void*>(weight_data),
                             static_cast<int32_t>(weight_t->numel())};
    // merge matmul+element
    nvinfer1::ILayer* merge_matmul_layer = nullptr;
    nvinfer1::ILayer* merge_element_layer = nullptr;
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    auto inputs = op_desc.Inputs();
    if (inputs.find("Bias") == inputs.end()) {
      nvinfer1::Weights bias{};
      std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;

      for (int i = 0; i < 5; i++) {
        reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
      }
      for (int i = 0; i < 3; i++) {
        reshape_before_fc_shape_tensor[i] =
            GetEleTensorOfShape(input_shape_tensor, i);
      }

      merge_matmul_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      merge_matmul_layer->setInput(1, *Concat(reshape_before_fc_shape_tensor));
      merge_matmul_layer->setName(
          ("shuffle_before_fc_multihead_matmul(Output: " + output_name + ")")
              .c_str());
      merge_element_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               FullyConnected,
                               *merge_matmul_layer->getOutput(0),
                               n,
                               weight,
                               bias);
      merge_element_layer->setName(
          ("multihead_mamul_fc(Output: " + output_name + ")").c_str());
    } else {
      // [length, 3, head_number, head_size]->[length, head_number, 3,
      // head_size]
      auto transpose_bias_v2 =
          [](const float* src, float* dst, int length, int N, int H) {
            for (int l = 0; l < length; l++) {
              for (int i = 0; i < 3; ++i) {
                for (int n = 0; n < N; ++n) {
                  for (int h = 0; h < H; ++h) {
                    dst[l * 3 * N * H + n * 3 * H + i * H + h] =
                        src[l * 3 * N * H + i * N * H + n * H + h];
                  }
                }
              }
            }
          };

      auto bias_name = op_desc.Input("Bias").front();
      auto* bias_v = scope.FindVar(bias_name);
      auto* bias_t = bias_v->GetMutable<phi::DenseTensor>();
      float* bias_data = const_cast<float*>(static_cast<const float*>(
          engine_->GetFp32TrtWeight(bias_name, *bias_t).get().values));

      const auto& bias_dims = bias_t->dims();
      int bias_length = bias_dims[0];

      std::vector<float> bias_data_tmp;
      bias_data_tmp.reserve(bias_t->numel());
      memcpy(bias_data_tmp.data(), bias_data, bias_t->numel() * sizeof(float));
      transpose_bias_v2(
          bias_data_tmp.data(), bias_data, bias_length, head_number, head_size);

      auto weight_shape = nvinfer1::Dims3{1, n, hidden_in};
      auto* weight_tensor = AddConstantLayer(weight_data, weight_shape, " ");
      auto bias_shape = nvinfer1::Dims3{1, bias_length, n};
      auto* bias_tensor = AddConstantLayer(bias_data, bias_shape, " ");

      // add MatrixMultiplyLayer layer

      nvinfer1::MatrixOperation matrix_operation_X =
          nvinfer1::MatrixOperation::kNONE;
      nvinfer1::MatrixOperation matrix_operation_Y =
          nvinfer1::MatrixOperation::kTRANSPOSE;
      merge_matmul_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                MatrixMultiply,
                                                *input,
                                                matrix_operation_X,
                                                *weight_tensor,
                                                matrix_operation_Y);
      merge_matmul_layer->setName(
          ("flash_attention_matrix_multiply(Output: " + output_name + ")")
              .c_str());

      // add ElementWiseLayer layer
      nvinfer1::ElementWiseOperation elementwise_operation =
          nvinfer1::ElementWiseOperation::kSUM;
      merge_element_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *merge_matmul_layer->getOutput(0),
                               *bias_tensor,
                               elementwise_operation);
      merge_element_layer->setName(
          ("flash_attention_elementwise(Output: " + output_name + ")").c_str());
    }
    // add shuffle

    auto* reshape_after_fc_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *merge_element_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> mha_input_tensor_shape;
    for (int i = 0; i < 5; i++) {
      mha_input_tensor_shape.push_back(Add1DConstantLayer(1));
    }
    mha_input_tensor_shape[0] = GetEleTensorOfShape(input_shape_tensor, 0);
    mha_input_tensor_shape[1] = GetEleTensorOfShape(input_shape_tensor, 1);
    mha_input_tensor_shape[2] = Add1DConstantLayer(head_number);
    mha_input_tensor_shape[3] = Add1DConstantLayer(3);
    mha_input_tensor_shape[4] = Add1DConstantLayer(head_size);
    reshape_after_fc_layer->setInput(1, *Concat(mha_input_tensor_shape));
    reshape_after_fc_layer->setName(
        ("shuffle_after_fc_multihead_matmul(Output: " + output_name + ")")
            .c_str());
    auto creator = GetPluginRegistry()->getPluginCreator("fMHA_V2", "1");
    assert(creator != nullptr);
    std::vector<nvinfer1::PluginField> fields{};
    nvinfer1::PluginFieldCollection* plugin_collection =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_collection) +
                   fields.size() *
                       sizeof(nvinfer1::PluginField)));  // remember to free

    plugin_collection->nbFields = static_cast<int>(fields.size());
    plugin_collection->fields = fields.data();
    auto plugin = creator->createPlugin("fMHA_V2", plugin_collection);
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
    auto* reshape_after_mha_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *plugin_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> reshape_tensor;
    reshape_tensor.push_back(batch_tensor);
    reshape_tensor.push_back(length_tensor);
    reshape_tensor.push_back(Add1DConstantLayer(-1));
    reshape_after_mha_layer->setInput(1, *Concat(reshape_tensor));
    reshape_after_mha_layer->setName(
        ("shuffle_last_multihead_matmul(Output: " + output_name + ")").c_str());
    // return
    layer = reshape_after_mha_layer;
    RreplenishLayerAndOutput(
        layer, "flash_multihead_matmul", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flash_multihead_matmul,
                          FlashMultiheadMatMulOpConverter);
