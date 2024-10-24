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

namespace paddle::inference::tensorrt {

class QkMultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a qk_multihead_matmul op to a corresponding tensorrt "
               "network structure";

    framework::OpDesc op_desc(op, nullptr);
    auto* input_qk = engine_->GetITensor(op_desc.Input("Input_qk").front());
    auto* input_v = engine_->GetITensor(op_desc.Input("Input_v").front());

    auto output_name = op_desc.Output("Out")[0];

    /* ------------------    weight_qk  -------------------------*/
    auto weight_qk_name = op_desc.Input("W_qk").front();
    auto* weight_qk_v = scope.FindVar(weight_qk_name);
    auto* weight_qk_t = weight_qk_v->GetMutable<phi::DenseTensor>();
    float* weight_qk_data = nullptr;
    weight_qk_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(weight_qk_name, *weight_qk_t).get().values));

    const auto& weight_qk_dims =
        weight_qk_t->dims();  // hidden_in_qk 2 hidden_out_qk
    int hidden_in_qk = weight_qk_dims[0];
    int num_qk = weight_qk_dims[1];
    int hidden_out_qk = weight_qk_dims[2];
    int head_number_qk = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));
    int head_size_qk = hidden_out_qk / head_number_qk;
    int n_qk = num_qk * hidden_out_qk;

    // [hidden_in, 2, head_number, head_size]
    // -> [head_number, 2, head_size, hidden_in]
    auto transpose_weight_qk = [](const float* src,
                                  float* dst,
                                  int two,
                                  int head_number,
                                  int head_size,
                                  int hidden_in) {
      for (int hn = 0; hn < head_number; hn++) {
        for (int t = 0; t < two; t++) {
          for (int hs = 0; hs < head_size; hs++) {
            for (int hi = 0; hi < hidden_in; hi++) {
              int out_index = hn * two * head_size * hidden_in +
                              t * head_size * hidden_in + hs * hidden_in + hi;
              int in_index = hi * two * head_number * head_size +
                             t * head_number * head_size + hn * head_size + hs;
              dst[out_index] = src[in_index];
            }
          }
        }
      }
    };

    std::vector<float> weight_qk_data_tmp;
    weight_qk_data_tmp.reserve(weight_qk_t->numel());
    memcpy(weight_qk_data_tmp.data(),
           weight_qk_data,
           weight_qk_t->numel() * sizeof(float));
    transpose_weight_qk(weight_qk_data_tmp.data(),
                        weight_qk_data,
                        num_qk,
                        head_number_qk,
                        head_size_qk,
                        hidden_in_qk);

    /* ------------------    bias_qk  -------------------------*/
    auto bias_qk_name = op_desc.Input("B_qk").front();
    auto* bias_qk_v = scope.FindVar(bias_qk_name);
    auto* bias_qk_t = bias_qk_v->GetMutable<phi::DenseTensor>();
    float* bias_qk_data = nullptr;
    bias_qk_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(bias_qk_name, *bias_qk_t).get().values));

    // [2, head_number, head_size] -> [head_number, 2, head_size]
    auto transpose_bias_qk = [](const float* src, float* dst, int N, int H) {
      for (int i = 0; i < 2; ++i) {
        for (int n = 0; n < N; ++n) {
          for (int h = 0; h < H; ++h) {
            dst[n * 2 * H + i * H + h] = src[i * N * H + n * H + h];
          }
        }
      }
    };

    std::vector<float> bias_qk_data_tmp;
    bias_qk_data_tmp.reserve(bias_qk_t->numel());
    memcpy(bias_qk_data_tmp.data(),
           bias_qk_data,
           bias_qk_t->numel() * sizeof(float));
    transpose_bias_qk(
        bias_qk_data_tmp.data(), bias_qk_data, head_number_qk, head_size_qk);

    auto weight_qk_shape = nvinfer1::Dims3{1, n_qk, hidden_in_qk};
    auto* weight_qk_tensor =
        AddConstantLayer(weight_qk_data, weight_qk_shape, " ");
    auto bias_qk_shape = nvinfer1::Dims3{1, 1, n_qk};
    auto* bias_qk_tensor = AddConstantLayer(bias_qk_data, bias_qk_shape, " ");
    nvinfer1::ITensor* input_qk_shape_tensor = Shape(input_qk);

    nvinfer1::ILayer* fc_qk_layer = nullptr;
    nvinfer1::ILayer* merge_qk_element_layer = nullptr;
    nvinfer1::MatrixOperation matrix_operation_X =
        nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_Y =
        nvinfer1::MatrixOperation::kTRANSPOSE;
    fc_qk_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       MatrixMultiply,
                                       *input_qk,
                                       matrix_operation_X,
                                       *weight_qk_tensor,
                                       matrix_operation_Y);
    fc_qk_layer->setName(
        ("qk_attention_matrix_multiply(Output: " + output_name + ")").c_str());

    // add qk ElementWiseLayer layer
    nvinfer1::ElementWiseOperation elementwise_operation =
        nvinfer1::ElementWiseOperation::kSUM;
    merge_qk_element_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                  ElementWise,
                                                  *fc_qk_layer->getOutput(0),
                                                  *bias_qk_tensor,
                                                  elementwise_operation);
    merge_qk_element_layer->setName(
        ("multihead_matmul_fc_qk(Output: " + output_name + ")").c_str());

    auto* reshape_after_fc_qk_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *merge_qk_element_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> mha_input_qk_tensor_shape;
    for (int i = 0; i < 5; i++) {
      mha_input_qk_tensor_shape.push_back(Add1DConstantLayer(1));
    }
    mha_input_qk_tensor_shape[0] =
        GetEleTensorOfShape(input_qk_shape_tensor, 0);
    mha_input_qk_tensor_shape[1] =
        GetEleTensorOfShape(input_qk_shape_tensor, 1);
    mha_input_qk_tensor_shape[2] = Add1DConstantLayer(head_number_qk);
    mha_input_qk_tensor_shape[3] = Add1DConstantLayer(2);
    mha_input_qk_tensor_shape[4] = Add1DConstantLayer(head_size_qk);
    reshape_after_fc_qk_layer->setInput(1, *Concat(mha_input_qk_tensor_shape));
    reshape_after_fc_qk_layer->setName(
        ("shuffle_after_fc_qk_multihead_matmul(Output: " + output_name + ")")
            .c_str());

    /* ------------------    weight_v  -------------------------*/
    auto weight_v_name = op_desc.Input("W_v").front();
    auto* weight_v_v = scope.FindVar(weight_v_name);
    auto* weight_v_t = weight_v_v->GetMutable<phi::DenseTensor>();
    float* weight_v_data = nullptr;
    weight_v_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(weight_v_name, *weight_v_t).get().values));
    int n_v = hidden_out_qk;

    // [hidden_in, head_number, head_size]
    // -> [head_number, head_size, hidden_in]
    auto transpose_weight_v = [](const float* src,
                                 float* dst,
                                 int head_number,
                                 int head_size,
                                 int hidden_in) {
      for (int hn = 0; hn < head_number; hn++) {
        for (int hs = 0; hs < head_size; hs++) {
          for (int hi = 0; hi < hidden_in; hi++) {
            int out_index = hn * head_size * hidden_in + hs * hidden_in + hi;
            int in_index = hi * head_number * head_size + hn * head_size + hs;
            dst[out_index] = src[in_index];
          }
        }
      }
    };
    std::vector<float> weight_v_data_tmp;
    weight_v_data_tmp.reserve(weight_v_t->numel());
    memcpy(weight_v_data_tmp.data(),
           weight_v_data,
           weight_v_t->numel() * sizeof(float));
    transpose_weight_v(weight_v_data_tmp.data(),
                       weight_v_data,
                       head_number_qk,
                       head_size_qk,
                       hidden_in_qk);

    /* ------------------    bias_v  -------------------------*/
    auto bias_v_name = op_desc.Input("B_v").front();
    auto* bias_v_v = scope.FindVar(bias_v_name);
    auto* bias_v_t = bias_v_v->GetMutable<phi::DenseTensor>();
    float* bias_v_data = nullptr;
    bias_v_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(bias_v_name, *bias_v_t).get().values));

    auto weight_v_shape = nvinfer1::Dims3{1, n_v, hidden_in_qk};
    auto* weight_v_tensor =
        AddConstantLayer(weight_v_data, weight_v_shape, " ");
    auto bias_v_shape = nvinfer1::Dims3{1, 1, n_v};
    auto* bias_v_tensor = AddConstantLayer(bias_v_data, bias_v_shape, " ");
    nvinfer1::ITensor* input_v_shape_tensor = Shape(input_v);

    nvinfer1::ILayer* fc_v_layer = nullptr;
    nvinfer1::ILayer* merge_v_element_layer = nullptr;
    fc_v_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      MatrixMultiply,
                                      *input_v,
                                      matrix_operation_X,
                                      *weight_v_tensor,
                                      matrix_operation_Y);
    fc_v_layer->setName(
        ("v_attention_matrix_multiply(Output: " + output_name + ")").c_str());

    // add v ElementWiseLayer layer
    merge_v_element_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                 ElementWise,
                                                 *fc_v_layer->getOutput(0),
                                                 *bias_v_tensor,
                                                 elementwise_operation);
    merge_v_element_layer->setName(
        ("multihead_matmul_fc_v(Output: " + output_name + ")").c_str());

    // add shuffle for fc layer
    auto* reshape_after_fc_v_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *merge_v_element_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> mha_input_v_tensor_shape;
    for (int i = 0; i < 5; i++) {
      mha_input_v_tensor_shape.push_back(Add1DConstantLayer(1));
    }
    mha_input_v_tensor_shape[0] = GetEleTensorOfShape(input_v_shape_tensor, 0);
    mha_input_v_tensor_shape[1] = GetEleTensorOfShape(input_v_shape_tensor, 1);
    mha_input_v_tensor_shape[2] = Add1DConstantLayer(head_number_qk);
    mha_input_v_tensor_shape[3] = Add1DConstantLayer(1);
    mha_input_v_tensor_shape[4] = Add1DConstantLayer(head_size_qk);
    reshape_after_fc_v_layer->setInput(1, *Concat(mha_input_v_tensor_shape));
    reshape_after_fc_v_layer->setName(
        ("shuffle_after_fc_v_multihead_matmul(Output: " + output_name + ")")
            .c_str());

    std::vector<nvinfer1::ITensor*> mha_input_tensor_vector{
        reshape_after_fc_qk_layer->getOutput(0),
        reshape_after_fc_v_layer->getOutput(0)};
    nvinfer1::ITensor* mha_input_tensor = Concat(mha_input_tensor_vector, 3);
    auto creator = GetPluginRegistry()->getPluginCreator("fMHA_V2", "1");
    assert(creator != nullptr);
    std::vector<nvinfer1::PluginField> fields{};
    std::unique_ptr<nvinfer1::PluginFieldCollection> plugin_collection(
        new nvinfer1::PluginFieldCollection);
    plugin_collection->nbFields = static_cast<int>(fields.size());
    plugin_collection->fields = fields.data();
    auto plugin = creator->createPlugin("fMHA_V2", plugin_collection.get());
    plugin_collection.reset();
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(mha_input_tensor);
    auto plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin);

    // add shuffle
    nvinfer1::ITensor* batch_tensor =
        GetEleTensorOfShape(input_qk_shape_tensor, 0);
    nvinfer1::ITensor* length_tensor =
        GetEleTensorOfShape(input_qk_shape_tensor, 1);
    auto* reshape_after_mha_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *plugin_layer->getOutput(0));
    std::vector<nvinfer1::ITensor*> reshape_tensor;
    reshape_tensor.push_back(batch_tensor);
    reshape_tensor.push_back(length_tensor);
    reshape_tensor.push_back(Add1DConstantLayer(-1));
    reshape_after_mha_layer->setInput(1, *Concat(reshape_tensor));
    reshape_after_mha_layer->setName(
        ("shuffle_last_multihead_matmul(Output: " + output_name + ")").c_str());
    nvinfer1::ILayer* layer = nullptr;
    layer = reshape_after_mha_layer;
    ReplenishLayerAndOutput(
        layer, "qk_multihead_matmul", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(qk_multihead_matmul, QkMultiheadMatMulOpConverter);
