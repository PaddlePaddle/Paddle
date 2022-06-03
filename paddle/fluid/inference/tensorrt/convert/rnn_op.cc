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

namespace paddle {
namespace inference {
namespace tensorrt {

class RnnOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid rnn op to tensorrt rnn layer";

    framework::OpDesc op_desc(op, nullptr);
    // [seq_len, batch ,in_size],
    // [num_layers, batch ,in_size], [num_layers, batch ,in_size]
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* prev_c = engine_->GetITensor(op_desc.Input("PreState")[0]);
    auto* prev_h = engine_->GetITensor(op_desc.Input("PreState")[1]);
    std::cout << input->getDimensions().d[0] << " "
              << input->getDimensions().d[1] << " "
              << input->getDimensions().d[2] << std::endl;
    PADDLE_ENFORCE_EQ(input->getDimensions().nbDims, 3);
    PADDLE_ENFORCE_EQ(prev_c->getDimensions().nbDims, 3);
    PADDLE_ENFORCE_EQ(prev_h->getDimensions().nbDims, 3);

    auto* trt_input_trans_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    auto* trt_prev_c_trans_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *prev_c);
    auto* trt_prev_h_trans_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *prev_h);
    nvinfer1::Permutation perm;
    perm.order[0] = 1;
    perm.order[1] = 0;
    perm.order[2] = 2;
    trt_input_trans_layer->setFirstTranspose(perm);
    trt_prev_c_trans_layer->setFirstTranspose(perm);
    trt_prev_h_trans_layer->setFirstTranspose(perm);

    auto* trt_input = trt_input_trans_layer->getOutput(0);
    auto* trt_prev_c = trt_prev_c_trans_layer->getOutput(0);
    auto* trt_prev_h = trt_prev_h_trans_layer->getOutput(0);

    auto trt_input_dims = trt_input->getDimensions();
    int seqlen = trt_input_dims.d[1];
    PADDLE_ENFORCE_GT(seqlen, 0);

    int num_layers = BOOST_GET_CONST(int, op_desc.GetAttr("num_layers"));
    int hidden_size = BOOST_GET_CONST(int, op_desc.GetAttr("hidden_size"));
    bool is_bidirec = BOOST_GET_CONST(bool, op_desc.GetAttr("is_bidirec"));
    int K = 1;
    nvinfer1::IRNNv2Layer* layer = TRT_ENGINE_ADD_LAYER(
        engine_, RNNv2, *trt_input, num_layers, hidden_size, seqlen,
        nvinfer1::RNNOperation::kLSTM);
    if (is_bidirec) {
      layer->setDirection(nvinfer1::RNNDirection::kBIDIRECTION);
      K = 2;
    }
    layer->setHiddenState(*trt_prev_c);
    layer->setCellState(*trt_prev_h);
    // 4 = 2(w) + 2 (bias)
    int weight_list_len = K * layer->getLayerCount() * 4;
    for (int k = 0; k < layer->getLayerCount() * K; k++) {
      auto set_weight_to_rnn = [&](int kk, bool is_w, int weight_list_k) {
        std::string filter_var_name =
            op_desc.Input("WeightList")[weight_list_k];
        std::string bias_var_name =
            op_desc.Input("WeightList")[weight_list_k + weight_list_len / 2];
        auto* filter_v = scope.FindVar(filter_var_name);
        auto* bias_v = scope.FindVar(bias_var_name);
        auto* filter_t = filter_v->GetMutable<framework::LoDTensor>();
        auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();
        float* weight_data =
            engine_->GetWeightCPUData(filter_var_name, filter_t);
        float* bias_data = engine_->GetWeightCPUData(bias_var_name, bias_t);
        auto filter_dims = filter_t->dims();
        auto bias_dims = bias_t->dims();
        auto vol = filter_dims[0] * filter_dims[1] / 4;
        TensorRTEngine::Weight weight;

        weight = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(weight_data),
                                        static_cast<size_t>(vol));
        layer->setWeightsForGate(kk, nvinfer1::RNNGateType::kINPUT, is_w,
                                 weight.get());
        weight = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(weight_data + vol),
                                        static_cast<size_t>(vol));
        layer->setWeightsForGate(kk, nvinfer1::RNNGateType::kFORGET, is_w,
                                 weight.get());
        weight =
            TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(weight_data + vol * 2),
                                   static_cast<size_t>(vol));
        layer->setWeightsForGate(kk, nvinfer1::RNNGateType::kCELL, is_w,
                                 weight.get());
        weight =
            TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(weight_data + vol * 3),
                                   static_cast<size_t>(vol));
        layer->setWeightsForGate(kk, nvinfer1::RNNGateType::kOUTPUT, is_w,
                                 weight.get());

        vol = bias_dims[0] / 4;

        TensorRTEngine::Weight bias;
        bias = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(bias_data),
                                      static_cast<size_t>(vol));
        layer->setBiasForGate(kk, nvinfer1::RNNGateType::kINPUT, is_w,
                              bias.get());
        bias = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(bias_data + vol),
                                      static_cast<size_t>(vol));
        layer->setBiasForGate(kk, nvinfer1::RNNGateType::kFORGET, is_w,
                              bias.get());
        bias = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(bias_data + vol * 2),
                                      static_cast<size_t>(vol));
        layer->setBiasForGate(kk, nvinfer1::RNNGateType::kCELL, is_w,
                              bias.get());
        bias = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(bias_data + vol * 3),
                                      static_cast<size_t>(vol));
        layer->setBiasForGate(kk, nvinfer1::RNNGateType::kOUTPUT, is_w,
                              bias.get());
      };
      set_weight_to_rnn(k, true, 2 * k);
      set_weight_to_rnn(k, false, 2 * k + 1);
    }
    auto post_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
    post_layer->setFirstTranspose(perm);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(post_layer, "rnn", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(rnn, RnnOpConverter);
