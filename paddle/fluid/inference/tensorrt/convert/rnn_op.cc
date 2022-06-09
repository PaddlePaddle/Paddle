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
    // PADDLE_ENFORCE_GT(seqlen, 0);

    auto* loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
    auto* input_shape_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
    std::string name = "_add_rnn_op_";
    auto* batch_scalar =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input_shape_tensor,
            *Add1DConstantLayer(1, name + "gather_batch_len", true), 0)
            ->getOutput(0);
    auto* seq_len_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Gather, *input_shape_tensor,
                             *Add1DConstantLayer(0, name + "gather_seq_len"), 0)
            ->getOutput(0);

    loop->addTripLimit(*batch_scalar, nvinfer1::TripLimit::kCOUNT);
    nvinfer1::IRecurrenceLayer* rec_layer =
        loop->addRecurrence(*seq_len_tensor);
    // loop->addIterator(*trt_input);
    auto* rec_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Identity, *rec_layer->getOutput(0))
            ->getOutput(0);
    rec_layer->setInput(1, *rec_tensor);
    auto* loop_out_layer =
        loop->addLoopOutput(*rec_tensor, nvinfer1::LoopOutput::kCONCATENATE);
    loop_out_layer->setInput(1, *batch_scalar);

    auto* reshape_seq_len_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *loop_out_layer->getOutput(0));
    auto* batch_tensor =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input_shape_tensor,
            *Add1DConstantLayer(1, name + "gather_batch_len", false), 0)
            ->getOutput(0);
    reshape_seq_len_layer->setInput(1, *batch_tensor);

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
    layer->setSequenceLengths(*reshape_seq_len_layer->getOutput(0));
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

class RnnNativeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid rnn op to tensorrt rnn native layer";

    framework::OpDesc op_desc(op, nullptr);
    // [seq_len, batch ,in_size],
    // [num_layers, batch ,in_size], [num_layers, batch ,in_size]
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* prev_c = engine_->GetITensor(op_desc.Input("PreState")[0]);
    auto* prev_h = engine_->GetITensor(op_desc.Input("PreState")[1]);
    PADDLE_ENFORCE_EQ(input->getDimensions().nbDims, 3);
    PADDLE_ENFORCE_EQ(prev_c->getDimensions().nbDims, 3);
    PADDLE_ENFORCE_EQ(prev_h->getDimensions().nbDims, 3);

    int num_layers = BOOST_GET_CONST(int, op_desc.GetAttr("num_layers"));
    int hidden_size = BOOST_GET_CONST(int, op_desc.GetAttr("hidden_size"));
    int input_size = input->getDimensions().d[2];
    bool is_bidirec = BOOST_GET_CONST(bool, op_desc.GetAttr("is_bidirec"));
    int K = is_bidirec ? 2 : 1;

    int batch = 16;
    int seq_len = input->getDimensions().d[0];

    // extract weights
    // if is_bidirec, make forward and backward weight/bias concated
    std::vector<float*> weight_bias_vec;
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
      if (is_bidirec) {
        auto extract_and_combine_weight = [&](int start) {
          // k and k + 2 is combined !
          // k + 1 and k + 3 is combined !
          for (int k = 0; k < K; k++) {
            std::string var0_name = op_desc.Input("WeightList")[k + start];
            std::string var1_name = op_desc.Input("WeightList")[k + 2 + start];
            auto* var0_v = scope.FindVar(var0_name);
            auto* var1_v = scope.FindVar(var1_name);
            auto* var0_t = var0_v->GetMutable<framework::LoDTensor>();
            auto* var1_t = var1_v->GetMutable<framework::LoDTensor>();
            float* data0_ptr = engine_->GetWeightCPUData(var0_name, var0_t);
            float* data1_ptr = engine_->GetWeightCPUData(var1_name, var1_t);
            float* data_ptr = new float[K * var0_t->numel()];
            // remember free
            memcpy(data_ptr, data0_ptr, sizeof(float) * var0_t->numel());
            memcpy(data_ptr + var0_t->numel(), data1_ptr,
                   sizeof(float) * var1_t->numel());
            weight_bias_vec.push_back(data_ptr);
          }
        };
        extract_and_combine_weight(4 * layer_id);
        extract_and_combine_weight(4 * layer_id + 4 * num_layers);
      } else {
        auto extract_weight = [&](int start) {
          for (int k = 0; k < 2 * K; k++) {
            std::string var_name = op_desc.Input("WeightList")[k + start];
            auto* var_v = scope.FindVar(var_name);
            auto* var_t = var_v->GetMutable<framework::LoDTensor>();
            float* data_ptr = engine_->GetWeightCPUData(var_name, var_t);
            weight_bias_vec.push_back(data_ptr);
          }
        };
        extract_weight(2 * layer_id);                   // filter
        extract_weight(2 * num_layers + 2 * layer_id);  // bias
      }
    }

    auto* loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
    // [seq_len, batch ,in_size],
    auto* input_shape_tensor = Shape(input);
    std::string name = "_add_rnnnative_op_";
    auto* seq_len_scalar =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input_shape_tensor,
            *Add1DConstantLayer(0, name + "gather_seq_len", true), 0)
            ->getOutput(0);
    loop->addTripLimit(*seq_len_scalar, nvinfer1::TripLimit::kCOUNT);

    nvinfer1::ITensor* iter_input_tensor;
    auto* iter_input_forward_tensor =
        loop->addIterator(*input)->getOutput(0);  // [batch, input_size]
    if (is_bidirec) {
      auto* iter_input_reverse_tensor =
          loop->addIterator(*input, 0, true)
              ->getOutput(0);  // [batch, input_size]
      std::vector<nvinfer1::ITensor*> concat_inputs{iter_input_forward_tensor,
                                                    iter_input_reverse_tensor};
      iter_input_tensor = Concat(concat_inputs);
    } else {
      iter_input_tensor = iter_input_forward_tensor;
    }

    nvinfer1::Dims reshape_dim;
    reshape_dim.nbDims = 3;
    reshape_dim.d[0] = K;
    reshape_dim.d[1] = batch;
    reshape_dim.d[2] = input_size;
    auto* tmp_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *iter_input_tensor);
    tmp_layer->setReshapeDimensions(reshape_dim);
    iter_input_tensor = tmp_layer->getOutput(0);

    std::vector<int32_t> tmp_vec(K);
    std::iota(tmp_vec.begin(), tmp_vec.end(), 0);
    auto* first_prev_h = Gather(prev_h, tmp_vec);
    auto* first_prev_c = Gather(prev_c, tmp_vec);

    nvinfer1::IRecurrenceLayer* Hlayer = loop->addRecurrence(*first_prev_h);
    nvinfer1::IRecurrenceLayer* Clayer = loop->addRecurrence(*first_prev_c);

    // k is weight
    // k + 2 is bias
    auto run_matmul_bias = [&](int k, bool is_input) -> nvinfer1::ITensor* {
      int h = 4 * hidden_size;
      int w = is_input ? input_size : hidden_size;

      std::vector<int32_t> weight_shape{K, h, w};
      auto* weight_tensor =
          AddConstantLayer(weight_bias_vec[k], weight_shape, " ");
      std::vector<int32_t> bias_shape{K, 1, h};
      auto* bias_tensor =
          AddConstantLayer(weight_bias_vec[k + 2], bias_shape, " ");

      nvinfer1::ITensor* iter_tensor =
          k % 2 ? Hlayer->getOutput(0) : iter_input_tensor;

      auto* iter_w_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, MatrixMultiply, *iter_tensor,
                               nvinfer1::MatrixOperation::kNONE, *weight_tensor,
                               nvinfer1::MatrixOperation::kTRANSPOSE)
              ->getOutput(0);

      auto* iter_w_b_tensor = Sum(iter_w_tensor, bias_tensor);
      return iter_w_b_tensor;
    };

    nvinfer1::ITensor* iter_input_w_b_tensor = run_matmul_bias(0, true);
    nvinfer1::ITensor* iter_hidden_w_b_tensor = run_matmul_bias(1, false);

    auto* iter_input_hidden_add_tensor =
        Sum(iter_input_w_b_tensor, iter_hidden_w_b_tensor);

    nvinfer1::Dims start_dims = nvinfer1::Dims3{0, 0, 0};
    nvinfer1::Dims size_dims = nvinfer1::Dims3{K, batch, hidden_size};
    nvinfer1::Dims step_dims = nvinfer1::Dims3{1, 1, 1};

    std::vector<nvinfer1::ActivationType> lstm_act{
        nvinfer1::ActivationType::kSIGMOID, nvinfer1::ActivationType::kTANH};
    auto* i_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    i_gate = Act(i_gate, lstm_act[0]);
    start_dims.d[2] = 1 * hidden_size;
    auto* f_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    f_gate = Act(f_gate, lstm_act[0]);
    start_dims.d[2] = 2 * hidden_size;
    auto* c_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    c_gate = Act(c_gate, lstm_act[1]);
    start_dims.d[2] = 3 * hidden_size;
    auto* o_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    o_gate = Act(o_gate, lstm_act[0]);

    // C_t = i_gate * c_gate + f_gate * C_{t-1}
    auto* ic_gate = Prod(i_gate, c_gate);
    auto* fCt1_gate = Prod(f_gate, Clayer->getOutput(0));
    auto* Ct = Sum(ic_gate, fCt1_gate);
    Clayer->setInput(1, *Ct);

    // H_t = tanh(C_t) * o_gate
    auto* tanh_Ct = Act(Ct, lstm_act[1]);
    auto* Ht = Prod(o_gate, tanh_Ct);
    Hlayer->setInput(1, *Ht);

    // Ht: [K, batch, hidden_size]
    nvinfer1::ILayer* layer = nullptr;
    if (is_bidirec) {
      auto* slice_forward_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *Ht, nvinfer1::Dims3{0, 0, 0},
          nvinfer1::Dims3{1, batch, hidden_size}, nvinfer1::Dims3{1, 1, 1});
      auto* slice_reverse_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *Ht, nvinfer1::Dims3{1, 0, 0},
          nvinfer1::Dims3{1, batch, hidden_size}, nvinfer1::Dims3{1, 1, 1});

      auto* layer0 = loop->addLoopOutput(*slice_forward_layer->getOutput(0),
                                         nvinfer1::LoopOutput::kCONCATENATE);
      layer0->setInput(1, *seq_len_scalar);
      auto* layer1 = loop->addLoopOutput(*slice_reverse_layer->getOutput(0),
                                         nvinfer1::LoopOutput::kREVERSE);
      layer1->setInput(1, *seq_len_scalar);

      std::vector<nvinfer1::ITensor*> concat_inputs{layer0->getOutput(0),
                                                    layer1->getOutput(0)};
      auto* concat_slice_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs.data(), 2);
      concat_slice_layer->setAxis(3);
      layer = concat_slice_layer;
    } else {
      layer = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE);
      layer->setInput(1, *seq_len_scalar);
    }

    auto* tensor = layer->getOutput(0);
    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *tensor);
    nvinfer1::Dims final_dims;
    final_dims.nbDims = 3;
    final_dims.d[0] = seq_len;
    final_dims.d[1] = batch;

    if (is_bidirec)
      final_dims.d[2] = hidden_size * K;
    else
      final_dims.d[2] = hidden_size;
    reshape_layer->setReshapeDimensions(final_dims);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(reshape_layer, "rnn", {output_name}, test_mode);
    // free
    for (size_t i = 0; i < weight_bias_vec.size(); i++)
      free(weight_bias_vec[i]);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(rnn, RnnNativeOpConverter);
