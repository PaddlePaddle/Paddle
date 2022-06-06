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

    // int num_layers = BOOST_GET_CONST(int, op_desc.GetAttr("num_layers"));
    // int hidden_size = BOOST_GET_CONST(int, op_desc.GetAttr("hidden_size"));
    // bool is_bidirec = BOOST_GET_CONST(bool, op_desc.GetAttr("is_bidirec"));
    // int K = 1;

    auto* loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
    auto* input_shape_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
    std::string name = "_add_rnn_op_";
    auto* seq_len_scalar =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input_shape_tensor,
            *Add1DConstantLayer(0, name + "gather_seq_len", true), 0)
            ->getOutput(0);
    loop->addTripLimit(*seq_len_scalar, nvinfer1::TripLimit::kCOUNT);
    auto* iter_input_tensor =
        loop->addIterator(*input)->getOutput(0);  // [batch, input_size]

    auto* first_prev_h =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *prev_h,
            *Add1DConstantLayer(0, name + "first_prev_h", true), 0)
            ->getOutput(0);
    auto* first_prev_c =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *prev_c,
            *Add1DConstantLayer(0, name + "first_prev_c", true), 0)
            ->getOutput(0);

    nvinfer1::IRecurrenceLayer* Hlayer = loop->addRecurrence(*first_prev_h);
    nvinfer1::IRecurrenceLayer* Clayer = loop->addRecurrence(*first_prev_c);

    auto run_matmul_bias = [&](int weight_k, int bias_k) -> nvinfer1::ITensor* {
      std::string filter_var_name = op_desc.Input("WeightList")[weight_k];
      std::string bias_var_name = op_desc.Input("WeightList")[bias_k];
      auto* filter_v = scope.FindVar(filter_var_name);
      auto* bias_v = scope.FindVar(bias_var_name);
      auto* filter_t = filter_v->GetMutable<framework::LoDTensor>();
      auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();
      float* weight_data = engine_->GetWeightCPUData(filter_var_name, filter_t);
      float* bias_data = engine_->GetWeightCPUData(bias_var_name, bias_t);
      auto filter_dims = phi::make_ddim({0});
      filter_dims = filter_t->dims();
      auto bias_dims = phi::make_ddim({0});
      bias_dims = bias_t->dims();
      auto vol = filter_dims[0] * filter_dims[1];
      TensorRTEngine::Weight weight;
      weight = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(weight_data),
                                      static_cast<size_t>(vol));
      nvinfer1::Dims weight_shape;
      weight_shape.nbDims = 2;
      weight_shape.d[0] = filter_dims[0];
      weight_shape.d[1] = filter_dims[1];
      auto* weight_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, weight_shape, weight.get())
              ->getOutput(0);

      vol = bias_dims[0];
      TensorRTEngine::Weight bias_weight;
      bias_weight = TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT,
                                           static_cast<void*>(bias_data),
                                           static_cast<size_t>(vol));
      nvinfer1::Dims bias_shape;
      bias_shape.nbDims = 2;
      bias_shape.d[0] = 1;
      bias_shape.d[1] = vol;
      auto* bias_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, bias_shape, bias_weight.get())
              ->getOutput(0);

      nvinfer1::ITensor* iter_tensor = nullptr;
      if (weight_k % 2 == 0) {
        iter_tensor = iter_input_tensor;
      } else {
        iter_tensor = Hlayer->getOutput(0);
      }

      auto* iter_w_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, MatrixMultiply, *iter_tensor,
                               nvinfer1::MatrixOperation::kNONE, *weight_tensor,
                               nvinfer1::MatrixOperation::kTRANSPOSE)
              ->getOutput(0);

      auto* iter_w_b_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *iter_w_tensor,
                               *bias_tensor,
                               nvinfer1::ElementWiseOperation::kSUM)
              ->getOutput(0);
      return iter_w_b_tensor;
    };

    nvinfer1::ITensor* iter_input_w_b_tensor = run_matmul_bias(0, 2);
    nvinfer1::ITensor* iter_hidden_w_b_tensor = run_matmul_bias(1, 3);
    auto* iter_input_hidden_add_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *iter_input_w_b_tensor,
                             *iter_hidden_w_b_tensor,
                             nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);

    nvinfer1::Dims start_dims;
    start_dims.nbDims = 2;
    start_dims.d[0] = 0;
    start_dims.d[1] = 0;
    nvinfer1::Dims size_dims;
    size_dims.nbDims = 2;
    size_dims.d[0] = 16;
    size_dims.d[1] = 32;
    nvinfer1::Dims step_dims;
    step_dims.nbDims = 2;
    step_dims.d[0] = 1;
    step_dims.d[1] = 1;

    auto* i_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    i_gate = TRT_ENGINE_ADD_LAYER(engine_, Activation, *i_gate,
                                  nvinfer1::ActivationType::kSIGMOID)
                 ->getOutput(0);
    start_dims.d[1] = 32;
    auto* f_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    f_gate = TRT_ENGINE_ADD_LAYER(engine_, Activation, *f_gate,
                                  nvinfer1::ActivationType::kSIGMOID)
                 ->getOutput(0);
    start_dims.d[1] = 64;
    auto* c_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    c_gate = TRT_ENGINE_ADD_LAYER(engine_, Activation, *c_gate,
                                  nvinfer1::ActivationType::kTANH)
                 ->getOutput(0);
    start_dims.d[1] = 96;
    auto* o_gate =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *iter_input_hidden_add_tensor,
                             start_dims, size_dims, step_dims)
            ->getOutput(0);
    o_gate = TRT_ENGINE_ADD_LAYER(engine_, Activation, *o_gate,
                                  nvinfer1::ActivationType::kSIGMOID)
                 ->getOutput(0);

    auto* ic_gate = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *i_gate, *c_gate,
                                         nvinfer1::ElementWiseOperation::kPROD)
                        ->getOutput(0);

    auto* fCt1_gate = TRT_ENGINE_ADD_LAYER(
                          engine_, ElementWise, *f_gate, *Clayer->getOutput(0),
                          nvinfer1::ElementWiseOperation::kPROD)
                          ->getOutput(0);

    auto* Ct = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *ic_gate, *fCt1_gate,
                                    nvinfer1::ElementWiseOperation::kSUM)
                   ->getOutput(0);
    Clayer->setInput(1, *Ct);

    auto* tanh_Ct = TRT_ENGINE_ADD_LAYER(engine_, Activation, *Ct,
                                         nvinfer1::ActivationType::kTANH)
                        ->getOutput(0);
    auto* Ht = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *o_gate, *tanh_Ct,
                                    nvinfer1::ElementWiseOperation::kPROD)
                   ->getOutput(0);
    Hlayer->setInput(1, *Ht);

    auto* layer = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE);
    layer->setInput(1, *seq_len_scalar);
    // loop->addLoopOutput(*Hlayer->getOutput(0),
    // nvinfer1::LoopOutput::kLAST_VALUE)
    // loop->addLoopOutput(*Clayer->getOutput(0),
    // nvinfer1::LoopOutput::kLAST_VALUE)

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "rnn", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(rnn, RnnNativeOpConverter);
