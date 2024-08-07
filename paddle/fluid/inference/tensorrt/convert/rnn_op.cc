/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle::inference::tensorrt {

class RnnNativeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert a rnn op to tensorrt rnn layer";

    framework::OpDesc op_desc(op, nullptr);
    // [seq_len, batch ,in_size],
    // [K * num_layers, batch ,in_size], [K * num_layers, batch ,in_size]
    // K is defined below
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* prev_c = engine_->GetITensor(op_desc.Input("PreState")[0]);
    auto* prev_h = engine_->GetITensor(op_desc.Input("PreState")[1]);

    PADDLE_ENFORCE_EQ(input->getDimensions().nbDims,
                      3,
                      common::errors::InvalidArgument(
                          "RNN(LSTM)'s input must be 3 dimensions, i.e. "
                          "[seq_len, batch, input_size],"
                          "but now is %d  dimensions.",
                          input->getDimensions().nbDims));

    PADDLE_ENFORCE_EQ(prev_h->getDimensions().nbDims,
                      3,
                      common::errors::InvalidArgument(
                          "RNN(LSTM)'s PreState(Hidden) must be 3 dimensions, "
                          "i.e. [num_layers, batch, hidden_size],"
                          "but now is %d  dimensions.",
                          prev_h->getDimensions().nbDims));

    PADDLE_ENFORCE_EQ(prev_c->getDimensions().nbDims,
                      3,
                      common::errors::InvalidArgument(
                          "RNN(LSTM)'s PreState(Cell) must be 3 dimensions, "
                          "i.e. [num_layers, batch, hidden_size],"
                          "but now is %d  dimensions.",
                          prev_c->getDimensions().nbDims));

    int num_layers = PADDLE_GET_CONST(int, op_desc.GetAttr("num_layers"));
    int hidden_size = PADDLE_GET_CONST(int, op_desc.GetAttr("hidden_size"));
    int input_size = PADDLE_GET_CONST(int, op_desc.GetAttr("input_size"));
    bool is_bidirec = PADDLE_GET_CONST(bool, op_desc.GetAttr("is_bidirec"));
    int K = is_bidirec ? 2 : 1;

    // extract weights
    // if is_bidirec, make forward and backward weight/bias concated
    std::vector<const float*> weight_bias_vec;
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
            auto* var0_t = var0_v->GetMutable<phi::DenseTensor>();
            auto* var1_t = var1_v->GetMutable<phi::DenseTensor>();
            const float* data0_ptr = reinterpret_cast<const float*>(
                engine_->GetTrtWeight(var0_name, *var0_t).get().values);
            const float* data1_ptr = reinterpret_cast<const float*>(
                engine_->GetTrtWeight(var1_name, *var1_t).get().values);
            float* data_ptr = new float[K * var0_t->numel()];
            // remember free
            memcpy(data_ptr, data0_ptr, sizeof(float) * var0_t->numel());
            memcpy(data_ptr + var0_t->numel(),
                   data1_ptr,
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
            auto* var_t = var_v->GetMutable<phi::DenseTensor>();
            const float* data_ptr = reinterpret_cast<const float*>(
                engine_->GetTrtWeight(var_name, *var_t).get().values);
            weight_bias_vec.push_back(data_ptr);
          }
        };
        extract_weight(2 * layer_id);                   // filter
        extract_weight(2 * num_layers + 2 * layer_id);  // bias
      }
    }
    // [seq_len, batch ,in_size]

    nvinfer1::ITensor* this_input =
        TRT_ENGINE_ADD_LAYER(engine_, Identity, *input)->getOutput(0);

    nvinfer1::ILayer* finally_layer = nullptr;
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
      auto* loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
      auto* input_shape_tensor = Shape(this_input);
      auto* seq_len_scalar = GetEleTensorOfShape(input_shape_tensor, 0, true);
      auto* seq_len_tensor = GetEleTensorOfShape(input_shape_tensor, 0);
      auto* batch_tensor = GetEleTensorOfShape(input_shape_tensor, 1);
      auto* K_tensor = Add1DConstantLayer(K);
      auto* hidden_size_tensor = Add1DConstantLayer(hidden_size);

      if (layer_id > 0) input_size = K * hidden_size;
      auto* input_size_tensor = Add1DConstantLayer(input_size);

      loop->addTripLimit(*seq_len_scalar, nvinfer1::TripLimit::kCOUNT);

      nvinfer1::ITensor* iter_input_tensor;
      auto* iter_input_forward_tensor =
          loop->addIterator(*this_input)->getOutput(0);  // [batch, input_size]

      // this function shuffle tensor -> 4 dims
      auto reshape2four = [&](nvinfer1::ITensor** tensor) {
#if TRT_VERSION == 7234
        auto* tmp_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, **tensor);
        std::vector<nvinfer1::ITensor*> concat_inputs{
            Add1DConstantLayer(1), Add1DConstantLayer(1), Shape(*tensor)};
        tmp_layer->setInput(1, *Concat(concat_inputs));
        *tensor = tmp_layer->getOutput(0);
#endif
      };

      reshape2four(&iter_input_forward_tensor);

      if (is_bidirec) {
        auto* iter_input_reverse_tensor =
            loop->addIterator(*this_input, 0, true)
                ->getOutput(0);  // [batch, input_size]

        reshape2four(&iter_input_reverse_tensor);

        std::vector<nvinfer1::ITensor*> concat_inputs{
            iter_input_forward_tensor, iter_input_reverse_tensor};
        iter_input_tensor = Concat(concat_inputs);
      } else {
        iter_input_tensor = iter_input_forward_tensor;
      }

      auto* tmp_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *iter_input_tensor);

      tmp_layer->setInput(1,
                          *Concat(std::vector<nvinfer1::ITensor*>{
                              K_tensor, batch_tensor, input_size_tensor}));

      iter_input_tensor = tmp_layer->getOutput(0);
      // [K, batch, input_size]

      std::vector<int32_t> tmp_vec(K);
      std::iota(tmp_vec.begin(), tmp_vec.end(), 2 * layer_id);
      auto* first_prev_h = Gather(prev_h, tmp_vec);
      auto* first_prev_c = Gather(prev_c, tmp_vec);

      nvinfer1::IRecurrenceLayer* Hlayer = loop->addRecurrence(*first_prev_h);
      nvinfer1::IRecurrenceLayer* Clayer = loop->addRecurrence(*first_prev_c);

      // k is weight
      // k + 2 is bias
      auto run_matmul_bias = [&](int k, bool is_input) -> nvinfer1::ITensor* {
        int h = 4 * hidden_size;
        int w = is_input ? input_size : hidden_size;
        if (is_input && k > 0) w = K * hidden_size;

        auto weight_shape = nvinfer1::Dims3{K, h, w};
        auto* weight_tensor =
            AddConstantLayer(weight_bias_vec[k], weight_shape, " ");
        auto bias_shape = nvinfer1::Dims3{K, 1, h};
        auto* bias_tensor =
            AddConstantLayer(weight_bias_vec[k + 2], bias_shape, " ");

        nvinfer1::ITensor* iter_tensor =
            k % 2 ? Hlayer->getOutput(0) : iter_input_tensor;

        auto* iter_w_tensor =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 MatrixMultiply,
                                 *iter_tensor,
                                 nvinfer1::MatrixOperation::kNONE,
                                 *weight_tensor,
                                 nvinfer1::MatrixOperation::kTRANSPOSE)
                ->getOutput(0);

        auto* iter_w_b_tensor = Sum(iter_w_tensor, bias_tensor);
        return iter_w_b_tensor;
      };

      nvinfer1::ITensor* iter_input_w_b_tensor =
          run_matmul_bias(layer_id * 4, true);
      nvinfer1::ITensor* iter_hidden_w_b_tensor =
          run_matmul_bias(layer_id * 4 + 1, false);
      auto* iter_input_hidden_add_tensor =
          Sum(iter_input_w_b_tensor, iter_hidden_w_b_tensor);

      nvinfer1::Dims start_dims = nvinfer1::Dims3{0, 0, 0};
      nvinfer1::Dims size_dims = nvinfer1::Dims3{0, 0, 0};
      auto* size_dims_tensor = Concat(std::vector<nvinfer1::ITensor*>{
          K_tensor, batch_tensor, hidden_size_tensor});
      nvinfer1::Dims step_dims = nvinfer1::Dims3{1, 1, 1};

      std::vector<nvinfer1::ActivationType> lstm_act{
          nvinfer1::ActivationType::kSIGMOID, nvinfer1::ActivationType::kTANH};

      auto split_gate = [&](int i, int act_i = 0) -> nvinfer1::ITensor* {
        start_dims.d[2] = i * hidden_size;
        auto* gate_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                Slice,
                                                *iter_input_hidden_add_tensor,
                                                start_dims,
                                                size_dims,
                                                step_dims);
        gate_layer->setInput(2, *size_dims_tensor);
        auto* gate = gate_layer->getOutput(0);
        gate = Act(gate, lstm_act[act_i]);
        return gate;
      };

      auto* i_gate = split_gate(0);
      auto* f_gate = split_gate(1);
      auto* c_gate = split_gate(2, 1);
      auto* o_gate = split_gate(3);

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
      nvinfer1::ITensor* tensor = nullptr;
      if (is_bidirec) {
        auto* slice_forward_layer =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 Slice,
                                 *Ht,
                                 nvinfer1::Dims3{0, 0, 0},
                                 nvinfer1::Dims3{0, 0, 0},
                                 nvinfer1::Dims3{1, 1, 1});
        auto* slice_reverse_layer =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 Slice,
                                 *Ht,
                                 nvinfer1::Dims3{1, 0, 0},
                                 nvinfer1::Dims3{0, 0, 0},
                                 nvinfer1::Dims3{1, 1, 1});
        auto* one_tensor = Add1DConstantLayer(1);
        auto* size_dims_tensor = Concat(std::vector<nvinfer1::ITensor*>{
            one_tensor, batch_tensor, hidden_size_tensor});
        slice_forward_layer->setInput(2, *size_dims_tensor);
        slice_reverse_layer->setInput(2, *size_dims_tensor);

        auto* layer0 = loop->addLoopOutput(*slice_forward_layer->getOutput(0),
                                           nvinfer1::LoopOutput::kCONCATENATE);
        auto* layer1 = loop->addLoopOutput(*slice_reverse_layer->getOutput(0),
                                           nvinfer1::LoopOutput::kREVERSE);
        layer0->setInput(1, *seq_len_scalar);
        layer1->setInput(1, *seq_len_scalar);

        std::vector<nvinfer1::ITensor*> concat_inputs{layer0->getOutput(0),
                                                      layer1->getOutput(0)};
        tensor = Concat(concat_inputs, 3);
      } else {
        layer = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE);
        layer->setInput(1, *seq_len_scalar);
        tensor = layer->getOutput(0);
      }
      finally_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *tensor);
      auto* hidden_size_k_tensor = Add1DConstantLayer(hidden_size * K);
      nvinfer1::ITensor* final_dims_tensor =
          Concat(std::vector<nvinfer1::ITensor*>{
              seq_len_tensor, batch_tensor, hidden_size_k_tensor});
      finally_layer->setInput(1, *final_dims_tensor);
      // update input
      this_input = finally_layer->getOutput(0);
    }

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(finally_layer, "rnn", {output_name}, test_mode);
    // free
    if (is_bidirec) {
      for (auto& weight_bias : weight_bias_vec) delete[] weight_bias;
    }
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(rnn, RnnNativeOpConverter);
