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
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/transformer_decoder_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/slice_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
using half = paddle::platform::float16;
class TransformerDecoderOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid transformer_decoder op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

    float* weight_data = nullptr;
    float in_scale = 0.;

    if (op_desc.HasAttr("Input_scale")) {
      in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      engine_->SetTensorDynamicRange(input, in_scale);
    }
    weight_data = engine_->GetWeightCPUData(weight_name, weight_t);

    float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t);
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(), weight_data,
           weight_t->numel() * sizeof(float));

    // (hidden_in, 3, hidden_out)
    auto weight_dims = weight_t->dims();

    int hidden_in = weight_dims[0];   // channels_in
    int three = weight_dims[1];       // channels_out
    int hidden_out = weight_dims[2];  // channels_out
    int m = hidden_in;
    int n = three * hidden_out;
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    int head_number = BOOST_GET_CONST(int, op_desc.GetAttr("head_number"));

    // nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];

    if (engine_->with_dynamic_shape()) {
      if (engine_->use_oss()) {
      
      } else {
        PADDLE_ENFORCE_EQ(
            input->getDimensions().nbDims, 3,
            platform::errors::InvalidArgument(
                "The Input dim of the TransformerDecoder should be 3, "
                "but it's (%d) now.",
                input->getDimensions().nbDims));
        // transpose weight_data from m * n to  n * m
        

        TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(weight_data),
                                      static_cast<size_t>(weight_t->numel())};
        weight.dims.assign({n, m});

        TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                    static_cast<void*>(bias_data),
                                    static_cast<size_t>(bias_t->numel())};

        // add shuffle before fc
        nvinfer1::Dims reshape_before_fc_dim;
        reshape_before_fc_dim.nbDims = 5;
        reshape_before_fc_dim.d[0] = 0;
        reshape_before_fc_dim.d[1] = 0;
        reshape_before_fc_dim.d[2] = 0;
        reshape_before_fc_dim.d[3] = 1;
        reshape_before_fc_dim.d[4] = 1;
        auto* reshape_before_fc_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
//        if (op_desc.HasAttr("Input_scale")) {
          engine_->SetTensorDynamicRange(reshape_before_fc_layer->getOutput(0),
                                         1.0);
                                         //in_scale);
//        }
        reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
//        reshape_before_fc_layer->setName(
//            ("transformer_decoder_shuffle(Output: " + output_name + ")")
//                .c_str());

        // add layer fc
        nvinfer1::ILayer* fc_layer = nullptr;
//        if (op_desc.HasAttr("Input_scale")) {
          nvinfer1::DimsHW nv_ksize(1, 1);
        VLOG(5) << "Add Convolution and set scale";
          fc_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Convolution, *reshape_before_fc_layer->getOutput(0), n,
              nv_ksize, weight.get(), bias.get());
//        } else {
//          fc_layer = TRT_ENGINE_ADD_LAYER(
//              engine_, FullyConnected, *reshape_before_fc_layer->getOutput(0),
//              n, weight.get(), bias.get());
//        }

        if (op_desc.HasAttr("fc_out_threshold")) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("fc_out_threshold"), true,
              platform::errors::InvalidArgument(
                  "must have out threshold in multihead layers in int8 mode"));
          float out_scale =
              BOOST_GET_CONST(float, op_desc.GetAttr("fc_out_threshold"));
          engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
        }
        engine_->SetTensorDynamicRange(fc_layer->getOutput(0), 1.0); // debugg
        fc_layer->setName(
            ("transformer_decoder_fc(Output: " + output_name + ")").c_str());

        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

        if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
          with_fp16 = true;
        }

        // no need to add shuffle after fc, just change it in
        // QkvToContextPluginDynamic

        // ################multihead matmul############################
        // FC output: (B, S, 3 * N * H, 1, 1)
        auto* fc_out = fc_layer->getOutput(0);

        auto fc_out_dims = fc_out->getDimensions();
        VLOG(4) << "Before reshape, fc_out dims: ";
        for (int i=0; i<fc_out_dims.nbDims; i++) {
          VLOG(4) << fc_out_dims.d[i];
        }
        int head_size = hidden_out / head_number;
        VLOG(3) << "head_size: " << head_size << "; head_number: " << head_number;
        float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));
        auto* bias_qk =
              engine_->GetITensor(op_desc.Input("BiasQK").front());
        // auto* kv_cache =
        //       engine_->GetITensor(op_desc.Input("KVCache").front());
        auto* k_cache =
              engine_->GetITensor(op_desc.Input("KCache").front());
        auto* v_cache =
              engine_->GetITensor(op_desc.Input("VCache").front());
        auto* time_step =
              engine_->GetITensor(op_desc.Input("TimeStep").front());

        // convert fc from (B, S, 3 * N * H, 1, 1) to [3, B, S, N * H]
        auto* reshape_transpose_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Shuffle, *fc_out);
        nvinfer1::Dims reshape_dim;
        reshape_dim.nbDims = 4;
        reshape_dim.d[0] = 0;
        reshape_dim.d[1] = 0;
        reshape_dim.d[2] = 3;
        reshape_dim.d[3] = head_number * head_size;
        reshape_transpose_layer->setReshapeDimensions(reshape_dim); // (B, S, 3, N * H)
        reshape_transpose_layer->setSecondTranspose({2,0,1,3}); // [3, B, S, N * H]
        fc_out = reshape_transpose_layer->getOutput(0);
        
        fc_out_dims = fc_out->getDimensions();
        VLOG(4) << "After reshape, fc_out dims: ";
        for (int i=0; i<fc_out_dims.nbDims; i++) {
          VLOG(4) << fc_out_dims.d[i];
        }

        // convert bias_qk from [B, N, 1, S'] to [B, 1, 1, S']
        // plugin::SlicePluginDynamic* slice_plugin =
        //     new plugin::SlicePluginDynamic({0}, {1}, {1}, with_fp16);
        // layer = engine_->AddDynamicPlugin(&bias_qk, 1, slice_plugin);
        // bias_qk = layer->getOutput(0);
        // auto bias_qk_dims = bias_qk->getDimensions();
        // VLOG(4) << "After reshape, bias_qk dims: ";
        // for (int i=0; i<bias_qk_dims.nbDims; i++) {
        //   VLOG(4) << bias_qk_dims.d[i];
        // }

        // concat k_cache and v_cache from [B, N, S', H] to [2, B, N, S', H]
        {
          auto dims_ = k_cache->getDimensions();
          VLOG(4) << "Before concat, k_cache dims: ";
          for (int i=0; i<dims_.nbDims; i++) {
            VLOG(4) << dims_.d[i];
          }
        }
        {
          auto dims_ = v_cache->getDimensions();
          VLOG(4) << "Before concat, v_cache dims: ";
          for (int i=0; i<dims_.nbDims; i++) {
            VLOG(4) << dims_.d[i];
          }
        }

        // plugin::DynamicPluginTensorRT* id_plugin = 
        //     new plugin::GeluPluginDynamic(with_fp16);
        // auto* id_layer = engine_->AddDynamicPlugin(&k_cache, 1, id_plugin);
        // k_cache = id_layer->getOutput(0);

        // id_plugin = 
        //     new plugin::GeluPluginDynamic(with_fp16);
        // id_layer = engine_->AddDynamicPlugin(&v_cache, 1, id_plugin);
        // v_cache = id_layer->getOutput(0);

  //       std::vector<nvinfer1::ITensor*> inputs;
  //       inputs.push_back(k_cache);
  //       inputs.push_back(v_cache);
  //       auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Concatenation, inputs.data(), inputs.size());
  // //      layer->setName(("transformer_decoder_concat_kv_cache(Output: " + output_name + ")").c_str());
  //       layer->setAxis(0);
  //       auto* kv_cache = layer->getOutput(0); // [2*B, N, S', H]
  //       auto dims_ = kv_cache->getDimensions();
  //       VLOG(4) << "Before reshape, kv_cache dims: ";
  //       for (int i=0; i<dims_.nbDims; i++) {
  //         VLOG(4) << dims_.d[i];
  //       }
        // reshape_transpose_layer = TRT_ENGINE_ADD_LAYER(
        //       engine_, Shuffle, *k_cache);
        // reshape_dim.nbDims = 5;
        // reshape_dim.d[0] = 2;
        // reshape_dim.d[1] = -1;
        // reshape_dim.d[2] = dims_.d[1];
        // reshape_dim.d[3] = dims_.d[2];
        // reshape_dim.d[4] = dims_.d[3];
        // reshape_transpose_layer->setReshapeDimensions(reshape_dim); // [2, B, N, S', H]
        // kv_cache = reshape_transpose_layer->getOutput(0);
        
        // auto kv_cache_dims = kv_cache->getDimensions();
        // VLOG(4) << "After reshape, kv_cache dims: ";
        // for (int i=0; i<kv_cache_dims.nbDims; i++) {
        //   VLOG(4) << kv_cache_dims.d[i];
        // }

        auto time_step_dims = time_step->getDimensions();
        VLOG(4) << "time_step dims: ";
        for (int i=0; i<time_step_dims.nbDims; i++) {
          VLOG(4) << time_step_dims.d[i];
        }
        
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.push_back(fc_out); // [3, B, S, N * H]
        plugin_inputs.push_back(bias_qk);// [bsz, 1, 1, time_step(cache_seq_length)+1] 
        plugin_inputs.push_back(k_cache); // [B, num_head, cache_seq_len(padding max_seq_len), dim_head]
        plugin_inputs.push_back(v_cache); // [B, num_head, cache_seq_len(padding max_seq_len), dim_head]
        plugin_inputs.push_back(time_step); //[]
        // qkv bias: [3, num_head, dim_head]
        
        auto half_bias_data = new half[bias_t->numel()];
        for (int i = 0; i < bias_t->numel(); i++) {
            half_bias_data[i] = static_cast<half>(bias_data[i]);
        }

        plugin::DynamicPluginTensorRT* decoder_plugin = new plugin::TransformerDecoderPluginDynamic<half>(half_bias_data, bias_t->numel(), head_number,
                                                  head_size, scale, with_fp16);
        auto* decoder_layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 5, decoder_plugin);

        // decoder_layer->setName(("transformer_decoder_mma(Output: " + output_name + ")").c_str());

        auto* decoder_out = decoder_layer->getOutput(0);
        auto decoder_out_dims = decoder_out->getDimensions();
        VLOG(4) << "decoder_out_dims: ";
        for (int i=0; i<decoder_out_dims.nbDims; i++) {
          VLOG(4) << decoder_out_dims.d[i];
        }
        RreplenishLayerAndOutput(decoder_layer, "transformer_decoder", {output_name},
                             test_mode);
      }
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    // RreplenishLayerAndOutput(decoder_layer, "transformer_decoder", {output_name},
    //                         test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(transformer_decoder, TransformerDecoderOpConverter);
