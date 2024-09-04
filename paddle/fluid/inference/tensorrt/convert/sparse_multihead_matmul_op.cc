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
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SparseMultiheadMatMulOpConverter : public OpConverter {
 public:
  plugin::SpmmPluginDynamic* new_spmm_plugin(TensorRTEngine::Weight* weight,
                                             TensorRTEngine::Weight* bias,
                                             nvinfer1::DataType type,
                                             int outdim) {
    plugin::SpmmPluginDynamic::Activation act =
        plugin::SpmmPluginDynamic::Activation::kNone;
    return new plugin::SpmmPluginDynamic("CustomSpmmPluginDynamic",
                                         type,
                                         outdim,
                                         weight->get(),
                                         bias->get(),
                                         act);
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a sparse_multihead_matmul op to a corresponding "
               "tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<phi::DenseTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<phi::DenseTensor>();

    float* weight_data = nullptr;
    bool qkv2context_plugin_int8 = op_desc.HasAttr("qkv2context_plugin_int8");
    float in_scale = 0.;

    if (op_desc.HasAttr("Input_scale")) {
      in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      engine_->SetTensorDynamicRange(input, in_scale);
    }
    weight_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(weight_name, *weight_t).get().values));

    float* bias_data = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(bias_name, *bias_t).get().values));
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(
        weight_data_tmp.data(), weight_data, weight_t->numel() * sizeof(float));

    // (hidden_in, 3, hidden_out)
    const auto& weight_dims = weight_t->dims();

    int hidden_in = weight_dims[0];   // channels_in
    int three = weight_dims[1];       // channels_out
    int hidden_out = weight_dims[2];  // channels_out
    int m = hidden_in;
    int n = three * hidden_out;
    auto transpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    transpose_weight(weight_data_tmp.data(), weight_data, m, n);

    int head_number = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];
    bool flag_varseqlen = engine_->use_varseqlen() &&
                          engine_->tensorrt_transformer_posid() != "" &&
                          engine_->tensorrt_transformer_maskid() != "";
    if (engine_->with_dynamic_shape()) {
      if (flag_varseqlen) {
        if (engine_->precision() == phi::DataType::FLOAT32) {
          PADDLE_THROW(common::errors::Fatal(
              "use use_varseqlen must be int8 or half, not float32."));
        }
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(weight_data),
                                 static_cast<int32_t>(weight_t->numel())};
        nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT,
                               static_cast<void*>(bias_data),
                               static_cast<int32_t>(bias_t->numel())};
        auto max_seqlen_tensor = engine_->GetITensor("max_seqlen_tensor");
        auto pos_id_tensor = engine_->GetITensor("pos_id");
        if (engine_->with_interleaved()) {
          VLOG(4) << "fused multihead_matmul op: use_varseqlen and "
                     "with_interleaved";
          if (!op_desc.HasAttr("Input_scale")) {
            PADDLE_THROW(
                common::errors::Fatal("use with_interleaved must be int8."));
          }
          nvinfer1::ILayer* fc_layer = nullptr;
          float dp_probs = 1.0 / 127.0;
          nvinfer1::DimsHW nv_ksize(1, 1);
          fc_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Convolution, *input, n, nv_ksize, weight, bias);
          fc_layer->setName(
              ("Multihead: Convolution/FullyConnected: (Output: " +
               output_name + ")")
                  .c_str());
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("fc_out_threshold"),
              true,
              common::errors::InvalidArgument(
                  "must have out_threshold in multihead layers in int8 mode"));
          float out_scale =
              PADDLE_GET_CONST(float, op_desc.GetAttr("fc_out_threshold"));
          engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
          if (qkv2context_plugin_int8) {
            dp_probs =
                PADDLE_GET_CONST(float, op_desc.GetAttr("dp_probs")) / 127.0;
          }
          auto creator = GetPluginRegistry()->getPluginCreator(
              "CustomQKVToContextPluginDynamic", "3");
          assert(creator != nullptr);
          std::vector<nvinfer1::PluginField> fields{
              {"hidden_size",
               &hidden_out,
               nvinfer1::PluginFieldType::kINT32,
               1},
              {"num_heads",
               &head_number,
               nvinfer1::PluginFieldType::kINT32,
               1}};
          if (qkv2context_plugin_int8) {
            fields.push_back({"dq_probs",
                              &dp_probs,
                              nvinfer1::PluginFieldType::kFLOAT32,
                              1});
          }
          nvinfer1::PluginFieldCollection* plugin_collection =
              static_cast<nvinfer1::PluginFieldCollection*>(malloc(
                  sizeof(*plugin_collection) +
                  fields.size() *
                      sizeof(nvinfer1::PluginField)));  // remember to free
          plugin_collection->nbFields = static_cast<int>(fields.size());
          plugin_collection->fields = fields.data();

          auto plugin = creator->createPlugin("CustomQKVToContextPluginDynamic",
                                              plugin_collection);
          free(plugin_collection);

          std::vector<nvinfer1::ITensor*> plugin_inputs;
          plugin_inputs.emplace_back(fc_layer->getOutput(0));
          plugin_inputs.emplace_back(pos_id_tensor);
          plugin_inputs.emplace_back(
              max_seqlen_tensor);  // max_seqlen, eval_placeholder_3
          auto plugin_layer = engine_->network()->addPluginV2(
              plugin_inputs.data(), plugin_inputs.size(), *plugin);
          layer = plugin_layer;
        } else {
          int head_size = hidden_out / head_number;
          // [3, head_number, head_size, hidden_in] -> [head_number, 3,
          // head_size,
          // hidden_in]
          auto transpose_weight_v2 = [](const float* src,
                                        float* dst,
                                        int three,
                                        int head_number,
                                        int head_size,
                                        int hidden_in) {
            const int HH = head_size * hidden_in;
            for (int i = 0; i < three; ++i) {
              for (int n = 0; n < head_number; ++n) {
                for (int hh = 0; hh < HH; ++hh) {
                  dst[n * three * HH + i * HH + hh] =
                      src[i * head_number * HH + n * HH + hh];
                }
              }
            }
          };
          // [3, head_number, head_size] -> [head_number, 3, head_size]
          auto transpose_bias_v2 =
              [](const float* src, float* dst, int N, int H) {
                for (int i = 0; i < 3; ++i) {
                  for (int n = 0; n < N; ++n) {
                    for (int h = 0; h < H; ++h) {
                      dst[n * 3 * H + i * H + h] = src[i * N * H + n * H + h];
                    }
                  }
                }
              };
          memcpy(weight_data_tmp.data(),
                 weight_data,
                 weight_t->numel() * sizeof(float));
          transpose_weight_v2(weight_data_tmp.data(),
                              weight_data,
                              three,
                              head_number,
                              head_size,
                              hidden_in);

          std::vector<float> bias_data_tmp;
          bias_data_tmp.reserve(bias_t->numel());
          memcpy(
              bias_data_tmp.data(), bias_data, bias_t->numel() * sizeof(float));
          transpose_bias_v2(
              bias_data_tmp.data(), bias_data, head_number, head_size);

          nvinfer1::ILayer* fc_layer = nullptr;
          float dp_probs = 1.0 / 127.0;
          if (op_desc.HasAttr("Input_scale")) {
            nvinfer1::DimsHW nv_ksize(1, 1);
            fc_layer = TRT_ENGINE_ADD_LAYER(
                engine_, Convolution, *input, n, nv_ksize, weight, bias);
          } else {
            fc_layer = TRT_ENGINE_ADD_LAYER(
                engine_, FullyConnected, *input, n, weight, bias);
          }

          if (op_desc.HasAttr("fc_out_threshold")) {
            PADDLE_ENFORCE_EQ(op_desc.HasAttr("fc_out_threshold"),
                              true,
                              common::errors::InvalidArgument(
                                  "must have out threshold in multihead layers "
                                  "in int8 mode"));
            float out_scale =
                PADDLE_GET_CONST(float, op_desc.GetAttr("fc_out_threshold"));
            engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
            if (qkv2context_plugin_int8) {
              dp_probs =
                  PADDLE_GET_CONST(float, op_desc.GetAttr("dp_probs")) / 127.0;
            }
          }
          auto creator = GetPluginRegistry()->getPluginCreator(
              "CustomQKVToContextPluginDynamic", "2");
          assert(creator != nullptr);
          int type = static_cast<int>(nvinfer1::DataType::kHALF);
          if (qkv2context_plugin_int8 &&
              (engine_->precision() == phi::DataType::INT8)) {
            type = static_cast<int>(nvinfer1::DataType::kINT8);
          }
          bool has_mask = true;
          int var_seqlen = 1;
          std::vector<nvinfer1::PluginField> fields{
              {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
              {"hidden_size",
               &hidden_out,
               nvinfer1::PluginFieldType::kINT32,
               1},
              {"num_heads", &head_number, nvinfer1::PluginFieldType::kINT32, 1},
              {"has_mask", &has_mask, nvinfer1::PluginFieldType::kINT32, 1},
              {"var_seqlen",
               &var_seqlen,
               nvinfer1::PluginFieldType::kINT32,
               1}};
          if (qkv2context_plugin_int8) {
            fields.push_back({"dq_probs",
                              &dp_probs,
                              nvinfer1::PluginFieldType::kFLOAT32,
                              1});
          }
          nvinfer1::PluginFieldCollection* plugin_collection =
              static_cast<nvinfer1::PluginFieldCollection*>(malloc(
                  sizeof(*plugin_collection) +
                  fields.size() *
                      sizeof(nvinfer1::PluginField)));  // remember to free
          plugin_collection->nbFields = static_cast<int>(fields.size());
          plugin_collection->fields = fields.data();

          auto plugin = creator->createPlugin("CustomQKVToContextPluginDynamic",
                                              plugin_collection);
          free(plugin_collection);

          std::vector<nvinfer1::ITensor*> plugin_inputs;
          plugin_inputs.emplace_back(fc_layer->getOutput(0));
          plugin_inputs.emplace_back(engine_->GetITensor("qkv_plugin_mask"));
          plugin_inputs.emplace_back(pos_id_tensor);
          plugin_inputs.emplace_back(
              max_seqlen_tensor);  // max_seqlen, eval_placeholder_3
          auto plugin_layer = engine_->network()->addPluginV2(
              plugin_inputs.data(), plugin_inputs.size(), *plugin);
          layer = plugin_layer;
        }
      } else {
        PADDLE_ENFORCE_EQ(
            input->getDimensions().nbDims,
            3,
            common::errors::InvalidArgument(
                "The Input dim of the SparseMultiheadMatMul should be 3, "
                "but it's (%d) now.",
                input->getDimensions().nbDims));
        // transpose weight_data from m * n to  n * m
        auto* input_bias_qk =
            engine_->GetITensor(op_desc.Input("BiasQK").front());

        half* half_data = nullptr;
        void* w_data = nullptr;
        if (with_fp16) {
          half_data = new half[weight_t->numel()];
          for (int i = 0; i < weight_t->numel(); i++) {
            half_data[i] = static_cast<half>(weight_data[i]);
          }
          w_data = static_cast<void*>(half_data);
        } else {
          w_data = static_cast<void*>(weight_data);
        }

        TensorRTEngine::Weight weight{
            with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
            static_cast<void*>(w_data),
            static_cast<size_t>(weight_t->numel())};
        weight.dims.assign({n, m});

        half* half_bias_data = nullptr;
        void* b_data = nullptr;
        if (with_fp16) {
          half_bias_data = new half[bias_t->numel()];
          for (int i = 0; i < bias_t->numel(); i++) {
            half_bias_data[i] = static_cast<half>(bias_data[i]);
          }
          b_data = static_cast<void*>(half_bias_data);
        } else {
          b_data = static_cast<void*>(bias_data);
        }

        TensorRTEngine::Weight bias{
            with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
            b_data,
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
        if (op_desc.HasAttr("Input_scale")) {
          engine_->SetTensorDynamicRange(reshape_before_fc_layer->getOutput(0),
                                         in_scale);
        }
        reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
        reshape_before_fc_layer->setName(
            ("shuffle_before_sparse_multihead_matmul(Output: " + output_name +
             ")")
                .c_str());

        // add layer fc
        nvinfer1::ILayer* fc_layer = nullptr;
        if (op_desc.HasAttr("Input_scale")) {
          plugin::SpmmPluginDynamic* plugin =
              new_spmm_plugin(&weight, &bias, nvinfer1::DataType::kINT8, n);
          std::vector<nvinfer1::ITensor*> plugin_inputs;
          plugin_inputs.emplace_back(reshape_before_fc_layer->getOutput(0));
          fc_layer = engine_->network()->addPluginV2(
              plugin_inputs.data(), plugin_inputs.size(), *plugin);
        } else {
          plugin::SpmmPluginDynamic* plugin =
              new_spmm_plugin(&weight,
                              &bias,
                              with_fp16 ? nvinfer1::DataType::kHALF
                                        : nvinfer1::DataType::kFLOAT,
                              n);
          std::vector<nvinfer1::ITensor*> plugin_inputs;
          plugin_inputs.emplace_back(reshape_before_fc_layer->getOutput(0));
          fc_layer = engine_->network()->addPluginV2(
              plugin_inputs.data(), plugin_inputs.size(), *plugin);
        }

        if (op_desc.HasAttr("fc_out_threshold")) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("fc_out_threshold"),
              true,
              common::errors::InvalidArgument(
                  "must have out threshold in multihead layers in int8 mode"));
          float out_scale =
              PADDLE_GET_CONST(float, op_desc.GetAttr("fc_out_threshold"));
          engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
        }
        fc_layer->setName(
            ("sparse_multihead_matmul_fc(Output: " + output_name + ")")
                .c_str());

        // no need to add shuffle after fc, just change it in
        // QkvToContextPluginDynamic

        // add qkv to context
        int head_size = hidden_out / head_number;
        float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));

        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.push_back(fc_layer->getOutput(0));
        plugin_inputs.push_back(input_bias_qk);

        if (engine_->precision() == phi::DataType::INT8) {
          with_fp16 = true;
        }
        plugin::DynamicPluginTensorRT* plugin =
            new plugin::QkvToContextPluginDynamic(
                hidden_in, head_number, head_size, scale, with_fp16);
        layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 2, plugin);
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    ReplenishLayerAndOutput(
        layer, "multihead_matmul", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(sparse_multihead_matmul,
                          SparseMultiheadMatMulOpConverter);
