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

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    auto input_dims = input->getDimensions();
    bool bias_qk_attr =
        (op_desc.Inputs().find("BiasQK") == op_desc.Inputs().end()) ? false
                                                                    : true;

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

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
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    int head_number = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));

    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];
    bool flag_varseqlen = engine_->use_varseqlen() &&
                          engine_->tensorrt_transformer_posid() != "" &&
                          engine_->tensorrt_transformer_maskid() != "";
    if (engine_->with_dynamic_shape()) {
      if (flag_varseqlen) {
        if (engine_->precision() == AnalysisConfig::Precision::kFloat32) {
          PADDLE_THROW(platform::errors::Fatal(
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
                platform::errors::Fatal("use with_interleaved must be int8."));
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
              platform::errors::InvalidArgument(
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
                              platform::errors::InvalidArgument(
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
              (engine_->precision() == AnalysisConfig::Precision::kInt8)) {
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
        if (input_dims.d[1] <= 384 && !bias_qk_attr &&
            engine_->precision() != AnalysisConfig::Precision::kFloat32) {
          /*
            * input_dims.d[0]: batch(-1)
            * input_dims.d[1]: length:256
            * input_dims.d[2]: hidden_size:768
            input
              |[b,256,768]
              |
            shuffle                 weight   bias
              |[b,256,768,1,1]      |         |
              |_____________________|_________|
              |
              fc
              |[b,256,2304,1,1]
              |
            shuffle                 mask(fake)  pos   max_length
              |[b*256,2304,1,1]       |         |        |
              |                       |         |        |
              |_______________________|_________|________|
              |
              MHA
              |[b*256,768]
              |
            shuffle
              |[b, 256, 768]
              |
              out
          */

          nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(weight_data),
                                   static_cast<int32_t>(weight_t->numel())};
          nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(bias_data),
                                 static_cast<int32_t>(bias_t->numel())};

          /*** transpose the weight and bias ***/
          int head_size = hidden_out / head_number;
          // [3, head_number, head_size, hidden_in] -> [head_number, 3,
          // head_size, hidden_in]
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

          // add shuffle for FullyConnected layer
          std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
          nvinfer1::ITensor* input_shape_tensor = Shape(input);
          for (int i = 0; i < 5; i++) {
            reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
          }
          for (int i = 0; i < 3; i++) {
            reshape_before_fc_shape_tensor[i] =
                GetEleTensorOfShape(input_shape_tensor, i);
          }
          auto* reshape_before_fc_layer =
              TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
          reshape_before_fc_layer->setInput(
              1, *Concat(reshape_before_fc_shape_tensor));
          reshape_before_fc_layer->setName(
              ("shuffle_before_fc_multihead_matmul(Output: " + output_name +
               ")")
                  .c_str());

          // add fc layer
          nvinfer1::ILayer* fc_layer = nullptr;
          fc_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   FullyConnected,
                                   *reshape_before_fc_layer->getOutput(0),
                                   n,
                                   weight,
                                   bias);

          // add shuffle for CustomQKVToContextPluginDynamic layer
          auto* reshape_after_fc_layer =
              TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *fc_layer->getOutput(0));
          std::vector<nvinfer1::ITensor*> mha_input_tensor_shape;
          mha_input_tensor_shape.push_back(Add1DConstantLayer(-1));
          mha_input_tensor_shape.push_back(
              Add1DConstantLayer(hidden_out * 3));  // Q,K,V
          mha_input_tensor_shape.push_back(Add1DConstantLayer(1));
          mha_input_tensor_shape.push_back(Add1DConstantLayer(1));
          reshape_after_fc_layer->setInput(1, *Concat(mha_input_tensor_shape));
          reshape_after_fc_layer->setName(
              ("shuffle_after_fc_multihead_matmul(Output: " + output_name + ")")
                  .c_str());

          // add mha_plugin
          auto creator = GetPluginRegistry()->getPluginCreator(
              "CustomQKVToContextPluginDynamic", "2");
          assert(creator != nullptr);
          // set the attributes of mha_plugin
          int type = static_cast<int>(nvinfer1::DataType::kHALF);
          int var_seqlen = 1;
          bool has_mask = true;
          std::vector<nvinfer1::PluginField> fields{
              {"hidden_size",
               &hidden_out,
               nvinfer1::PluginFieldType::kINT32,
               1},
              {"num_heads", &head_number, nvinfer1::PluginFieldType::kINT32, 1},
              {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
              {"has_mask", &has_mask, nvinfer1::PluginFieldType::kINT32, 1},
              {"var_seqlen",
               &var_seqlen,
               nvinfer1::PluginFieldType::kINT32,
               1}};
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
          // set inputs
          std::vector<nvinfer1::ITensor*> plugin_inputs;
          // input_0 for plugin
          plugin_inputs.emplace_back(reshape_after_fc_layer->getOutput(0));
          // input_1(fake) for plugin
          std::vector<int> mask = {1};
          nvinfer1::ITensor* mask_tensor = Add1DConstantLayer(mask);
          plugin_inputs.emplace_back(mask_tensor);
          // input_2 for plugin
          std::vector<int> pos_id = {0};
          int max_batch = 500;
          for (int i = 1; i < max_batch; i++) {
            pos_id.push_back(i);
          }
          nvinfer1::ITensor* fake_pos_id_tensor = Add1DConstantLayer(pos_id);
          nvinfer1::ITensor* length_tensor =
              GetEleTensorOfShape(input_shape_tensor, 1);
          auto pos_id_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *fake_pos_id_tensor,
                                   *length_tensor,
                                   nvinfer1::ElementWiseOperation::kPROD);
          // size = batch + 1;
          nvinfer1::ITensor* batch_tensor =
              GetEleTensorOfShape(input_shape_tensor, 0);
          std::vector<int> const_data = {1};
          nvinfer1::ITensor* const_tensor = Add1DConstantLayer(const_data);
          auto size_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *batch_tensor,
                                   *const_tensor,
                                   nvinfer1::ElementWiseOperation::kSUM);
          // get size(batch + 1) data from pos_id_tensor
          nvinfer1::Dims start;
          nvinfer1::Dims stride;
          nvinfer1::Dims size;

          start.nbDims = 1;
          stride.nbDims = 1;
          size.nbDims = 1;

          start.d[0] = 0;
          stride.d[0] = 1;
          size.d[0] = 1;

          auto* slice_pos_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Slice, *pos_id_layer->getOutput(0), start, size, stride);
          slice_pos_layer->setInput(2, *size_layer->getOutput(0));
          plugin_inputs.emplace_back(slice_pos_layer->getOutput(0));

          // input_3 for plugin
          std::vector<int> data(500, 1);
          nvinfer1::ITensor* fake_max_seqlen_tensor = Add1DConstantLayer(data);
          auto* slice_max_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Slice, *fake_max_seqlen_tensor, start, size, stride);
          slice_max_layer->setInput(2, *length_tensor);
          plugin_inputs.emplace_back(slice_max_layer->getOutput(0));
          // plugin_layer
          auto plugin_layer = engine_->network()->addPluginV2(
              plugin_inputs.data(), plugin_inputs.size(), *plugin);

          // add shuffle
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
        } else {
          PADDLE_ENFORCE_EQ(
              input->getDimensions().nbDims,
              3,
              platform::errors::InvalidArgument(
                  "The Input dim of the MultiheadMatMul should be 3, "
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
          std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
          nvinfer1::ITensor* input_shape_tensor = Shape(input);

          for (int i = 0; i < 5; i++) {
            reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
          }
          for (int i = 0; i < 3; i++) {
            reshape_before_fc_shape_tensor[i] =
                GetEleTensorOfShape(input_shape_tensor, i);
          }
          auto* reshape_before_fc_layer =
              TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
          if (op_desc.HasAttr("Input_scale")) {
            engine_->SetTensorDynamicRange(
                reshape_before_fc_layer->getOutput(0), in_scale);
          }
          reshape_before_fc_layer->setInput(
              1, *Concat(reshape_before_fc_shape_tensor));
          reshape_before_fc_layer->setName(
              ("shuffle_before_multihead_mamul(Output: " + output_name + ")")
                  .c_str());

          // add layer fc
          nvinfer1::ILayer* fc_layer = nullptr;
          if (op_desc.HasAttr("Input_scale")) {
            nvinfer1::DimsHW nv_ksize(1, 1);
            fc_layer =
                TRT_ENGINE_ADD_LAYER(engine_,
                                     Convolution,
                                     *reshape_before_fc_layer->getOutput(0),
                                     n,
                                     nv_ksize,
                                     weight.get(),
                                     bias.get());
          } else {
            fc_layer =
                TRT_ENGINE_ADD_LAYER(engine_,
                                     FullyConnected,
                                     *reshape_before_fc_layer->getOutput(0),
                                     n,
                                     weight.get(),
                                     bias.get());
          }

          if (op_desc.HasAttr("fc_out_threshold")) {
            PADDLE_ENFORCE_EQ(op_desc.HasAttr("fc_out_threshold"),
                              true,
                              platform::errors::InvalidArgument(
                                  "must have out threshold in multihead layers "
                                  "in int8 mode"));
            float out_scale =
                PADDLE_GET_CONST(float, op_desc.GetAttr("fc_out_threshold"));
            engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
          }
          fc_layer->setName(
              ("multihead_mamul_fc(Output: " + output_name + ")").c_str());

          // no need to add shuffle after fc, just change it in
          // QkvToContextPluginDynamic

          // add qkv to context
          int head_size = hidden_out / head_number;
          float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));

          std::vector<nvinfer1::ITensor*> plugin_inputs;
          plugin_inputs.push_back(fc_layer->getOutput(0));
          auto inputs = op_desc.Inputs();
          bool hasBiasQK =
              (inputs.find("BiasQK") == inputs.end()) ? false : true;
          nvinfer1::ITensor* input_bias_qk = nullptr;
          if (hasBiasQK) {
            input_bias_qk =
                engine_->GetITensor(op_desc.Input("BiasQK").front());
          } else {
            // fake input will be updated in qkv_plugin
            input_bias_qk = fc_layer->getOutput(0);
          }
          plugin_inputs.push_back(input_bias_qk);
          bool with_fp16 =
              engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

          if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
            with_fp16 = true;
          }
          plugin::DynamicPluginTensorRT* plugin =
              new plugin::QkvToContextPluginDynamic(
                  hidden_in, head_number, head_size, scale, with_fp16);
          layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 2, plugin);
        }
      }
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    RreplenishLayerAndOutput(
        layer, "multihead_matmul", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
