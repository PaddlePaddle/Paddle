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
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    // Shouble be a 5 dims tensor.
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

    float* weight_data =
        engine_->GetWeightCPUData(weight_name, weight_t, false);
    float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t, false);
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(), weight_data,
           weight_t->numel() * sizeof(float));

    // (hidden, 3, all_head_size)
    auto weight_dims = weight_t->dims();

    int hidden = weight_dims[0];         // channels_in
    int three = weight_dims[1];          // channels_out
    int all_head_size = weight_dims[2];  // channels_out
    int m = hidden;
    int n = three * all_head_size;
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    int head_number = BOOST_GET_CONST(int, op_desc.GetAttr("head_number"));

    nvinfer1::ILayer* layer = nullptr;

    if (engine_->with_dynamic_shape()) {
#ifdef USE_NVINFER_PLUGIN
      int head_size = hidden / head_number;
      // [3, Nout, Hout, Nin, Hin] -> [Nout, 3, Hout, Nin, Hin]
      auto transpose_weight_v2 = [](const float* src, float* dst, int N,
                                    int H) {
        const int HNH = H * N * H;
        for (int i = 0; i < 3; ++i) {
          for (int n = 0; n < N; ++n) {
            for (int hnh = 0; hnh < HNH; ++hnh) {
              dst[n * 3 * HNH + i * HNH + hnh] =
                  src[i * N * HNH + n * HNH + hnh];
            }
          }
        }
      };
      // [3, N, H] -> [N, 3, H]
      auto transpose_bias_v2 = [](const float* src, float* dst, int N, int H) {
        for (int i = 0; i < 3; ++i) {
          for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
              dst[n * 3 * H + i * H + h] = src[i * N * H + n * H + h];
            }
          }
        }
      };
      memcpy(weight_data_tmp.data(), weight_data,
             weight_t->numel() * sizeof(float));
      transpose_weight_v2(weight_data_tmp.data(), weight_data, head_number,
                          head_size);
      nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                               static_cast<void*>(weight_data),
                               static_cast<int32_t>(weight_t->numel())};

      std::vector<float> bias_data_tmp;
      bias_data_tmp.reserve(bias_t->numel());
      memcpy(bias_data_tmp.data(), bias_data, bias_t->numel() * sizeof(float));
      transpose_bias_v2(bias_data_tmp.data(), bias_data, head_number,
                        head_size);
      nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT,
                             static_cast<void*>(bias_data),
                             static_cast<int32_t>(bias_t->numel())};

      nvinfer1::Permutation permutation{0, 1, 2, 3, 4};
      auto trans_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      trans_layer->setFirstTranspose(permutation);

      auto* fc_layer = TRT_ENGINE_ADD_LAYER(
          engine_, FullyConnected, *trans_layer->getOutput(0), n, weight, bias);
      /*
            auto pos_tensor = engine_->GetITensor("eval_placeholder_2");
            plugin::CastIntPluginDynamic* cast_plugin =
                new plugin::CastIntPluginDynamic();
            auto cast_layer = engine_->AddPluginV2(&pos_tensor, 1, cast_plugin);

            auto casted_pos_tensor = cast_layer->getOutput(0);
            auto reshape_layer =
                TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *casted_pos_tensor);

            nvinfer1::Dims2 reshape_dim(0, 0);
            nvinfer1::Permutation perm{1, 0, 2};
            reshape_layer->setFirstTranspose(perm);
            reshape_layer->setReshapeDimensions(reshape_dim);
            auto reduce_layer =
                TRT_ENGINE_ADD_LAYER(engine_, Reduce,
         *reshape_layer->getOutput(0),
                                     nvinfer1::ReduceOperation::kMAX, 1, false);
      */
      // auto imask_tensor = engine_->GetITensor("imask_tensor");
      auto imask_tensor = engine_->GetITensor("fused_mha_mask");

      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomQKVToContextPluginDynamic", "1");
      assert(creator != nullptr);
      int type = static_cast<int>((engine_->WithFp16() == 1)
                                      ? nvinfer1::DataType::kHALF
                                      : nvinfer1::DataType::kFLOAT);
      bool has_mask = true;
      const std::vector<nvinfer1::PluginField> fields{
          {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
          {"hidden_size", &hidden, nvinfer1::PluginFieldType::kINT32, 1},
          {"num_heads", &head_number, nvinfer1::PluginFieldType::kINT32, 1},
          {"has_mask", &has_mask, nvinfer1::PluginFieldType::kINT32, 1},
      };
      nvinfer1::PluginFieldCollection* pluginPtr =
          static_cast<nvinfer1::PluginFieldCollection*>(
              malloc(sizeof(*pluginPtr) +
                     fields.size() *
                         sizeof(nvinfer1::PluginField)));  // remember to free
      pluginPtr->nbFields = static_cast<int>(fields.size());
      pluginPtr->fields = fields.data();

      auto pluginObj =
          creator->createPlugin("CustomQKVToContextPluginDynamic", pluginPtr);
      std::vector<nvinfer1::ITensor*> plugin_inputs;
      plugin_inputs.push_back(fc_layer->getOutput(0));
      // plugin_inputs.push_back(reduce_layer->getOutput(0));
      plugin_inputs.push_back(imask_tensor);
      auto plugin_layer = engine_->network()->addPluginV2(
          plugin_inputs.data(), plugin_inputs.size(), *pluginObj);
      assert(plugin_layer != nullptr);
      auto trans_r_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *plugin_layer->getOutput(0));
      assert(trans_r_layer != nullptr);
      trans_r_layer->setFirstTranspose(permutation);
      layer = trans_r_layer;
#else
      // transpose weight_data from m * n to  n * m
      auto* input_bias_qk =
          engine_->GetITensor(op_desc.Input("BiasQK").front());

      TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                    static_cast<void*>(weight_data),
                                    static_cast<size_t>(weight_t->numel())};
      weight.dims.assign({n, m});

      TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(bias_data),
                                  static_cast<size_t>(bias_t->numel())};

      auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *input, n,
                                            weight.get(), bias.get());
      auto* fc_out = fc_layer->getOutput(0);
      // add qkv to context
      int head_size = all_head_size / head_number;
      float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));

      std::vector<nvinfer1::ITensor*> plugin_inputs;
      plugin_inputs.push_back(fc_out);
      plugin_inputs.push_back(input_bias_qk);
      bool ban_fp16 = engine_->disable_trt_plugin_fp16();
      plugin::DynamicPluginTensorRT* plugin =
          new plugin::QkvToContextPluginDynamic(hidden, head_number, head_size,
                                                scale, ban_fp16);
      layer = engine_->AddPluginV2(plugin_inputs.data(), 2, plugin);
#endif
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "multihead_matmul", {output_name},
                             test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
