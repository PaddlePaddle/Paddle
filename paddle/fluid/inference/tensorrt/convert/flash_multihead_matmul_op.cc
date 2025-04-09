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

#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::inference::tensorrt {

class FlashMultiheadMatMulOpConverter : public OpConverter {
 public:
  void flash_multihead_matmul_trt(const framework::proto::OpDesc& op,
                                  const framework::Scope& scope,
                                  bool test_mode) {
    VLOG(3)
        << "convert a flash_multihead_matmul op to a corresponding tensorrt "
           "network structure\n";

    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    if (engine_->precision() == phi::DataType::INT8) {
      with_fp16 = true;
    }
    PADDLE_ENFORCE_EQ(
        with_fp16,
        true,
        common::errors::Unimplemented(
            "Trt flash attention oss plugin only support fp16 mode yet."));

    framework::OpDesc op_desc(op, nullptr);
    bool weight_is_constant =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("weight_is_constant"));

    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    int head_number = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));
    int hidden_out = PADDLE_GET_CONST(int, op_desc.GetAttr("hidden_out"));
    int head_size = hidden_out / head_number;

    nvinfer1::ILayer* reshape_before_mha_layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::ILayer* layer = nullptr;
    if (weight_is_constant) {
      // do weight transpose
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

      int n = three * hidden_out;

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
                               t * head_number * head_size + hn * head_size +
                               hs;
                dst[out_index] = src[in_index];
              }
            }
          }
        }
      };
      std::vector<float> weight_data_tmp;
      weight_data_tmp.reserve(weight_t->numel());
      memcpy(weight_data_tmp.data(),
             weight_data,
             weight_t->numel() * sizeof(float));

      transpose_weight(weight_data_tmp.data(),
                       weight_data,
                       three,
                       head_number,
                       head_size,
                       hidden_in);
      nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                               static_cast<void*>(weight_data),
                               static_cast<int32_t>(weight_t->numel())};
#if IS_TRT_VERSION_GE(8600)
      auto* fc_weight_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Constant, nvinfer1::Dims3(1, n, hidden_in), weight);
      auto* fc_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               MatrixMultiply,
                               *input,
                               nvinfer1::MatrixOperation::kNONE,
                               *fc_weight_layer->getOutput(0),
                               nvinfer1::MatrixOperation::kTRANSPOSE);
#else
      nvinfer1::Weights bias{};
      // add shuffle for FullyConnected layer
      std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;

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
          ("shuffle_before_fc_multihead_matmul(Output: " + output_name + ")")
              .c_str());
      nvinfer1::ILayer* fc_layer = nullptr;

      // TODO(wangbojun) need replace FullConnected layer into MatrixMultiply
      fc_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      FullyConnected,
                                      *reshape_before_fc_layer->getOutput(0),
                                      n,
                                      weight,
                                      bias);
#endif
      fc_layer->setName(
          ("multihead_matmul_fc(Output: " + output_name + ")").c_str());
      // add shuffle for fc layer
      reshape_before_mha_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *fc_layer->getOutput(0));
    } else {
      VLOG(3) << "build MatrixMultiply trt layer";
      nvinfer1::MatrixOperation matrix_operation_x =
          nvinfer1::MatrixOperation::kNONE;
      nvinfer1::MatrixOperation matrix_operation_y =
          nvinfer1::MatrixOperation::kNONE;
      std::vector<std::string> qkv_weight_name = {
          "weight_query", "weight_key", "weight_value"};
      std::vector<nvinfer1::ILayer*> qkv_fc_layers(3);
      std::vector<nvinfer1::IShuffleLayer*> reshape_after_fc_layer(3);
      std::vector<nvinfer1::IShuffleLayer*> weight_reshape_before_mm(3);
      std::vector<std::vector<nvinfer1::ITensor*>> reshape_after_fc_shape(3);
      std::vector<nvinfer1::ITensor*> concat_after_qkv_fc_input_tensors;
      for (int i = 0; i < 3; i++) {
        auto* weight_tensor =
            engine_->GetITensor(op_desc.Input(qkv_weight_name[i]).front());
        weight_reshape_before_mm[i] =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *weight_tensor);
        weight_reshape_before_mm[i]->setInput(
            1,
            *Concat({Add1DConstantLayer(1),
                     Add1DConstantLayer(hidden_out),
                     Add1DConstantLayer(hidden_out)}));
        qkv_fc_layers[i] =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 MatrixMultiply,
                                 *input,
                                 matrix_operation_x,
                                 *weight_reshape_before_mm[i]->getOutput(0),
                                 matrix_operation_y);
        reshape_after_fc_shape[i] = {GetEleTensorOfShape(input_shape_tensor, 0),
                                     GetEleTensorOfShape(input_shape_tensor, 1),
                                     Add1DConstantLayer(head_number),
                                     Add1DConstantLayer(1),
                                     Add1DConstantLayer(head_size)};
        reshape_after_fc_layer[i] = TRT_ENGINE_ADD_LAYER(
            engine_, Shuffle, *qkv_fc_layers[i]->getOutput(0));
        reshape_after_fc_layer[i]->setInput(1,
                                            *Concat(reshape_after_fc_shape[i]));
        concat_after_qkv_fc_input_tensors.push_back(
            reshape_after_fc_layer[i]->getOutput(0));
      }

      auto* concat_after_qkv_fc_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Concatenation, concat_after_qkv_fc_input_tensors.data(), 3);
      concat_after_qkv_fc_layer->setAxis(3);
      reshape_before_mha_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Shuffle, *concat_after_qkv_fc_layer->getOutput(0));
    }

    VLOG(3) << "convert fmha_v2 plugin";
    std::vector<nvinfer1::ITensor*> mha_input_tensor_shape;
    for (int i = 0; i < 5; i++) {
      mha_input_tensor_shape.push_back(Add1DConstantLayer(1));
    }
    mha_input_tensor_shape[0] = GetEleTensorOfShape(input_shape_tensor, 0);
    mha_input_tensor_shape[1] = GetEleTensorOfShape(input_shape_tensor, 1);
    mha_input_tensor_shape[2] = Add1DConstantLayer(head_number);
    mha_input_tensor_shape[3] = Add1DConstantLayer(3);
    mha_input_tensor_shape[4] = Add1DConstantLayer(head_size);
    reshape_before_mha_layer->setInput(1, *Concat(mha_input_tensor_shape));
    reshape_before_mha_layer->setName(
        ("shuffle_before_multihead_matmul(Output: " + output_name + ")")
            .c_str());
    auto creator = GetPluginRegistry()->getPluginCreator("fMHA_V2", "1");
    assert("fmha_v2 plugin creator must not be null" && creator != nullptr);
    std::vector<nvinfer1::PluginField> fields{};
    std::unique_ptr<nvinfer1::PluginFieldCollection> plugin_collection(
        new nvinfer1::PluginFieldCollection);
    plugin_collection->nbFields = static_cast<int>(fields.size());
    plugin_collection->fields = fields.data();
    auto plugin = creator->createPlugin("fMHA_V2", plugin_collection.get());
    plugin_collection.reset();
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(reshape_before_mha_layer->getOutput(0));
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
    layer = reshape_after_mha_layer;
    ReplenishLayerAndOutput(
        layer, "flash_multihead_matmul", {output_name}, test_mode);
  }

  void flash_multihead_matmul(const framework::proto::OpDesc& op,
                              const framework::Scope& scope,
                              bool test_mode) {
    VLOG(3) << "convert a flash_multihead_matmul op to a "
               "MemoryEfficientAttention OP "
               "network structure\n";
    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    std::vector<nvinfer1::ITensor*> reshape_before_fc_shape;
    nvinfer1::ITensor* input_shape = Shape(input);

    for (int i = 0; i < 5; i++) {
      reshape_before_fc_shape.push_back(Add1DConstantLayer(1));
    }
    for (int i = 0; i < 3; i++) {
      reshape_before_fc_shape[i] = GetEleTensorOfShape(input_shape, i);
    }
    auto output_name = op_desc.Output("Out")[0];

    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    reshape_before_fc_layer->setInput(1, *Concat(reshape_before_fc_shape));
    reshape_before_fc_layer->setName(
        (std::string("reshape_after_fc_") + "_(Output: " + output_name + ")")
            .c_str());

    int hidden_out = PADDLE_GET_CONST(int, op_desc.GetAttr("hidden_out"));
    int head_number = PADDLE_GET_CONST(int, op_desc.GetAttr("head_number"));
    bool weight_is_constant =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("weight_is_constant"));

    std::vector<std::string> qkv_weight_name = {
        "weight_query", "weight_key", "weight_value"};
    std::vector<nvinfer1::ILayer*> qkv_fc_layers(3);
    std::vector<nvinfer1::IShuffleLayer*> reshape_after_fc_layer(3);
    std::vector<nvinfer1::IShuffleLayer*> weight_reshape_before_mm(3);
    std::vector<std::vector<nvinfer1::ITensor*>> reshape_after_fc_shape(3);

    for (int i = 0; i < 3; ++i) {
      auto weight_name = op_desc.Input(qkv_weight_name[i]).front();
      if (weight_is_constant) {
        auto* weight_value = scope.FindVar(weight_name);
        auto* weight_tensor = weight_value->GetMutable<phi::DenseTensor>();
        float* weight_data = const_cast<float*>(static_cast<const float*>(
            engine_->GetFp32TrtWeight(weight_name, *weight_tensor)
                .get()
                .values));
        const auto& weight_dims = weight_tensor->dims();
        for (int k = 0; k < weight_dims[0]; ++k) {
          for (int j = k; j < weight_dims[1]; ++j) {
            float temp = weight_data[k * weight_dims[1] + j];
            weight_data[k * weight_dims[1] + j] =
                weight_data[j * weight_dims[1] + k];
            weight_data[j * weight_dims[1] + k] = temp;
          }
        }
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(weight_data),
                                 static_cast<int32_t>(weight_tensor->numel())};
#if IS_TRT_VERSION_GE(8600)
        auto* qkv_fc_weight_layer =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 Constant,
                                 nvinfer1::Dims3(1, hidden_out, hidden_out),
                                 weight);
        qkv_fc_layers[i] =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 MatrixMultiply,
                                 *input,
                                 nvinfer1::MatrixOperation::kNONE,
                                 *qkv_fc_weight_layer->getOutput(0),
                                 nvinfer1::MatrixOperation::kTRANSPOSE);
#else
        nvinfer1::Weights bias{};
        qkv_fc_layers[i] =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 FullyConnected,
                                 *reshape_before_fc_layer->getOutput(0),
                                 hidden_out,
                                 weight,
                                 bias);
#endif
        qkv_fc_layers[i]->setName(("multihead_matmul_fc_" + std::to_string(i) +
                                   "_(Output: " + output_name + ")")
                                      .c_str());
      } else {
        nvinfer1::MatrixOperation matrix_operation_x =
            nvinfer1::MatrixOperation::kNONE;
        nvinfer1::MatrixOperation matrix_operation_y =
            nvinfer1::MatrixOperation::kNONE;
        auto* weight_tensor =
            engine_->GetITensor(op_desc.Input(qkv_weight_name[i]).front());
        weight_reshape_before_mm[i] =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *weight_tensor);
        weight_reshape_before_mm[i]->setInput(
            1,
            *Concat({Add1DConstantLayer(1),
                     Add1DConstantLayer(hidden_out),
                     Add1DConstantLayer(hidden_out)}));
        qkv_fc_layers[i] =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 MatrixMultiply,
                                 *input,
                                 matrix_operation_x,
                                 *weight_reshape_before_mm[i]->getOutput(0),
                                 matrix_operation_y);
        qkv_fc_layers[i]->setName(("multihead_matmul_matmul_" +
                                   std::to_string(i) +
                                   "_(Output: " + output_name + ")")
                                      .c_str());
      }

      reshape_after_fc_shape[i] = {
          GetEleTensorOfShape(input_shape, 0),
          GetEleTensorOfShape(input_shape, 1),
          Add1DConstantLayer(head_number),
          Add1DConstantLayer(hidden_out / head_number)};
      reshape_after_fc_layer[i] = TRT_ENGINE_ADD_LAYER(
          engine_, Shuffle, *qkv_fc_layers[i]->getOutput(0));
      reshape_after_fc_layer[i]->setInput(1,
                                          *Concat(reshape_after_fc_shape[i]));
      reshape_after_fc_layer[i]->setName(("reshape_after_fc_" +
                                          std::to_string(i) +
                                          "_(Output: " + output_name + ")")
                                             .c_str());
    }

    nvinfer1::ITensor* query = reshape_after_fc_layer[0]->getOutput(0);
    nvinfer1::ITensor* key = reshape_after_fc_layer[1]->getOutput(0);
    nvinfer1::ITensor* value = reshape_after_fc_layer[2]->getOutput(0);

    std::string op_type = "memory_efficient_attention";
    framework::proto::OpDesc memory_efficient_attention_op;
    memory_efficient_attention_op.set_type(op_type);
    float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("scale"));

    set_op_attr_f(&memory_efficient_attention_op, "scale", scale);
    set_op_attr_i(&memory_efficient_attention_op, "max_seqlen_q", -1);
    set_op_attr_i(&memory_efficient_attention_op, "max_seqlen_k", -1);
    set_op_attr_b(&memory_efficient_attention_op, "causal", false);
    set_op_attr_f64(&memory_efficient_attention_op, "dropout_p", 0);
    set_op_attr_b(&memory_efficient_attention_op, "is_test", true);

    paddle::framework::proto::OpDesc::Var* var =
        memory_efficient_attention_op.add_inputs();
    const std::vector<std::string> input_params = {"query",
                                                   "key",
                                                   "value",
                                                   "bias",
                                                   "cu_seqlens_q",
                                                   "cu_seqlens_k",
                                                   "causal_diagonal",
                                                   "seqlen_k"};
    add_inputs_outputs(var, input_params, input_params);

    var = memory_efficient_attention_op.add_outputs();
    const std::vector<std::string> output_params = {
        "output", "logsumexp", "seed_and_offset"};
    add_inputs_outputs(var, output_params, output_params);

    plugin::GenericPlugin::InputOutPutVarInfo in_out_info;
    using paddle::inference::tensorrt::plugin::GeneratePluginDataType;
    auto input_data_type = GeneratePluginDataType::PLUGIN_FP32;

    PADDLE_ENFORCE_EQ(
        input->getType() == nvinfer1::DataType::kHALF ||
            input->getType() == nvinfer1::DataType::kFLOAT,
        true,
        common::errors::InvalidArgument(
            "This op has no dynamic plugin infershape function!"));

    if (input->getType() == nvinfer1::DataType::kHALF) {
      input_data_type = GeneratePluginDataType::PLUGIN_FP16;
    }
    for (size_t i = 0; i < 3; ++i) {
      in_out_info.inputs_data_type.push_back(input_data_type);
    }
    for (size_t i = 3; i < input_params.size(); ++i) {
      in_out_info.inputs_data_type.push_back(
          GeneratePluginDataType::PLUGIN_OPTIONAL);
    }
    in_out_info.outputs_data_type.push_back(input_data_type);
    in_out_info.outputs_data_type.push_back(
        GeneratePluginDataType::PLUGIN_FP32);
    in_out_info.outputs_data_type.push_back(
        GeneratePluginDataType::PLUGIN_INT64);

    std::vector<nvinfer1::ITensor*> inputs = {
        query,
        key,
        value,
        Add1DConstantLayer(0.0f),
        Add1DConstantLayer(0.0f),
        Add1DConstantLayer(0.0f),
        Add1DConstantLayer(0.0f),
        Add1DConstantLayer(0.0f),
    };

    plugin::GenericPlugin* plugin = new plugin::GenericPlugin(
        memory_efficient_attention_op, in_out_info, true);
    nvinfer1::ILayer* memory_efficient_attention_layer =
        engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    std::vector<nvinfer1::ITensor*> attention_out_shape = {
        GetEleTensorOfShape(input_shape, 0),
        GetEleTensorOfShape(input_shape, 1),
        GetEleTensorOfShape(input_shape, 2)};

    auto reshape_after_attention_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *memory_efficient_attention_layer->getOutput(0));
    reshape_after_attention_layer->setInput(1, *Concat(attention_out_shape));
    reshape_after_attention_layer->setName(
        (std::string("reshape_after_attention_") + "_(Output: " + output_name +
         ")")
            .c_str());
    std::vector<std::string> output_names = {output_name};
    ReplenishLayerAndOutput(
        reshape_after_attention_layer, op_desc.Type(), output_names, test_mode);
  }

  void set_op_attr_i(framework::proto::OpDesc* op,
                     const std::string& name,
                     const int value) {
    auto attr = op->add_attrs();
    attr->set_name(name);
    attr->set_type(paddle::framework::proto::AttrType::INT);
    attr->set_i(value);
  }

  void set_op_attr_f(framework::proto::OpDesc* op,
                     const std::string& name,
                     const float value) {
    auto attr = op->add_attrs();
    attr->set_name(name);
    attr->set_type(paddle::framework::proto::AttrType::FLOAT);
    attr->set_f(value);
  }

  void set_op_attr_b(framework::proto::OpDesc* op,
                     const std::string& name,
                     const bool value) {
    auto attr = op->add_attrs();
    attr->set_name(name);
    attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);
    attr->set_b(value);
  }

  void set_op_attr_f64(framework::proto::OpDesc* op,
                       const std::string& name,
                       const double value) {
    auto attr = op->add_attrs();
    attr->set_name(name);
    attr->set_type(paddle::framework::proto::AttrType::FLOAT64);
    attr->set_float64(value);
  }

  void add_inputs_outputs(paddle::framework::proto::OpDesc::Var* var,
                          const std::vector<std::string>& params_name,
                          const std::vector<std::string>& names) {
    for (size_t i = 0; i < params_name.size(); ++i) {
      var->set_parameter(params_name[i]);
      *var->mutable_arguments()->Add() = names[i];
    }
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    bool use_trt_fma = PADDLE_GET_CONST(bool, op_desc.GetAttr("use_trt_fma"));
    if (use_trt_fma) {
      flash_multihead_matmul_trt(op, scope, test_mode);
    } else {
      flash_multihead_matmul(op, scope, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(flash_multihead_matmul,
                          FlashMultiheadMatMulOpConverter);
