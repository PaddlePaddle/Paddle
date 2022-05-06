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
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class SparseFcOpConverter : public OpConverter {
 public:
  nvinfer1::ILayer* reshape_before_fc(nvinfer1::ITensor* before_fc,
                                      nvinfer1::Dims x_dim, int x_num_col_dims,
                                      std::string output_name) {
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = x_num_col_dims + 3;
    // padding shape "* x q x 1 x 1"
    for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
      reshape_before_fc_dim.d[i] = 1;
    }
    for (int i = 0; i < x_dim.nbDims; i++) {
      if (i < x_num_col_dims) {
        reshape_before_fc_dim.d[i] = 0;
      } else {
        if (x_dim.d[i] < 0) {
          reshape_before_fc_dim.d[x_num_col_dims] = -1;
          break;
        }
        reshape_before_fc_dim.d[x_num_col_dims] *= x_dim.d[i];
      }
    }
    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *before_fc);
    reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    reshape_before_fc_layer->setName(
        ("sparse_fc_op_reshape_before_fc: Shuffle (Output: " + output_name +
         ")")
            .c_str());
    return reshape_before_fc_layer;
  }

  nvinfer1::ILayer* reshape_after_fc(nvinfer1::ITensor* after_fc,
                                     nvinfer1::Dims x_dim, int x_num_col_dims) {
    // add shuffle after fc
    nvinfer1::Dims reshape_after_fc_dim;
    reshape_after_fc_dim.nbDims = x_num_col_dims + 1;
    for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
      reshape_after_fc_dim.d[i] = 0;
    }
    auto* reshape_after_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *after_fc);
    reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
    return reshape_after_fc_layer;
  }

  plugin::SpmmPluginDynamic* new_spmm_plugin(TensorRTEngine::Weight* weight,
                                             TensorRTEngine::Weight* bias,
                                             const std::string& activation_type,
                                             nvinfer1::DataType type,
                                             int outdim) {
    plugin::SpmmPluginDynamic::Activation act =
        plugin::SpmmPluginDynamic::Activation::kNone;
    if (activation_type == "relu") {
      act = plugin::SpmmPluginDynamic::Activation::kRelu;
    } else if (activation_type == "gelu") {
      act = plugin::SpmmPluginDynamic::Activation::kGelu;
    } else if (activation_type != "") {
      PADDLE_THROW(paddle::platform::errors::Fatal("unknown activation_type %s",
                                                   activation_type.c_str()));
    }
    return new plugin::SpmmPluginDynamic("CustomSpmmPluginDynamic", type,
                                         outdim, weight->get(), bias->get(),
                                         act);
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid sparse_fc op to tensorrt sparse_fc layer";
    framework::OpDesc op_desc(op, nullptr);
    auto output_name = op_desc.Output("Out").front();
    auto input_names = op_desc.InputNames();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    std::cout << "output name: " << output_name << std::endl;
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    auto x_dim = X->getDimensions();
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v,
        platform::errors::NotFound(
            "Can not find %s presistale var of sparse_fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? BOOST_GET_CONST(int, op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? BOOST_GET_CONST(int, op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? BOOST_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, which can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    bool support_int8 = false;
    if (op_desc.HasAttr("support_int8")) {
      support_int8 = BOOST_GET_CONST(bool, op_desc.GetAttr("support_int8"));
    }
    float in_scale = 0;
    if (enable_int8 || support_int8) {
      if (enable_int8) {
        in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      } else {
        in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X"));
      }
      engine_->SetTensorDynamicRange(X, in_scale);
    }
    weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t);

    PADDLE_ENFORCE_EQ(
        Y_t->dims().size(), 2UL,
        platform::errors::InvalidArgument(
            "The sparse_fc's weight should be a matrix with 2 dims, but "
            "it's %d-dimensional.",
            Y_t->dims().size()));  // a matrix
    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    auto regist_fc = [&](nvinfer1::ITensor* inputs, int n_output,
                         TensorRTEngine::Weight* weight,
                         TensorRTEngine::Weight* bias) {
      if (enable_int8 || support_int8) {
        // add conv layer
        float out_scale = 0;
        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"), true,
              platform::errors::InvalidArgument(
                  "must have out threshold in sparse_fc layers in int8 mode"));
          out_scale = BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        } else {
          out_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Out"));
        }
        plugin::SpmmPluginDynamic* plugin = new_spmm_plugin(
            weight, bias, activation_type, nvinfer1::DataType::kINT8, n);
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.emplace_back(inputs);
        auto fc_layer_int8 = engine_->network()->addPluginV2(
            plugin_inputs.data(), plugin_inputs.size(), *plugin);
        fc_layer_int8->setName(
            ("sparse_fc_op_int8_conv1x1: Convolution (Output: " + output_name +
             ")")
                .c_str());
        engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0), out_scale);
        auto* fc_after_reshape_int8 = reshape_after_fc(
            fc_layer_int8->getOutput(0), x_dim, x_num_col_dims);

        RreplenishLayerAndOutput(fc_after_reshape_int8,
                                 "sparse_fc_op_int8_reshape_after_fc: Shuffle",
                                 {output_name}, test_mode);
      } else {
        plugin::SpmmPluginDynamic* plugin = new_spmm_plugin(
            weight, bias, activation_type,
            with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
            n);
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.emplace_back(inputs);
        auto fc_layer_float = engine_->network()->addPluginV2(
            plugin_inputs.data(), plugin_inputs.size(), *plugin);
        fc_layer_float->setName(
            ("sparse_fc_op_float: FullyConnected (Output: " + output_name + ")")
                .c_str());
        auto* fc_after_reshape_float = reshape_after_fc(
            fc_layer_float->getOutput(0), x_dim, x_num_col_dims);

        RreplenishLayerAndOutput(fc_after_reshape_float,
                                 "shuffle_after_sparse_fc", {output_name},
                                 test_mode);
      }
    };

    bool transpose_y = false;
    if (op_desc.HasAttr("transpose_Y")) {
      transpose_y = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    }
    // transpose_y = true;
    int weight_w, weight_h;
    if (!transpose_y) {
      std::vector<float> weight_data_tmp;
      weight_data_tmp.reserve(Y_t->numel());
      memcpy(weight_data_tmp.data(), weight_data, Y_t->numel() * sizeof(float));
      tranpose_weight(weight_data_tmp.data(), weight_data, m, n);
      weight_w = m;
      weight_h = n;
    } else {
      weight_w = m;
      weight_h = n;
    }
    half* half_data = nullptr;
    void* w_data = nullptr;
    if (with_fp16) {
      half_data = new half[Y_t->numel()];
      for (int i = 0; i < Y_t->numel(); i++) {
        half_data[i] = static_cast<half>(weight_data[i]);
      }
      w_data = static_cast<void*>(half_data);
    } else {
      w_data = static_cast<void*>(weight_data);
    }
    size_t n_output = weight_w;
    TensorRTEngine::Weight weight{
        with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
        w_data, static_cast<size_t>(Y_t->numel())};
    weight.dims.assign({weight_w, weight_h});

    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data = engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    // Running the TRT Static Shape mode: x_num_col_dims-1
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }
    // If use tensorrt'oss, the x_dim and x_num_col_dims need change, and can
    // not add Shuffle layer in ernie's multihead.
    if (engine_->use_oss() && engine_->with_ernie() && x_dim.nbDims == 4 &&
        x_dim.d[3] == 1 && x_num_col_dims == 2) {
      if (enable_int8 || support_int8) {
        plugin::SpmmPluginDynamic* plugin = new_spmm_plugin(
            &weight, &bias, activation_type, nvinfer1::DataType::kINT8, n);
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.emplace_back(X);
        auto fc_layer_int8 = engine_->network()->addPluginV2(
            plugin_inputs.data(), plugin_inputs.size(), *plugin);
        RreplenishLayerAndOutput(fc_layer_int8,
                                 "ernie_sparse_fc_op_int8: Convolution",
                                 {output_name}, test_mode);
      } else {
        plugin::SpmmPluginDynamic* plugin = new_spmm_plugin(
            &weight, &bias, activation_type,
            with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
            n);
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.emplace_back(X);
        auto fc_layer_float = engine_->network()->addPluginV2(
            plugin_inputs.data(), plugin_inputs.size(), *plugin);
        RreplenishLayerAndOutput(fc_layer_float, "ernie_sparse_fc_op_float",
                                 {output_name}, test_mode);
      }
    } else {  // need reshape input before and after fc
      PADDLE_ENFORCE_GT(
          x_dim.nbDims, x_num_col_dims,
          platform::errors::InvalidArgument(
              "Params and input dims mismatch. Paddle-TRT SPARSE_FC "
              "converter expects x_dim.nbDims > x_num_col_dims, but "
              "x_dim.nbDims : %d, x_num_col_dims : %d.",
              x_dim.nbDims, x_num_col_dims));
      auto* reshape_before_fc_layer =
          reshape_before_fc(X, x_dim, x_num_col_dims, output_name);
      auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);
      if (enable_int8 || support_int8) {
        engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
      }
      regist_fc(reshape_itensor, n_output, &weight, &bias);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(sparse_fc, SparseFcOpConverter);
