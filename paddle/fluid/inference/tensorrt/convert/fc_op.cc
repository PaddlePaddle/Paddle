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
class FcOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
    framework::OpDesc op_desc(op, nullptr);

    auto input_names = op_desc.InputNames();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v, platform::errors::NotFound(
                 "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    const int x_num_col_dims =
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
    float in_scale = 0.;
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      CHECK(op_desc.HasAttr(i_name + "_scale"));
      in_scale =
          BOOST_GET_CONST(float, op_desc.GetAttr(i_name + "_scale")) * 127;
      auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(),
                                              Y_t, true, weight_scale);
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    size_t n_output = Y_t->dims()[1];

    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];

    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };

    auto regist_fc = [&](nvinfer1::ITensor* inputs, int n_output,
                         TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias) {
      nvinfer1::ILayer* fc_layer = nullptr;
      if (enable_int8) {
        PADDLE_ENFORCE_EQ(
            op_desc.HasAttr("out_threshold"), true,
            platform::errors::InvalidArgument(
                "must have out threshold in fc layers in int8 mode"));
        float out_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        nvinfer1::DimsHW nv_ksize(1, 1);
        fc_layer = TRT_ENGINE_ADD_LAYER(engine_, Convolution, *inputs, n_output,
                                        nv_ksize, weight.get(), bias.get());
        engine_->SetTensorDynamicRange(fc_layer->getOutput(0), out_scale);
      } else {
        fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *inputs,
                                        n_output, weight.get(), bias.get());
      }

      auto output_name = op_desc.Output("Out").front();
      if (activation_type == "relu") {
        nvinfer1::IActivationLayer* relu_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Activation, *(fc_layer->getOutput(0)),
                                 nvinfer1::ActivationType::kRELU);
        RreplenishLayerAndOutput(relu_layer, "fc", {output_name}, test_mode);
      } else {
        RreplenishLayerAndOutput(fc_layer, "fc", {output_name}, test_mode);
      }
    };

    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(Y_t->numel());
    memcpy(weight_data_tmp.data(), weight_data, Y_t->numel() * sizeof(float));
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    weight.dims.assign({n, m});

    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data =
          engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t, false);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    if (engine_->with_dynamic_shape()) {
      // not NCHW layout, but NLP layout with added 'x 1 x 1'
      auto x_dim = X->getDimensions();
      if (engine_->use_oss() && engine_->with_ernie() && x_dim.nbDims == 4 &&
          x_dim.d[2] == 1 && x_dim.d[3] == 1 && x_num_col_dims == 2) {
        // fc which is just after self attention
        regist_fc(X, n_output, weight, bias);
        return;
      }
      PADDLE_ENFORCE_LE(
          x_dim.nbDims - x_num_col_dims, 3,
          platform::errors::InvalidArgument(
              "Params and input dims mismatch. Paddle-TRT FC "
              "converter expects x_dim.nbDims - x_num_col_dims <= 3, but "
              "x_dim.nbDims = %d, x_num_col_dims = %d.",
              x_dim.nbDims, x_num_col_dims));
      auto output_name = op_desc.Output("Out").front();
      // add shuffle before fc
      nvinfer1::Dims reshape_before_fc_dim;
      // padding shape "x 1 x 1"
      int padding_length = 3 - (x_dim.nbDims - x_num_col_dims);
      reshape_before_fc_dim.nbDims = x_dim.nbDims + padding_length;
      int cur_dim_index = reshape_before_fc_dim.nbDims - 1;
      while (padding_length-- > 0) {
        reshape_before_fc_dim.d[cur_dim_index--] = 1;
      }
      while (cur_dim_index >= 0) {
        reshape_before_fc_dim.d[cur_dim_index--] = 0;
      }

      auto* reshape_before_fc_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
      reshape_before_fc_layer->setName(
          ("shuffle_before_fc(Output: " + output_name + ")").c_str());

      // add fc layer
      auto* fc_layer = TRT_ENGINE_ADD_LAYER(
          engine_, FullyConnected, *reshape_before_fc_layer->getOutput(0),
          n_output, weight.get(), bias.get());
      fc_layer->setName(("fc_layer(Output: " + output_name + ")").c_str());

      // add shuffle after fc
      nvinfer1::Dims reshape_after_fc_dim;
      reshape_after_fc_dim.nbDims = x_num_col_dims + 1;
      for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
        reshape_after_fc_dim.d[i] = 0;
      }

      auto* reshape_after_fc_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *fc_layer->getOutput(0));
      reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);

      if (activation_type == "relu") {
        reshape_after_fc_layer->setName(
            ("shuffle_after_fc(Output: " + output_name + ")").c_str());
        nvinfer1::IActivationLayer* relu_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Activation, *(reshape_after_fc_layer->getOutput(0)),
            nvinfer1::ActivationType::kRELU);
        RreplenishLayerAndOutput(relu_layer, "relu_after_fc_shuffle",
                                 {output_name}, test_mode);
      } else {
        RreplenishLayerAndOutput(reshape_after_fc_layer, "shuffle_after_fc",
                                 {output_name}, test_mode);
      }
      return;
    }
    // in order to handle situations in NLP models(input dims < 3,
    // x_num_col_dims != 1, etc.), reshape input to perform FC correctly.
    auto* reshape_itensor = X;
    int input_dims = X->getDimensions().nbDims;
    auto input_d = X->getDimensions().d;
    int reshape_dim3[3] = {0};
    int reshape_dim4[4] = {0};
    PADDLE_ENFORCE_LE(x_num_col_dims, input_dims,
                      platform::errors::InvalidArgument(
                          "Params and input dims mismatch. Paddle-TRT FC "
                          "converter expects x_num_col_dims <= input dims"));
    if (x_num_col_dims == 1) {
      if (input_dims == 4) {
        PADDLE_ENFORCE_EQ(
            input_d[3], 1,
            platform::errors::InvalidArgument(
                "Invalid dimensions. When x_num_col_dims equals to 1 and input "
                "dims equals to 4, the last dim of input must be 1, but got %d",
                input_d[3]));
      }
      if (enable_int8) {
        reshape_dim3[0] = 1;
        for (int i = 0; i < 3; i++) {
          reshape_dim3[0] *= input_d[i];
          if (i > 0) {
            reshape_dim3[i] = 1;
          }
        }
      } else {
        for (int i = 0; i < 3; i++) {
          if (i < input_dims) {
            reshape_dim3[i] = input_d[i];
          } else {
            reshape_dim3[i] = 1;
          }
        }
      }

      nvinfer1::Dims3 reshape_dim(reshape_dim3[0], reshape_dim3[1],
                                  reshape_dim3[2]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      reshape_layer->setReshapeDimensions(reshape_dim);
      reshape_itensor = reshape_layer->getOutput(0);
      if (enable_int8) {
        engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
      }
    } else {
      PADDLE_ENFORCE_NE(input_dims, 1,
                        platform::errors::InvalidArgument(
                            "Invalid dimensions. When x_num_col_dims equals to "
                            "2, input_dims should not be 1"));

      if (enable_int8) {
        for (int i = 0; i < 4; i++) {
          if (i == 0) {
            reshape_dim4[i] = input_d[i];
          } else {
            reshape_dim4[i] = 1;
            if (i < input_dims) {
              reshape_dim4[1] *= input_d[i];
            }
          }
        }
      } else {
        for (int i = 0; i < 4; i++) {
          if (i < input_dims) {
            reshape_dim4[i] = input_d[i];
          } else {
            reshape_dim4[i] = 1;
          }
        }
      }
      nvinfer1::Dims4 reshape_dim(reshape_dim4[0], reshape_dim4[1],
                                  reshape_dim4[2], reshape_dim4[3]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      reshape_layer->setReshapeDimensions(reshape_dim);
      reshape_itensor = reshape_layer->getOutput(0);
      if (enable_int8) {
        engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
      }
    }
    regist_fc(reshape_itensor, n_output, weight, bias);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
