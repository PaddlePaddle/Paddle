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
namespace {
template <typename T>
void tranpose_weight(const T* src, T* dst, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[j * m + i] = src[i * n + j];
    }
  }
}
}  // namespace

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  nvinfer1::ILayer* reshape_before_fc(nvinfer1::ITensor* before_fc,
                                      nvinfer1::Dims x_dim,
                                      int x_num_col_dims,
                                      std::string output_name) {
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = x_num_col_dims + 3;
    // padding shape "* x q x 1 x 1"

    nvinfer1::ITensor* filal_reshape_before_fc_shape_tensor = nullptr;

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_dim.d[i] = 1;
      }
      for (int i = 0; i < x_dim.nbDims; i++) {
        if (i < x_num_col_dims) {
          reshape_before_fc_dim.d[i] = 0;
        } else {
          reshape_before_fc_dim.d[x_num_col_dims] *= x_dim.d[i];
        }
      }
    } else {
      std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
      nvinfer1::ITensor* input_shape_tensor = Shape(before_fc);

      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
      }
      for (int i = 0; i < x_dim.nbDims; i++) {
        if (i < x_num_col_dims) {
          reshape_before_fc_shape_tensor[i] =
              GetEleTensorOfShape(input_shape_tensor, i);
        } else {
          reshape_before_fc_shape_tensor[x_num_col_dims] =
              Prod(GetEleTensorOfShape(input_shape_tensor, i),
                   reshape_before_fc_shape_tensor[x_num_col_dims]);
        }
      }
      filal_reshape_before_fc_shape_tensor =
          Concat(reshape_before_fc_shape_tensor);
    }

    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *before_fc);
    if (!engine_->with_dynamic_shape()) {
      reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    } else {
      reshape_before_fc_layer->setInput(1,
                                        *filal_reshape_before_fc_shape_tensor);
    }

    reshape_before_fc_layer->setName(
        ("fc_op_reshape_before_fc: Shuffle (Output: " + output_name + ")")
            .c_str());
    return reshape_before_fc_layer;
  }

  nvinfer1::ILayer* reshape_after_fc(nvinfer1::ITensor* after_fc,
                                     nvinfer1::Dims x_dim,
                                     int x_num_col_dims) {
    // add shuffle after fc
    nvinfer1::Dims reshape_after_fc_dim;
    reshape_after_fc_dim.nbDims = x_num_col_dims + 1;

    nvinfer1::ITensor* filal_reshape_after_fc_shape_tensor = nullptr;

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
        reshape_after_fc_dim.d[i] = 0;
      }
    } else {
      std::vector<int> gather_indices(x_num_col_dims + 1);
      std::iota(gather_indices.begin(), gather_indices.end(), 0);
      filal_reshape_after_fc_shape_tensor =
          Gather(Shape(after_fc), gather_indices);
    }

    auto* reshape_after_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *after_fc);
    if (!engine_->with_dynamic_shape()) {
      reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
    } else {
      reshape_after_fc_layer->setInput(1, *filal_reshape_after_fc_shape_tensor);
    }

    return reshape_after_fc_layer;
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
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
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    auto x_dim = X->getDimensions();
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v,
        platform::errors::NotFound(
            "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<phi::DenseTensor>();
    int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? PADDLE_GET_CONST(int, op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? PADDLE_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";

    bool enable_int8 = op_desc.HasAttr("enable_int8");
    bool support_int8 = false;
    if (op_desc.HasAttr("support_int8")) {
      support_int8 = PADDLE_GET_CONST(bool, op_desc.GetAttr("support_int8"));
    }
    float in_scale = 0;
    if (enable_int8 || support_int8) {
      if (enable_int8) {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      } else {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("X"));
      }
      engine_->SetTensorDynamicRange(X, in_scale);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(),
                      2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];

    auto regist_fc = [&](nvinfer1::ITensor* inputs,
                         int n_output,
                         TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias) {
      if (enable_int8 || support_int8) {
        // add conv layer
        float out_scale = 0;
        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        } else {
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
        }
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 = TRT_ENGINE_ADD_LAYER(engine_,
                                                   Convolution,
                                                   *inputs,
                                                   n_output,
                                                   nv_ksize,
                                                   weight.get(),
                                                   bias.get());
        fc_layer_int8->setName(
            ("fc_op_int8_conv1x1: Convolution (Output: " + output_name + ")")
                .c_str());
        engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0), out_scale);
        auto* fc_after_reshape_int8 = reshape_after_fc(
            fc_layer_int8->getOutput(0), x_dim, x_num_col_dims);
        if (activation_type == "relu") {
          fc_after_reshape_int8->setName(
              ("int8_reshape_after_fc: Shuffle (Output: " + output_name + ")")
                  .c_str());
          engine_->SetTensorDynamicRange(fc_after_reshape_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_after_reshape_int8->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8,
                                   "relu_after_fc_shuffle",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_after_reshape_int8,
                                   "fc_op_int8_reshape_after_fc: Shuffle",
                                   {output_name},
                                   test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(engine_,
                                                    FullyConnected,
                                                    *inputs,
                                                    n_output,
                                                    weight.get(),
                                                    bias.get());
        fc_layer_float->setName(
            ("fc_op_float: FullyConnected (Output: " + output_name + ")")
                .c_str());
        auto* fc_after_reshape_float = reshape_after_fc(
            fc_layer_float->getOutput(0), x_dim, x_num_col_dims);
        if (activation_type == "relu") {
          fc_after_reshape_float->setName(
              ("float_reshape_after_fc: Shuffle (Output: " + output_name + ")")
                  .c_str());
          nvinfer1::IActivationLayer* relu_layer_float =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_after_reshape_float->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float,
                                   "relu_after_fc_shuffle",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_after_reshape_float,
                                   "shuffle_after_fc",
                                   {output_name},
                                   test_mode);
        }
      }
    };

    bool transpose_y = false;
    if (op_desc.HasAttr("transpose_Y")) {
      transpose_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    }
    int weight_w, weight_h;
    auto weight = engine_->GetTrtWeight(op_desc.Input(w_name).front(), *Y_t);

    if (!transpose_y) {
      if (weight.get().type == nvinfer1::DataType::kFLOAT) {
        std::vector<float> weight_data_tmp;
        weight_data_tmp.reserve(Y_t->numel());
        memcpy(weight_data_tmp.data(),
               weight.get().values,
               Y_t->numel() * sizeof(float));
        tranpose_weight(
            weight_data_tmp.data(),
            const_cast<float*>(static_cast<const float*>(weight.get().values)),
            m,
            n);
      } else if (weight.get().type == nvinfer1::DataType::kHALF) {
        std::vector<float16> weight_data_tmp;
        weight_data_tmp.reserve(Y_t->numel());
        memcpy(weight_data_tmp.data(),
               weight.get().values,
               Y_t->numel() * sizeof(float16));
        tranpose_weight(weight_data_tmp.data(),
                        const_cast<float16*>(
                            static_cast<const float16*>(weight.get().values)),
                        m,
                        n);
      } else {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Paddle-TRT fc convert not supporte dtype, now only support fp32 "
            "and fp16."));
      }
      weight_w = n;
      weight_h = m;
    } else {
      weight_w = m;
      weight_h = n;
    }
    size_t n_output = weight_w;
    weight.dims.assign({weight_w, weight_h});

    TensorRTEngine::Weight bias{weight.get().type, nullptr, 0};
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<phi::DenseTensor>();
      bias = engine_->GetTrtWeight(op_desc.Input("Bias").front(), *b_t);
    }

    // Running the TRT Static Shape mode: x_num_col_dims-1
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }
    // If use tensorrt'oss, the x_dim and x_num_col_dims need change, and can
    // not add Shuffle layer in ernie's multihead.
    if (x_dim.nbDims == 4 && x_dim.d[2] == 1 && x_dim.d[3] == 1) {
      if (enable_int8 || support_int8) {
        // add conv1x1 layer
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 = TRT_ENGINE_ADD_LAYER(engine_,
                                                   Convolution,
                                                   *X,
                                                   n_output,
                                                   nv_ksize,
                                                   weight.get(),
                                                   bias.get());
        if (activation_type == "relu") {
          fc_layer_int8->setName(
              ("ernie_fc_op_int8: Convolution (Output: " + output_name + ")")
                  .c_str());
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          float out_scale = 0;
          if (enable_int8) {
            out_scale =
                PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
          } else {
            out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
          }
          engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_layer_int8->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8,
                                   "relu_after_ernie_fc_int8",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_int8,
                                   "ernie_fc_op_int8: Convolution",
                                   {output_name},
                                   test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(
            engine_, FullyConnected, *X, n_output, weight.get(), bias.get());
        if (activation_type == "relu") {
          fc_layer_float->setName(
              ("ernie_fc_op_float: (Output: " + output_name + ")").c_str());
          nvinfer1::IActivationLayer* relu_layer_float =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_layer_float->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float,
                                   "relu_after_ernie_fc_float",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(
              fc_layer_float, "ernie_fc_op_float", {output_name}, test_mode);
        }
      }
    } else {  // need reshape input before and after fc
      PADDLE_ENFORCE_GT(
          x_dim.nbDims,
          x_num_col_dims,
          platform::errors::InvalidArgument(
              "Params and input dims mismatch. Paddle-TRT FC "
              "converter expects x_dim.nbDims > x_num_col_dims, but "
              "x_dim.nbDims : %d, x_num_col_dims : %d.",
              x_dim.nbDims,
              x_num_col_dims));
      auto* reshape_before_fc_layer =
          reshape_before_fc(X, x_dim, x_num_col_dims, output_name);
      auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);
      if (enable_int8 || support_int8) {
        engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
      }
      regist_fc(reshape_itensor, n_output, weight, bias);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
