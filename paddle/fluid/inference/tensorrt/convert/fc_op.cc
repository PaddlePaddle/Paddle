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

// Reorder the elements from istrides to ostrides, borrowed from TRT convert in
// tensorflow.
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensorrt/convert/convert_nodes.cc#L318
template <typename T>
void Reorder2(nvinfer1::DimsHW shape, const T* idata, nvinfer1::DimsHW istrides,
              T* odata, nvinfer1::DimsHW ostrides) {
  for (int h = 0; h < shape.h(); ++h) {
    for (int w = 0; w < shape.w(); ++w) {
      odata[h * ostrides.h() + w * ostrides.w()] =
          idata[h * istrides.h() + w * istrides.w()];
    }
  }
}
// indata c * k
// Reorder the data layout from CK to KC.
void ReorderCKtoKC(TensorRTEngine::Weight& iweights,  // NOLINT
                   TensorRTEngine::Weight* oweights) {
  int c = iweights.dims[0];
  int k = iweights.dims[1];
  oweights->dims.assign({k, c});
  nvinfer1::DimsHW istrides = {1, k};
  nvinfer1::DimsHW ostrides = {c, 1};
  Reorder2({k, c}, static_cast<float const*>(iweights.get().values), istrides,
           static_cast<float*>(const_cast<void*>(oweights->get().values)),
           ostrides);
}
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
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    const int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? boost::get<int>(op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? boost::get<int>(op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? boost::get<std::string>(op_desc.GetAttr("activation_type"))
            : "";
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, which can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = boost::get<bool>(op_desc.HasAttr("enable_int8"));
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      CHECK(op_desc.HasAttr(i_name + "_scale"));
      float in_scale = boost::get<float>(op_desc.GetAttr(i_name + "_scale"));
      auto weight_scale =
          boost::get<std::vector<float>>(op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(),
                                              Y_t, true, weight_scale);
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL);  // a matrix
    size_t n_output = Y_t->dims()[1];

    std::unique_ptr<framework::Tensor> tmp(new framework::LoDTensor());
    tmp->Resize(Y_t->dims());

    memcpy(tmp->mutable_data<float>(platform::CPUPlace()), weight_data,
           Y_t->dims()[0] * Y_t->dims()[1] * sizeof(float));
    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    TensorRTEngine::Weight tmp_weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(tmp->data<float>()),
                                      static_cast<size_t>(Y_t->numel()));
    weight.dims.assign({Y_t->dims()[0], Y_t->dims()[1]});
    tmp_weight.dims = weight.dims;

    // The data layout of TRT FC layer's weight is different from fluid's FC,
    // need to reorder the elements.
    ReorderCKtoKC(weight, &tmp_weight);

    // Currently, the framework can only handle one fluid op -> one TRT layer,
    // but fc fuses `mul` and `bias` (2 fluid ops), so here is a trick, just
    // handle `mul`, leave `add` as another layer.
    // DEBUG
    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.FindVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data =
          engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t, false);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    // in order to handle situations in NLP models(input dims < 3,
    // x_num_col_dims != 1, etc.), reshape input to perform FC correctly.
    auto* reshape_itensor = X;
    int input_dims = X->getDimensions().nbDims;
    auto input_d = X->getDimensions().d;
    int reshape_dim3[3] = {0};
    int reshape_dim4[4] = {0};
    PADDLE_ENFORCE_EQ(
        x_num_col_dims == 1 || x_num_col_dims == 2, true,
        platform::errors::InvalidArgument(
            "Wrong x_num_col_dims param of op mul. Paddle-TRT FC converter "
            "expects x_num_col_dims is either 1 or 2, but got %d",
            x_num_col_dims));
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
      for (int i = 0; i < 3; i++) {
        if (i < input_dims) {
          reshape_dim3[i] = input_d[i];
        } else {
          reshape_dim3[i] = 1;
        }
      }
      nvinfer1::Dims3 reshape_dim(reshape_dim3[0], reshape_dim3[1],
                                  reshape_dim3[2]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      reshape_layer->setReshapeDimensions(reshape_dim);
      reshape_itensor = reshape_layer->getOutput(0);
    } else {
      PADDLE_ENFORCE_NE(input_dims, 1,
                        platform::errors::InvalidArgument(
                            "Invalid dimensions. When x_num_col_dims equals to "
                            "2, input_dims should not be 1"));
      for (int i = 0; i < 4; i++) {
        if (i < input_dims) {
          reshape_dim4[i] = input_d[i];
        } else {
          reshape_dim4[i] = 1;
        }
      }
      nvinfer1::Dims4 reshape_dim(reshape_dim4[0], reshape_dim4[1],
                                  reshape_dim4[2], reshape_dim4[3]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      reshape_layer->setReshapeDimensions(reshape_dim);
      reshape_itensor = reshape_layer->getOutput(0);
    }
    auto* fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *reshape_itensor,
                             n_output, tmp_weight.get(), bias.get());

    engine_->SetWeights(op_desc.Input(w_name).front(), std::move(tmp));
    auto output_name = op_desc.Output("Out").front();
    if (activation_type == "relu") {
      nvinfer1::IActivationLayer* relu_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Activation, *(fc_layer->getOutput(0)),
                               nvinfer1::ActivationType::kRELU);
      RreplenishLayerAndOutput(relu_layer, "fc", {output_name}, test_mode);
    } else {
      RreplenishLayerAndOutput(fc_layer, "fc", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
