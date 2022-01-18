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

template <typename RegistFunc, typename SetDilationFunc>
void ConvertConv2d(TensorRTEngine* engine, const framework::proto::OpDesc& op,
                   const framework::Scope& scope, bool test_mode,
                   RegistFunc fadd_layer, SetDilationFunc fset_dilation,
                   const std::string& name) {
  VLOG(3) << "convert a fluid " << name << " op to tensorrt layer without bias";

  framework::OpDesc op_desc(op, nullptr);

  auto* X = engine->GetITensor(op_desc.Input("Input").front());
  std::string filter_var_name = op_desc.Input("Filter").front();
  auto* Y_v = scope.FindVar(filter_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      Y_v, platform::errors::NotFound(
               "Can not find %s presistale var in scope.", filter_var_name));
  auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
  float* weight_data = nullptr;
  bool enable_int8 = op_desc.HasAttr("enable_int8");

  if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
    float in_scale =
        BOOST_GET_CONST(float, op_desc.GetAttr("Input_scale")) * 127;
    auto weight_scale =
        BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
    weight_data = engine->GetWeightCPUData(op_desc.Input("Filter").front(), Y_t,
                                           true, weight_scale);
    engine->SetTensorDynamicRange(X, in_scale);
#endif
  } else {
    weight_data =
        engine->GetWeightCPUData(op_desc.Input("Filter").front(), Y_t, false);
  }

  PADDLE_ENFORCE_EQ(Y_t->dims().size(), 4UL,
                    platform::errors::InvalidArgument(
                        "The conv2d filter's dims size should be 4, but got %d",
                        Y_t->dims().size()));

  const int n_output = Y_t->dims()[0];
  const int n_input = Y_t->dims()[1];
  const int filter_h = Y_t->dims()[2];
  const int filter_w = Y_t->dims()[3];
  const int groups = BOOST_GET_CONST(int, op_desc.GetAttr("groups"));
  const std::vector<int> dilations =
      BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  const std::vector<int> strides =
      BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
  std::vector<int> paddings =
      BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  std::string padding_algorithm = "EXPLICIT";
  if (op_desc.HasAttr("padding_algorithm"))
    padding_algorithm =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("padding_algorithm"));
  if (padding_algorithm == "VALID") {
    for (size_t i = 0; i < paddings.size(); i++) {
      paddings[i] = 0;
    }
  }

  nvinfer1::DimsHW nv_ksize(filter_h, filter_w);
  nvinfer1::DimsHW nv_dilations(dilations[0], dilations[1]);
  nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
  nvinfer1::DimsHW nv_paddings;
  nvinfer1::Dims nv_pre_paddings;
  nvinfer1::Dims nv_post_paddings;
  if (paddings.size() == 2) {
    nv_paddings.d[0] = paddings[0];
    nv_paddings.d[1] = paddings[1];
  } else {
    nv_pre_paddings.nbDims = 2;
    nv_post_paddings.nbDims = 2;
    nv_pre_paddings.d[0] = paddings[0];
    nv_pre_paddings.d[1] = paddings[2];
    nv_post_paddings.d[0] = paddings[1];
    nv_post_paddings.d[1] = paddings[3];
  }

  TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(weight_data),
                                static_cast<size_t>(Y_t->numel())};
  float* bias_data = nullptr;
  size_t bias_size = 0;
  if (op_desc.Type() == "conv2d_fusion") {
    auto* bias_tensor = scope.GetVar(op_desc.Input("Bias").front());
    auto* bias_tensor_data = bias_tensor->GetMutable<framework::LoDTensor>();
    bias_data = engine->GetWeightCPUData(op_desc.Input("Bias").front(),
                                         bias_tensor_data, false);
    bias_size = static_cast<size_t>(bias_tensor_data->numel());
  }

  TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                              static_cast<void*>(bias_data), bias_size};
  // In conv2d_transpose and depthwise_conv2d_transpose,
  // output channels = filter_dims[1] * groups
  auto* layer = (op_desc.Type() == "conv2d_transpose" ||
                 op_desc.Type() == "depthwise_conv2d_transpose")
                    ? fadd_layer(const_cast<nvinfer1::ITensor*>(X),
                                 n_input * groups, nv_ksize, weight, bias)
                    : fadd_layer(const_cast<nvinfer1::ITensor*>(X), n_output,
                                 nv_ksize, weight, bias);

  PADDLE_ENFORCE_NOT_NULL(
      layer, platform::errors::Fatal("TensorRT create conv2d/conv2d_transpose"
                                     " layer failed."));
  layer->setStride(nv_strides);
  if (paddings.size() == 2) {
    layer->setPadding(nv_paddings);
  } else {
    layer->setPrePadding(nv_pre_paddings);
    layer->setPostPadding(nv_post_paddings);
  }

  layer->setNbGroups(groups);
  if (padding_algorithm == "SAME") {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    nv_dilations.d[0] = 1;
    nv_dilations.d[1] = 1;
  }
  // set dilations
  fset_dilation(layer, nv_dilations);

  auto output_name = op_desc.Output("Output").front();
  layer->setName((name + " (Output: " + output_name + ")").c_str());
  layer->getOutput(0)->setName(output_name.c_str());
  engine->SetITensor(output_name, layer->getOutput(0));

  if (test_mode) {
    engine->DeclareOutput(output_name);
  }
}

class Conv2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    ConvertConv2d(
        engine_, op, scope, test_mode,
        [&](nvinfer1::ITensor* inputs, int n_output, /* Conv output maps */
            nvinfer1::DimsHW& ksize, TensorRTEngine::Weight& weight,
            TensorRTEngine::Weight& bias) -> nvinfer1::IConvolutionLayer* {
          auto* layer =
              TRT_ENGINE_ADD_LAYER(engine_, Convolution, *inputs, n_output,
                                   ksize, weight.get(), bias.get());
          return layer;
        },
        [](nvinfer1::IConvolutionLayer* layer, nvinfer1::DimsHW& dilations) {
          layer->setDilation(dilations);
        },
        "conv2d");
  }
};

class Deconv2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    ConvertConv2d(
        engine_, op, scope, test_mode,
        [&](nvinfer1::ITensor* inputs, int n_output, /* Deconv input maps */
            nvinfer1::DimsHW& ksize, TensorRTEngine::Weight& weight,
            TensorRTEngine::Weight& bias) -> nvinfer1::IDeconvolutionLayer* {
          auto* layer =
              TRT_ENGINE_ADD_LAYER(engine_, Deconvolution, *inputs, n_output,
                                   ksize, weight.get(), bias.get());
          return layer;
        },
        [](nvinfer1::IDeconvolutionLayer* layer, nvinfer1::DimsHW& dilations) {
        },
        "conv2d_transpose");
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);
REGISTER_TRT_OP_CONVERTER(conv2d_fusion, Conv2dOpConverter);
REGISTER_TRT_OP_CONVERTER(conv2d_transpose, Deconv2dOpConverter);
