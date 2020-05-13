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

template <typename RegistFunc, typename SetDilationFunc>
void ConvertConv2d(TensorRTEngine* engine, const framework::proto::OpDesc& op,
                   const framework::Scope& scope, bool test_mode,
                   RegistFunc fadd_layer, SetDilationFunc fset_dilation,
                   const std::string& name) {
  VLOG(3) << "convert a fluid " << name << " op to tensorrt layer without bias";

  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1UL,
                    platform::errors::InvalidArgument(
                        "TRT Conv2d expect 1 input, but got %d input.",
                        op_desc.Input("Input").size()));
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(), 1UL,
                    platform::errors::InvalidArgument(
                        "TRT Conv2d expect 1 filter, but got %d filter.",
                        op_desc.Input("Filter").size()));
  PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(), 1UL,
                    platform::errors::InvalidArgument(
                        "TRT Conv2d expect 1 output, but got %d output.",
                        op_desc.Output("Output").size()));

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
    CHECK(op_desc.HasAttr("Input_scale"));
    float in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Input_scale"));
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
  const std::vector<int> paddings =
      BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));

  nvinfer1::DimsHW nv_ksize(filter_h, filter_w);
  nvinfer1::DimsHW nv_dilations(dilations[0], dilations[1]);
  nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
  nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

  TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(weight_data),
                                static_cast<size_t>(Y_t->numel())};

  TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  auto* layer = fadd_layer(const_cast<nvinfer1::ITensor*>(X), n_output, n_input,
                           nv_ksize, weight, bias);
  PADDLE_ENFORCE(layer != nullptr);
  layer->setStride(nv_strides);
  layer->setPadding(nv_paddings);
  layer->setNbGroups(groups);
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
            int n_input,                             /* Conv input maps */
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
            int n_input,                             /* Deconv output maps */
            nvinfer1::DimsHW& ksize, TensorRTEngine::Weight& weight,
            TensorRTEngine::Weight& bias) -> nvinfer1::IDeconvolutionLayer* {
          auto* layer =
              TRT_ENGINE_ADD_LAYER(engine_, Deconvolution, *inputs, n_input,
                                   ksize, weight.get(), bias.get());
          return layer;
        },
        [](nvinfer1::IDeconvolutionLayer* layer, nvinfer1::DimsHW& dilations) {
          // In trt Deconv, dilation should be 1, ohter values are not
          // supported.
          bool condition = (dilations.d[0] == 1 && dilations.d[1] == 1);
          PADDLE_ENFORCE_EQ(condition, true,
                            platform::errors::InvalidArgument(
                                "In Deconv, Dilations must be (1, 1) for "
                                "tensorRT, but given (%d, %d)",
                                dilations.d[0], dilations.d[1]));
        },
        "conv2d_transpose");
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);
REGISTER_TRT_OP_CONVERTER(conv2d_transpose, Deconv2dOpConverter);
