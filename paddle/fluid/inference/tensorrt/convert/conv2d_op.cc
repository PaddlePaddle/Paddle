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
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(), 1);  // Y is a weight
  PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(), 1);

  PADDLE_ENFORCE(engine != nullptr);
  auto* X = engine->GetITensor(op_desc.Input("Input").front());

  // Declare weights
  auto* Y_v = scope.FindVar(op_desc.Input("Filter").front());
  PADDLE_ENFORCE_NOT_NULL(Y_v);
  auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();

  platform::CPUPlace cpu_place;
  std::unique_ptr<framework::LoDTensor> weight_tensor(
      new framework::LoDTensor());
  weight_tensor->Resize(Y_t->dims());
  TensorCopySync((*Y_t), cpu_place, weight_tensor.get());

  auto* weight_data = weight_tensor->mutable_data<float>(cpu_place);

  PADDLE_ENFORCE_EQ(weight_tensor->dims().size(), 4UL);
  const int n_output = weight_tensor->dims()[0];
  const int n_input = weight_tensor->dims()[1];
  const int filter_h = weight_tensor->dims()[2];
  const int filter_w = weight_tensor->dims()[3];
  const int groups = boost::get<int>(op_desc.GetAttr("groups"));
  const std::vector<int> dilations =
      boost::get<std::vector<int>>(op_desc.GetAttr("dilations"));
  const std::vector<int> strides =
      boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
  const std::vector<int> paddings =
      boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));

  nvinfer1::DimsHW nv_ksize(filter_h, filter_w);
  nvinfer1::DimsHW nv_dilations(dilations[0], dilations[1]);
  nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
  nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

  TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(weight_data),
                                static_cast<size_t>(weight_tensor->numel())};

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
  engine->weight_map[op_desc.Input("Filter").front()] =
      std::move(weight_tensor);
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
          PADDLE_ENFORCE(
              dilations.d[0] == 1 && dilations.d[1] == 1,
              "Dilations must be (1, 1) for tensorRT, but given (%d, %d)",
              dilations.d[0], dilations.d[1]);
        },
        "conv2d_transpose");
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);
REGISTER_TRT_OP_CONVERTER(conv2d_transpose, Deconv2dOpConverter);
