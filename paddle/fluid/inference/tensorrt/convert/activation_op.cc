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

#include <NvInfer.h>

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::inference::tensorrt {

class ActivationOpConverter : public OpConverter {
 public:
  ActivationOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert a Activation op to tensorrt activation layer whose "
               "type is "
            << op_type_;
    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);

    auto op_pair = ops.find(op_type_);
    nvinfer1::IActivationLayer* layer = nullptr;
    if (op_type_ == "softplus") {
      const float beta = op_desc.HasAttr("beta")
                             ? PADDLE_GET_CONST(float, op_desc.GetAttr("beta"))
                             : 1.0f;
      const float threshold =
          op_desc.HasAttr("threshold")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("threshold"))
              : 20.0f;
      auto* layer_clip = TRT_ENGINE_ADD_LAYER(
          engine_, Activation, *input_tensor, nvinfer1::ActivationType::kCLIP);
      layer_clip->setAlpha(-3.40282e+038);
      layer_clip->setBeta(threshold / beta);
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Activation, *layer_clip->getOutput(0), op_pair->second);
      layer->setAlpha(1.0f / beta);
      layer->setBeta(beta);
    } else {
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Activation, *input_tensor, op_pair->second);
    }

#if IS_TRT_VERSION_GE(5130)
    // max(alpha, min(beta, x))
    if (op_type_ == "relu6") {
      layer->setAlpha(0.);
      layer->setBeta(6.);
    }
    if (op_type_ == "elu") {
      const float alpha =
          op_desc.HasAttr("alpha")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"))
              : 1.0f;
      layer->setAlpha(alpha);
    }
    if (op_type_ == "selu") {
      const float alpha =
          op_desc.HasAttr("alpha")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"))
              : 1.0507009873554804934193349852946;
      const float scale =
          op_desc.HasAttr("scale")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("scale"))
              : 1.6732632423543772848170429916717;
      layer->setAlpha(alpha);
      layer->setBeta(scale);
    }
    if (op_type_ == "stanh") {
      const float scale_a =
          op_desc.HasAttr("scale_a")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("scale_a"))
              : 0.67f;
      const float scale_b =
          op_desc.HasAttr("scale_b")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("scale_b"))
              : 1.7159f;
      layer->setAlpha(scale_b);
      layer->setBeta(scale_a);
    }
    if (op_type_ == "thresholded_relu") {
      const float threshold =
          op_desc.HasAttr("threshold")
              ? PADDLE_GET_CONST(float, op_desc.GetAttr("threshold"))
              : 1.0f;
      layer->setAlpha(threshold);
    }
#endif

    auto output_name = op_desc.Output("Out")[0];

    ReplenishLayerAndOutput(layer, op_type_, {output_name}, test_mode);
  }

 protected:
  std::string op_type_;
  static const std::unordered_map<std::string, nvinfer1::ActivationType> ops;
};

const std::unordered_map<std::string, nvinfer1::ActivationType>
    ActivationOpConverter::ops = {
        {"relu", nvinfer1::ActivationType::kRELU},
        {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
        {"tanh", nvinfer1::ActivationType::kTANH},
#if IS_TRT_VERSION_GE(5130)
        {"relu6", nvinfer1::ActivationType::kCLIP},
        {"elu", nvinfer1::ActivationType::kELU},
        {"selu", nvinfer1::ActivationType::kSELU},
        {"softsign", nvinfer1::ActivationType::kSOFTSIGN},
        {"softplus", nvinfer1::ActivationType::kSOFTPLUS},
        {"stanh", nvinfer1::ActivationType::kSCALED_TANH},
        {"thresholded_relu", nvinfer1::ActivationType::kTHRESHOLDED_RELU}};
#endif

class ReluOpConverter : public ActivationOpConverter {
 public:
  ReluOpConverter() { op_type_ = "relu"; }
};

class SigmoidOpConverter : public ActivationOpConverter {
 public:
  SigmoidOpConverter() { op_type_ = "sigmoid"; }
};

class TanhOpConverter : public ActivationOpConverter {
 public:
  TanhOpConverter() { op_type_ = "tanh"; }
};

#if IS_TRT_VERSION_GE(5130)
class Relu6OpConverter : public ActivationOpConverter {
 public:
  Relu6OpConverter() { op_type_ = "relu6"; }
};

class EluOpConverter : public ActivationOpConverter {
 public:
  EluOpConverter() { op_type_ = "elu"; }
};

class SeluOpConverter : public ActivationOpConverter {
 public:
  SeluOpConverter() { op_type_ = "selu"; }
};

class SoftsignOpConverter : public ActivationOpConverter {
 public:
  SoftsignOpConverter() { op_type_ = "softsign"; }
};

class SoftplusOpConverter : public ActivationOpConverter {
 public:
  SoftplusOpConverter() { op_type_ = "softplus"; }
};

class STanhOpConverter : public ActivationOpConverter {
 public:
  STanhOpConverter() { op_type_ = "stanh"; }
};

class ThresholdedReluOpConverter : public ActivationOpConverter {
 public:
  ThresholdedReluOpConverter() { op_type_ = "thresholded_relu"; }
};
#endif

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(relu, ReluOpConverter);
REGISTER_TRT_OP_CONVERTER(sigmoid, SigmoidOpConverter);
REGISTER_TRT_OP_CONVERTER(tanh, TanhOpConverter);
#if IS_TRT_VERSION_GE(5130)
REGISTER_TRT_OP_CONVERTER(relu6, Relu6OpConverter);
REGISTER_TRT_OP_CONVERTER(elu, EluOpConverter);
REGISTER_TRT_OP_CONVERTER(selu, SeluOpConverter);
REGISTER_TRT_OP_CONVERTER(softsign, SoftsignOpConverter);
REGISTER_TRT_OP_CONVERTER(softplus, SoftplusOpConverter);
REGISTER_TRT_OP_CONVERTER(stanh, STanhOpConverter);
REGISTER_TRT_OP_CONVERTER(thresholded_relu, ThresholdedReluOpConverter);
#endif
