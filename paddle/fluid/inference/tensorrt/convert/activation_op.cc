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

class ActivationOpConverter : public OpConverter {
 public:
  ActivationOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3)
        << "convert a fluid Activation op to tensorrt activation layer whose "
           "type is "
        << op_type_;
    const nvinfer1::ITensor* input_tensor =
        engine_->GetITensor(op_desc.Input("X")[0]);

    auto op_pair = ops.find(op_type_);

    nvinfer1::IActivationLayer* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *const_cast<nvinfer1::ITensor*>(input_tensor),
        op_pair->second);

#if IS_TRT_VERSION_GE(5130)
    // max(alpha, min(beta, x))
    if (op_type_ == "relu6") {
      layer->setAlpha(0.);
      layer->setBeta(6.);
    }
#endif

    auto output_name = op_desc.Output("Out")[0];

    RreplenishLayerAndOutput(layer, op_type_, {output_name}, test_mode);
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
#endif
};

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

class Relu6OpConverter : public ActivationOpConverter {
 public:
  Relu6OpConverter() { op_type_ = "relu6"; }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(relu, ReluOpConverter);
REGISTER_TRT_OP_CONVERTER(sigmoid, SigmoidOpConverter);
REGISTER_TRT_OP_CONVERTER(tanh, TanhOpConverter);
REGISTER_TRT_OP_CONVERTER(relu6, Relu6OpConverter);
