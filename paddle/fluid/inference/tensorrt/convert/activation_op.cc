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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

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
    if (op_pair == ops.end()) {
      PADDLE_THROW("Wrong activation op type!");
    }

    nvinfer1::IActivationLayer* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *const_cast<nvinfer1::ITensor*>(input_tensor),
        op_pair->second);
    auto output_name = op_desc.Output("Out")[0];
    layer->setName((op_type_ + " (Output: " + output_name + ")").c_str());
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
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

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(relu, ReluOpConverter);
REGISTER_TRT_OP_CONVERTER(sigmoid, SigmoidOpConverter);
REGISTER_TRT_OP_CONVERTER(tanh, TanhOpConverter);
