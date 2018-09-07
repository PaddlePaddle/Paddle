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

class ReluOpConverter : public OpConverter {
 public:
  ReluOpConverter() {}
  void operator()(const framework::proto::OpDesc& op) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr, nullptr);
    LOG(INFO) << "convert a fluid relu op to tensorrt activation layer whose "
                 "type is Relu";
    const nvinfer1::ITensor* input_tensor =
        engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::IActivationLayer* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *const_cast<nvinfer1::ITensor*>(input_tensor),
        nvinfer1::ActivationType::kRELU);
    engine_->SetITensor(op_desc.Output("Out")[0], layer->getOutput(0));
  }
};

REGISTER_TRT_OP_CONVERTER(relu, ReluOpConverter);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
