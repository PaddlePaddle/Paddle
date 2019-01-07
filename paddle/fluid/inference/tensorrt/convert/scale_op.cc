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

/*
 * ScaleOp.
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid scale op to tensorrt scale layer";

    nvinfer1::ILayer* layer = nullptr;
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    bool need_to_expand_dims = (input_dims.nbDims == 2);

    if (need_to_expand_dims) {
      nvinfer1::DimsCHW reshape_dims(1, input_dims.d[0], input_dims.d[1]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      reshape_layer->setReshapeDimensions(reshape_dims);
      layer = reshape_layer;
    }

    float scale_v = boost::get<float>(op_desc.GetAttr("scale"));
    float bias_v = boost::get<float>(op_desc.GetAttr("bias"));
    bool bias_after_scale =
        boost::get<bool>(op_desc.GetAttr("bias_after_scale"));

    platform::CPUPlace place;
    std::unique_ptr<framework::LoDTensor> scale_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> bias_tensor(
        new framework::LoDTensor());
    scale_tensor->Resize(framework::make_ddim({1}));
    bias_tensor->Resize(framework::make_ddim({1}));

    float* scale_data = scale_tensor->mutable_data<float>(place);
    float* bias_data = bias_tensor->mutable_data<float>(place);
    scale_data[0] = scale_v;
    bias_data[0] = bias_v;

    TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(scale_data), 1};
    TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(bias_data), 1};
    TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    if (bias_after_scale) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Scale, *layer->getOutput(0),
                                   nvinfer1::ScaleMode::kUNIFORM, shift.get(),
                                   scale.get(), power.get());
    } else {
      TensorRTEngine::Weight temp_scale{nvinfer1::DataType::kFLOAT, nullptr, 0};
      TensorRTEngine::Weight temp_shift{nvinfer1::DataType::kFLOAT, nullptr, 0};

      layer = TRT_ENGINE_ADD_LAYER(engine_, Scale, *layer->getOutput(0),
                                   nvinfer1::ScaleMode::kUNIFORM, shift.get(),
                                   temp_scale.get(), power.get());

      layer = TRT_ENGINE_ADD_LAYER(engine_, Scale, *layer->getOutput(0),
                                   nvinfer1::ScaleMode::kUNIFORM,
                                   temp_shift.get(), scale.get(), power.get());
    }

    if (need_to_expand_dims) {
      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
      reshape_layer->setReshapeDimensions(input_dims);
      layer = reshape_layer;
    }

    PADDLE_ENFORCE(layer != nullptr);
    auto output_name = op_desc.Output("Out")[0];
    engine_->weight_map["scale_op_scale" + output_name] =
        std::move(scale_tensor);
    engine_->weight_map["scale_op_bias" + output_name] = std::move(bias_tensor);
    engine_->SetITensor(output_name, layer->getOutput(0));
    layer->setName(("scale (Output: " + output_name + ")").c_str());
    layer->getOutput(0)->setName(output_name.c_str());
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
