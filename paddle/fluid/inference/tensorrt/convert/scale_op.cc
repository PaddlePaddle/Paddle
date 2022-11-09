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

/*
 * ConcatOp
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid scale op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    std::string input_name = op_desc.Input("X").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    bool bias_after_scale =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("bias_after_scale"));
    float bias = PADDLE_GET_CONST(float, op_desc.GetAttr("bias"));
    float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("scale"));
    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      nvinfer1::ITensor* bias_tensor = Add1DConstantLayer(bias);
      bool is_bias_0 = (bias < 1e-06 && bias > -1e-06);

      std::vector<int32_t> bias_shapes(input->getDimensions().nbDims, 1);
      auto* bias_shapes_tensor = Add1DConstantLayer(bias_shapes);
      auto* reshape_layer_bias =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *bias_tensor);
      reshape_layer_bias->setInput(1, *bias_shapes_tensor);

      bool has_scale_tensor;
      nvinfer1::ITensor* scale_tensor;
      bool is_scale_1;

      auto scale_inputs = op_desc.Inputs();
      if (scale_inputs.find("ScaleTensor") != scale_inputs.end() &&
          op_desc.Input("ScaleTensor").size()) {  // has EndsTensor input
        has_scale_tensor = true;
        scale_tensor = engine_->GetITensor(op_desc.Input("ScaleTensor")[0]);
        is_scale_1 = false;
      } else {
        has_scale_tensor = false;
        scale_tensor = Add1DConstantLayer(scale);
        is_scale_1 = ((scale - 1.0) < 1e-06 && (scale - 1.0) > -1e-06);
      }

      std::vector<int32_t> scale_shapes(input->getDimensions().nbDims, 1);
      auto* scale_shapes_tensor = Add1DConstantLayer(scale_shapes);
      auto* reshape_layer_scale =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *scale_tensor);
      reshape_layer_scale->setInput(1, *scale_shapes_tensor);

      if (!has_scale_tensor && is_scale_1 && is_bias_0) {
        layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input);
      } else {
        if (bias_after_scale) {
          if (!is_scale_1) {
            layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *input,
                                         *reshape_layer_scale->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kPROD);
            input = layer->getOutput(0);
          }
          if (!is_bias_0) {
            layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *input,
                                         *reshape_layer_bias->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kSUM);
          }
        } else {
          if (!is_bias_0) {
            layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *input,
                                         *reshape_layer_bias->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kSUM);
            input = layer->getOutput(0);
          }
          if (!is_scale_1) {
            layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *input,
                                         *reshape_layer_scale->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kPROD);
          }
        }
      }
    } else {
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_scale_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };

      float* bias_ptr = create_weights(bias, "bias");
      float* scale_ptr = create_weights(scale, "scale");

      TensorRTEngine::Weight scale_weights{
          nvinfer1::DataType::kFLOAT, static_cast<void*>(scale_ptr), 1};
      TensorRTEngine::Weight shift_weights{
          nvinfer1::DataType::kFLOAT, static_cast<void*>(bias_ptr), 1};
      TensorRTEngine::Weight power_weights{
          nvinfer1::DataType::kFLOAT, nullptr, 0};

      auto input_dim = input->getDimensions();

      nvinfer1::IShuffleLayer* expand_layer = nullptr;
      nvinfer1::IShuffleLayer* squeeze_layer = nullptr;

      if (input_dim.nbDims < 3) {
        nvinfer1::Dims expand_shape;
        expand_shape.nbDims = 3;
        for (int i = 0; i < 3; i++) {
          if (i < input_dim.nbDims) {
            expand_shape.d[i] = input_dim.d[i] < 0 ? 0 : input_dim.d[i];
          } else {
            expand_shape.d[i] = 1;
          }
        }
        expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        expand_layer->setReshapeDimensions(expand_shape);
        input = expand_layer->getOutput(0);
        expand_layer->getOutput(0)->setName(
            ("before_reshape_out: " + out_name).c_str());
        expand_layer->setName(
            ("Scale: before_reshape (Output: " + out_name + ")").c_str());
      }

      if (bias_after_scale) {
        layer = TRT_ENGINE_ADD_LAYER(engine_,
                                     Scale,
                                     *input,
                                     nvinfer1::ScaleMode::kUNIFORM,
                                     shift_weights.get(),
                                     scale_weights.get(),
                                     power_weights.get());
        layer->getOutput(0)->setName(
            ("bias_after_scale_out: " + out_name).c_str());
        layer->setName(("Scale: scale (Output: " + out_name + ")").c_str());
      } else {
        // add bias
        layer = TRT_ENGINE_ADD_LAYER(engine_,
                                     Scale,
                                     *(input),
                                     nvinfer1::ScaleMode::kUNIFORM,
                                     shift_weights.get(),
                                     power_weights.get(),
                                     power_weights.get());
        layer->getOutput(0)->setName(
            ("bias_before_scale：bias_out: " + out_name).c_str());
        layer->setName(
            ("Scale: scale_bias (Output: " + out_name + ")").c_str());
        // mul scale
        layer = TRT_ENGINE_ADD_LAYER(engine_,
                                     Scale,
                                     *(layer->getOutput(0)),
                                     nvinfer1::ScaleMode::kUNIFORM,
                                     power_weights.get(),
                                     scale_weights.get(),
                                     power_weights.get());
        layer->getOutput(0)->setName(
            ("bias_before_scale：scale_out: " + out_name).c_str());
        layer->setName(
            ("Scale: scale_scale (Output: " + out_name + ")").c_str());
      }

      PADDLE_ENFORCE_EQ(layer != nullptr,
                        true,
                        platform::errors::Fatal("Create scale layer failed."));

      if (input_dim.nbDims < 3) {
        nvinfer1::Dims squeeze_shape;
        squeeze_shape.nbDims = input_dim.nbDims;
        for (int i = 0; i < squeeze_shape.nbDims; i++) {
          squeeze_shape.d[i] = input_dim.d[i] < 0 ? 0 : input_dim.d[i];
        }
        squeeze_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
        squeeze_layer->setReshapeDimensions(squeeze_shape);
        layer = static_cast<nvinfer1::ILayer*>(squeeze_layer);
        layer->getOutput(0)->setName(
            ("after_reshape_out: " + out_name).c_str());
        layer->setName(
            ("Scale: Shuffle_reshape (Output: " + out_name + ")").c_str());
      }
    }
    RreplenishLayerAndOutput(layer, "scale", {out_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
