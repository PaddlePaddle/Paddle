/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle::inference::tensorrt {

/*
 * GridSampler Op
 */
class GridSamplerOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8510)
    VLOG(3) << "convert a grid_sampler op to tensorrt GridSample layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    std::string input_grid_name = op_desc.Input("Grid").front();
    std::string output_name = op_desc.Output("Output").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto* input_grid_tensor = engine_->GetITensor(input_grid_name);

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, GridSample, *input_x_tensor, *input_grid_tensor);

    const std::string mode =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    const std::string padding_mode =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("padding_mode"));
    const bool align_corners =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("align_corners"));

    nvinfer1::InterpolationMode interpolationMode{
        nvinfer1::InterpolationMode::kNEAREST};
    if (mode == "nearest") {
#if IS_TRT_VERSION_GE(8600)
      interpolationMode = nvinfer1::InterpolationMode::kNEAREST;
#else
      interpolationMode = nvinfer1::ResizeMode::kNEAREST;
#endif
    } else if (mode == "bilinear") {
#if IS_TRT_VERSION_GE(8600)
      interpolationMode = nvinfer1::InterpolationMode::kLINEAR;
#else
      interpolationMode = nvinfer1::ResizeMode::kLINEAR;
#endif
    }

    nvinfer1::SampleMode sampleMode{nvinfer1::SampleMode::kFILL};
    if (padding_mode == "zeros") {
      sampleMode = nvinfer1::SampleMode::kFILL;
    } else if (padding_mode == "border") {
      sampleMode = nvinfer1::SampleMode::kCLAMP;
    } else if (padding_mode == "reflection") {
      sampleMode = nvinfer1::SampleMode::kREFLECT;
    }

    layer->setInterpolationMode(interpolationMode);
    layer->setSampleMode(sampleMode);
    layer->setAlignCorners(align_corners);

    ReplenishLayerAndOutput(layer, "grid_sampler", {output_name}, test_mode);
#else
    VLOG(3) << "grid_sampler is not supported when TensorRT < 8.5.1";
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(grid_sampler, GridSamplerOpConverter);
