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

class FlipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a flip op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();

    // Get Attrs
    std::vector<int> axis =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"));
    for (size_t i = 0; i < axis.size(); ++i) {
      axis[i] += (axis[i] < 0) ? input_dims.nbDims : 0;
    }

    nvinfer1::ITensor* shape_tensor = Shape(input);
    // getAxisLength default is a scalar
    auto getAxisLength = [&](int axis, bool scalar = true) {
      int d = input_dims.d[axis];
      if (d >= 0) {
        return Add1DConstantLayer(d, "", scalar);
      } else {
        return GetEleTensorOfShape(shape_tensor, axis, scalar);
      }
    };
    for (size_t i = 0; i < axis.size(); ++i) {
      auto loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
      nvinfer1::ITensor* tripLimit = getAxisLength(axis[i]);
      loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
      auto iterator = loop->addIterator(*input, axis[i], true);
      std::vector<int32_t> zero_vec{0};
      std::vector<int32_t> one_vec{1};
      auto zero = Add1DConstantLayer(zero_vec);
      auto one = Add1DConstantLayer(one_vec);
      nvinfer1::IRecurrenceLayer* iRec = loop->addRecurrence(*zero);
      nvinfer1::ITensor* iCur = iRec->getOutput(0);
      auto iNext = TRT_ENGINE_ADD_LAYER(engine_,
                                        ElementWise,
                                        *iCur,
                                        *one,
                                        nvinfer1::ElementWiseOperation::kSUM);
      iRec->setInput(1, *iNext->getOutput(0));
      nvinfer1::ILoopOutputLayer* loopOut = loop->addLoopOutput(
          *iterator->getOutput(0), nvinfer1::LoopOutput::kCONCATENATE, axis[i]);
      loopOut->setInput(1, *tripLimit);
      input = loopOut->getOutput(0);
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input);
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "flip", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(flip, FlipOpConverter);
