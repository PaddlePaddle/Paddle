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

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Cumsum Op
 */
class CumsumOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7220)
    VLOG(3) << "convert a cumsum op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto dims = input_x_tensor->getDimensions();
    auto rank = dims.nbDims;
    int axis = 0;
    if (op_desc.HasAttr("axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
      if (axis < 0) {
        axis += rank;
      }
    }

    // getAxisLength default is a scalar
    auto getAxisLength = [&](nvinfer1::ITensor* inpTensor,
                             int axis,
                             bool scalar = true) {
      auto dims = inpTensor->getDimensions();
      int d = dims.d[axis];
      if (d >= 0) {
        return Add1DConstantLayer(d, "", scalar);
      } else {
        nvinfer1::ITensor* inpShape = Shape(inpTensor);
        auto* tensor = TRT_ENGINE_ADD_LAYER(engine_,
                                            Gather,
                                            *inpShape,
                                            *Add1DConstantLayer(d, " ", true),
                                            0)
                           ->getOutput(0);

        return tensor;
      }
    };

    // Scan through each slice across axis and add it to the running sum
    auto loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
    nvinfer1::ITensor* tripLimit = getAxisLength(input_x_tensor, axis);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
    auto iterator = loop->addIterator(*input_x_tensor, axis);
    auto data = iterator->getOutput(0);

    // get sliced shape
    std::vector<nvinfer1::ITensor*> concat_shape;
    auto input_x_shape = Shape(input_x_tensor);
    for (int i = 0; i < rank; i++) {
      if (i != axis) {
        concat_shape.push_back(GetEleTensorOfShape(input_x_shape, i));
      }
    }
    std::string name = "_cumsum_op_";
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
    std::vector<float> value_vec(1, 0);
    std::vector<float> beta_vec(rank - 1, 0.);
    layer->setAlpha(0);
    layer->setBeta(0.f);
    layer->setInput(0, *Concat(concat_shape));
    auto* inputSliced_output = layer->getOutput(0);
    layer->setInput(1, *Add1DConstantLayer(value_vec, name + "alpha", true));
    layer->setInput(2, *Add1DConstantLayer(beta_vec, name + "beta", false));

    // creat ZeroTensor
    std::vector<float> zero_vec{0.f};
    auto zero = Add1DConstantLayer(zero_vec);

    zero = TRT_ENGINE_ADD_LAYER(engine_,
                                ElementWise,
                                *inputSliced_output,
                                *BroadcastTensors(zero, inputSliced_output),
                                nvinfer1::ElementWiseOperation::kPROD)
               ->getOutput(0);

    auto cast = TRT_ENGINE_ADD_LAYER(engine_, Identity, *zero);
    cast->setOutputType(0, input_x_tensor->getType());

    auto runningSum = loop->addRecurrence(*cast->getOutput(0));
    auto runningSumTensor = runningSum->getOutput(0);
    auto curSum = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *data,
                                       *runningSumTensor,
                                       nvinfer1::ElementWiseOperation::kSUM);
    runningSum->setInput(1, *curSum->getOutput(0));
    auto reverseFlag = nvinfer1::LoopOutput::kCONCATENATE;
    nvinfer1::ILoopOutputLayer* loopOut =
        loop->addLoopOutput(*curSum->getOutput(0), reverseFlag, axis);
    loopOut->setInput(1, *tripLimit);
    RreplenishLayerAndOutput(loopOut, "cumsum", {output_name}, test_mode);
#else
    VLOG(3) << "Cumsum is not supported when TensorRT < 7.2.2";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(cumsum, CumsumOpConverter);
