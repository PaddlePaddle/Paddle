/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "NeonDepthwiseConv.h"
#include "paddle/function/ConvOp.h"

namespace paddle {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

template <DeviceType Device>
class NeonDepthwiseConvFunction : public ConvFunctionBase {
 public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    check(inputs, outputs);

    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();

    int batchSize = input[0];
    int inputChannels = input[1];
    int inputHeight = input[2];
    int inputWidth = input[3];
    int filterHeight = getFilterHeight(filter);
    int filterWidth = getFilterWidth(filter);
    int outputChannels = output[1];
    int outputHeight = output[2];
    int outputWidth = output[3];
    int filterMultiplier = outputChannels / groups_;
    CHECK_EQ(static_cast<size_t>(inputChannels), groups_);

    // only support strideH() == strideW() and filterHeight == filterWidth.
    CHECK_EQ(strideH(), strideW());
    CHECK_EQ(filterHeight, filterWidth);

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();

    // padding the input
    float* inputPadding = inputData;
    int padInputHeight = inputHeight + 2 * paddingH();
    int padInputWidth = inputWidth + 2 * paddingW();
    int newSize =
        batchSize * (inputChannels + 1) * padInputHeight * padInputWidth;

    resizeBuffer<Device>(newSize);
    inputPadding = reinterpret_cast<float*>(memory_->getBuf());
    neon::Padding<float>::run(inputData,
                              inputPadding,
                              batchSize * inputChannels,
                              inputHeight,
                              inputWidth,
                              padInputHeight,
                              padInputWidth);

    std::function<void(
        const float*, const float*, int, int, int, int, int, int, float*)>
        DepthWiseConv;

    if (filterWidth == 3 && strideW() == 1) {
      DepthWiseConv = neon::DepthwiseConvKernel<3, 1>::run;
    } else if (filterWidth == 3 && strideW() == 2) {
      DepthWiseConv = neon::DepthwiseConvKernel<3, 2>::run;
    } else if (filterWidth == 4 && strideW() == 1) {
      DepthWiseConv = neon::DepthwiseConvKernel<4, 1>::run;
    } else if (filterWidth == 4 && strideW() == 2) {
      DepthWiseConv = neon::DepthwiseConvKernel<4, 2>::run;
    } else {
      LOG(FATAL) << "Not supported";
    }

    for (int i = 0; i < batchSize; i++) {
      DepthWiseConv(inputPadding,
                    filterData,
                    padInputHeight,
                    padInputWidth,
                    outputChannels,
                    outputHeight,
                    outputWidth,
                    filterMultiplier,
                    outputData);
      inputPadding += inputChannels * padInputHeight * padInputWidth;
      outputData += outputChannels * outputHeight * outputWidth;
    }
  }
};

#ifndef PADDLE_TYPE_DOUBLE
REGISTER_TYPED_FUNC(NeonDepthwiseConv, CPU, NeonDepthwiseConvFunction);
#endif

#endif

}  // namespace paddle
