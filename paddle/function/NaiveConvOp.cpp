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

#include "ConvOp.h"

namespace paddle {

/*
 * The three arguments are stored in memory in row major order.
 * inputData  = [batchSize, inputChannels, inputHeight, inputWidth]
 * filterData = [outputChannels, inputChannels, filterHeight, filterWidth]
 * outputData = [batchSize, outputChannels, outputHeight, outputWidth]
 */
template <class T>
class NaiveConvFunctor {
 public:
  void operator()(const T* inputData,
                  size_t batchSize,
                  size_t inputChannels,
                  size_t inputHeight,
                  size_t inputWidth,
                  const T* filterData,
                  size_t filterHeight,
                  size_t filterWidth,
                  T* outputData,
                  size_t outputChannels,
                  size_t outputHeight,
                  size_t outputWidth,
                  size_t paddingH,
                  size_t paddingW,
                  size_t strideH,
                  size_t strideW) {
    for (size_t batch = 0; batch < batchSize; batch++) {
      for (size_t outC = 0; outC < outputChannels; outC++) {
        for (size_t outH = 0; outH < outputHeight; outH++) {
          for (size_t outW = 0; outW < outputWidth; outW++) {
            const int inStartH = (outH * strideH) - paddingH;
            const int inStartW = (outW * strideW) - paddingW;
            T outValue = (T)0;
            for (size_t inC = 0; inC < inputChannels; inC++) {
              for (size_t fH = 0; fH < filterHeight; fH++) {
                for (size_t fW = 0; fW < filterWidth; fW++) {
                  T inValue;
                  const int inH = inStartH + fH;
                  const int inW = inStartW + fW;
                  if ((inH >= 0 && inH < (int)inputHeight) &&
                      (inW >= 0 && inW < (int)inputWidth)) {
                    size_t offsetInput =
                        batch * inputChannels * inputHeight * inputWidth +
                        inC * inputHeight * inputWidth + inH * inputWidth + inW;
                    inValue = inputData[offsetInput];
                  } else {
                    inValue = (T)0;
                  }
                  size_t offsetFilter =
                      outC * inputChannels * filterHeight * filterWidth +
                      inC * filterHeight * filterWidth + fH * filterWidth + fW;
                  T filterValue = filterData[offsetFilter];
                  outValue += (inValue * filterValue);
                }
              }
            }

            size_t offset =
                batch * outputChannels * outputHeight * outputWidth +
                outC * outputHeight * outputWidth + outH * outputWidth + outW;
            outputData[offset] = outValue;
          }
        }
      }
    }
  }
};

template <DeviceType Device>
class NaiveConvFunction : public ConvFunctionBase {
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
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    check(inputs, outputs);

    size_t batchSize = inputs[0].shape()[0];
    size_t inputChannels = inputs[0].shape()[1];
    size_t inputHeight = inputs[0].shape()[2];
    size_t inputWidth = inputs[0].shape()[3];
    size_t filterHeight = inputs[1].shape()[2];
    size_t filterWidth = inputs[1].shape()[3];
    size_t outputChannels = outputs[0].shape()[1];
    size_t outputHeight = outputs[0].shape()[2];
    size_t outputWidth = outputs[0].shape()[3];

    real* inputData = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* outputData = outputs[0].data<real>();
    NaiveConvFunctor<real> conv;
    conv(inputData,
         batchSize,
         inputChannels,
         inputHeight,
         inputWidth,
         filterData,
         filterHeight,
         filterWidth,
         outputData,
         outputChannels,
         outputHeight,
         outputWidth,
         paddingH(),
         paddingW(),
         strideH(),
         strideW());
  }
};

REGISTER_TYPED_FUNC(NaiveConv, CPU, NaiveConvFunction);

}  // namespace paddle
