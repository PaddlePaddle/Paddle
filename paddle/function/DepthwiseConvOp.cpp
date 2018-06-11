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

#include "DepthwiseConvOp.h"
#include "ConvOp.h"

namespace paddle {

template <class T>
class DepthwiseConvFunctor<DEVICE_TYPE_CPU, T> {
 public:
  void operator()(const T* inputData,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* outputData) {
    // TODO(zhaolong) : cpu implementation of depthwise convolution
  }
};

template <class T>
class DepthwiseConvGradInputFunctor<DEVICE_TYPE_CPU, T> {
 public:
  void operator()(const T* outputGrad,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* inputGrad) {}
  // TODO(zhaolong) : cpu implementation of depthwise convolution
};

template <class T>
class DepthwiseConvGradFilterFunctor<DEVICE_TYPE_CPU, T> {
 public:
  void operator()(const T* outputGrad,
                  const T* inputData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* colData,
                  T* filterGrad) {}
  // TODO(zhaolong) : cpu implementation of depthwise convolution
};

/*
 * \brief Forward calculation of depthwise convolution.
 */
template <DeviceType Device>
class DepthwiseConvFunction : public ConvFunctionBase {
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

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];
    size_t filterMultiplier = outputChannels / groups_;
    CHECK_EQ(inputChannels, groups_);

    real* inputData = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* outputData = outputs[0].data<real>();

    DepthwiseConvFunctor<Device, real> depthwiseConv;
    depthwiseConv(inputData,
                  filterData,
                  batchSize,
                  outputChannels,
                  outputHeight,
                  outputWidth,
                  inputChannels,
                  inputHeight,
                  inputWidth,
                  filterMultiplier,
                  filterHeight,
                  filterWidth,
                  strideH(),
                  strideW(),
                  paddingH(),
                  paddingW(),
                  outputData);
  }
};

/*
 * \brief Backward input calculation of depthwise convolution.
 */
template <DeviceType Device>
class DepthwiseConvGradInputFunction : public ConvFunctionBase {
 public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& output = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& input = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    check(inputs, outputs);
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    const TensorShape& output = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& input = outputs[0].shape();

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];
    size_t filterMultiplier = outputChannels / groups_;
    CHECK_EQ(inputChannels, groups_);

    real* outputGrad = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* inputGrad = outputs[0].data<real>();

    DepthwiseConvGradInputFunctor<Device, real> depthwiseConvGradInput;
    depthwiseConvGradInput(outputGrad,
                           filterData,
                           batchSize,
                           outputChannels,
                           outputHeight,
                           outputWidth,
                           inputChannels,
                           inputHeight,
                           inputWidth,
                           filterMultiplier,
                           filterHeight,
                           filterWidth,
                           strideH(),
                           strideW(),
                           paddingH(),
                           paddingW(),
                           inputGrad);
  }
};

/*
 * \brief Backward filter calculation of depthwise convolution.
 */
template <DeviceType Device>
class DepthwiseConvGradFilterFunction : public ConvFunctionBase {
 public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& output = inputs[0].shape();
    const TensorShape& input = inputs[1].shape();
    const TensorShape& filter = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    check(inputs, outputs);
    const TensorShape& output = inputs[0].shape();
    const TensorShape& input = inputs[1].shape();
    const TensorShape& filter = outputs[0].shape();

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];
    size_t filterMultiplier = outputChannels / groups_;
    CHECK_EQ(inputChannels, groups_);

    real* outputGrad = inputs[0].data<real>();
    real* inputData = inputs[1].data<real>();
    real* filterGrad = outputs[0].data<real>();

    int size = outputChannels * filterHeight * filterWidth * outputHeight *
               outputWidth;
    resizeBuffer<Device>(size);
    real* colData = reinterpret_cast<real*>(memory_->getBuf());

    DepthwiseConvGradFilterFunctor<Device, real> depthwiseConvGradFilter;

    depthwiseConvGradFilter(outputGrad,
                            inputData,
                            batchSize,
                            outputChannels,
                            outputHeight,
                            outputWidth,
                            inputChannels,
                            inputHeight,
                            inputWidth,
                            filterMultiplier,
                            filterHeight,
                            filterWidth,
                            strideH(),
                            strideW(),
                            paddingH(),
                            paddingW(),
                            colData,
                            filterGrad);
  }
};

REGISTER_TYPED_FUNC(DepthwiseConv, CPU, DepthwiseConvFunction);
REGISTER_TYPED_FUNC(DepthwiseConvGradInput,
                    CPU,
                    DepthwiseConvGradInputFunction);
REGISTER_TYPED_FUNC(DepthwiseConvGradFilter,
                    CPU,
                    DepthwiseConvGradFilterFunction);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(DepthwiseConv, GPU, DepthwiseConvFunction);
REGISTER_TYPED_FUNC(DepthwiseConvGradInput,
                    GPU,
                    DepthwiseConvGradInputFunction);
REGISTER_TYPED_FUNC(DepthwiseConvGradFilter,
                    GPU,
                    DepthwiseConvGradFilterFunction);
#endif

}  // namespace paddle
