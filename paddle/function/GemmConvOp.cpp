/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "GemmConvOp.h"
#include "GemmFunctor.h"
#include "paddle/math/MemoryHandle.h"

namespace paddle {

/*
 * imData = [input_channels, input_height, input_width]
 * colData = [input_channels, filter_height, filter_width,
 *            output_height, output_width]
 */
template <class T>
class Im2ColFunctor<DEVICE_TYPE_CPU, T> {
public:
  void operator()(const T* imData,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int outputHeight,
                  int outputWidth,
                  T* colData) {
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterWidth / filterHeight;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          int imRowIdx = h * strideHeight + hOffset;
          int imColIdx = w * strideWidth + wOffset;
          if ((imRowIdx - paddingHeight) < 0 ||
              (imRowIdx - paddingHeight) >= inputHeight ||
              (imColIdx - paddingWidth) < 0 ||
              (imColIdx - paddingWidth) >= inputWidth) {
            colData[(c * outputHeight + h) * outputWidth + w] = T(0);
          } else {
            imRowIdx += c_im * inputHeight - paddingHeight;
            imColIdx -= paddingWidth;
            colData[(c * outputHeight + h) * outputWidth + w] =
                imData[imRowIdx * inputWidth + imColIdx];
          }
        }
      }
    }
  }
};

template <class T>
class Col2ImFunctor<DEVICE_TYPE_CPU, T> {
public:
  void operator()(const T* colData,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int outputHeight,
                  int outputWidth,
                  T* imData) {
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterWidth / filterHeight;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          int imRowIdx = h * strideHeight + hOffset;
          int imColIdx = w * strideWidth + wOffset;
          if ((imRowIdx - paddingHeight) >= 0 &&
              (imRowIdx - paddingHeight) < inputHeight &&
              (imColIdx - paddingWidth) >= 0 &&
              (imColIdx - paddingWidth) < inputWidth) {
            imRowIdx += c_im * inputHeight - paddingHeight;
            imColIdx -= paddingWidth;
            imData[imRowIdx * inputWidth + imColIdx] +=
                colData[(c * outputHeight + h) * outputWidth + w];
          }
        }
      }
    }
  }
};

/*
 * \brief Forward calculation of convolution.
 */
template <DeviceType Device>
class GemmConvFunction : public ConvFunctionBase {
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
    // TODO(hedaoyuan): Need to define some index macros,
    // to avoid useing 0 and 1.
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();

    real beta;
    if (outputs[0].getArgType() == ADD_TO) {
      beta = 1.0;
    } else {
      beta = 0.0;
    }

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];

    real* inputData = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* outputData = outputs[0].data<real>();

    size_t size = inputChannels / groups_ * filterHeight * filterWidth *
                  outputHeight * outputWidth;
    resizeBuffer<Device>(size);
    real* colData = reinterpret_cast<real*>(memory_->getBuf());

    Im2ColFunctor<Device, real> im2col;
    GemmFunctor<Device, real> gemm;
    size_t inputOffset = (inputChannels / groups_) * inputHeight * inputWidth;
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        im2col(inputData + g * inputOffset,
               inputChannels / groups_,
               inputHeight,
               inputWidth,
               filterHeight,
               filterWidth,
               strideH(),
               strideW(),
               paddingH(),
               paddingW(),
               outputHeight,
               outputWidth,
               colData);

        int M = outputChannels / groups_;
        int N = outputHeight * outputWidth;
        int K = inputChannels / groups_ * filterHeight * filterWidth;
        gemm(CblasNoTrans,
             CblasNoTrans,
             M,
             N,
             K,
             1.0f,
             filterData + g * filterOffset,
             K,
             colData,
             N,
             beta,
             outputData + g * outputOffset,
             N);
      }
      inputData += inputChannels * inputHeight * inputWidth;
      outputData += outputChannels * outputHeight * outputWidth;
    }
  }
};

/*
 * \brief Backward input calculation of convolution.
 */
template <DeviceType Device>
class GemmConvGradInputFunction : public ConvFunctionBase {
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
    check(inputs, outputs);
    // Since the implementation of Col2ImFunctor is ADD_TO,
    // this function only supports ADD_TO mode.
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

    real* outputGrad = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* inputGrad = outputs[0].data<real>();

    size_t size = inputChannels / groups_ * filterHeight * filterWidth *
                  outputHeight * outputWidth;
    resizeBuffer<Device>(size);
    real* colData = reinterpret_cast<real*>(memory_->getBuf());

    Col2ImFunctor<Device, real> col2im;
    GemmFunctor<Device, real> gemm;
    size_t inputOffset = (inputChannels / groups_) * inputHeight * inputWidth;
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        int K = outputChannels / groups_;
        int N = outputHeight * outputWidth;
        int M = inputChannels / groups_ * filterHeight * filterWidth;
        gemm(CblasTrans,
             CblasNoTrans,
             M,
             N,
             K,
             1.0f,
             filterData + g * filterOffset,
             M,
             outputGrad + g * outputOffset,
             N,
             0.0f,
             colData,
             N);

        col2im(colData,
               inputChannels / groups_,
               inputHeight,
               inputWidth,
               filterHeight,
               filterWidth,
               strideH(),
               strideW(),
               paddingH(),
               paddingW(),
               outputHeight,
               outputWidth,
               inputGrad + g * inputOffset);
      }
      inputGrad += inputChannels * inputHeight * inputWidth;
      outputGrad += outputChannels * outputHeight * outputWidth;
    }
  }
};

/*
 * \brief Backward filter calculation of convolution.
 */
template <DeviceType Device>
class GemmConvGradFilterFunction : public ConvFunctionBase {
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
    check(inputs, outputs);
    const TensorShape& output = inputs[0].shape();
    const TensorShape& input = inputs[1].shape();
    const TensorShape& filter = outputs[0].shape();

    real beta;
    if (outputs[0].getArgType() == ADD_TO) {
      beta = 1.0;
    } else {
      beta = 0.0;
    }

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];

    real* outputGrad = inputs[0].data<real>();
    real* inputData = inputs[1].data<real>();
    real* filterGrad = outputs[0].data<real>();

    size_t size = inputChannels / groups_ * filterHeight * filterWidth *
                  outputHeight * outputWidth;
    resizeBuffer<Device>(size);
    real* colData = reinterpret_cast<real*>(memory_->getBuf());

    Im2ColFunctor<Device, real> im2col;
    GemmFunctor<Device, real> gemm;
    size_t inputOffset = (inputChannels / groups_) * inputHeight * inputWidth;
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;
    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        im2col(inputData + g * inputOffset,
               inputChannels / groups_,
               inputHeight,
               inputWidth,
               filterHeight,
               filterWidth,
               strideH(),
               strideW(),
               paddingH(),
               paddingW(),
               outputHeight,
               outputWidth,
               colData);

        int M = outputChannels / groups_;
        int K = outputHeight * outputWidth;
        int N = inputChannels / groups_ * filterHeight * filterWidth;
        gemm(CblasNoTrans,
             CblasTrans,
             M,
             N,
             K,
             1.0f,
             outputGrad + g * outputOffset,
             K,
             colData,
             K,
             i == 0 ? beta : 1.0f,
             filterGrad + g * filterOffset,
             N);
      }
      inputData += inputChannels * inputHeight * inputWidth;
      outputGrad += outputChannels * outputHeight * outputWidth;
    }
  }
};

REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvFunction);
REGISTER_TYPED_FUNC(GemmConvGradInput, CPU, GemmConvGradInputFunction);
REGISTER_TYPED_FUNC(GemmConvGradFilter, CPU, GemmConvGradFilterFunction);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(GemmConv, GPU, GemmConvFunction);
REGISTER_TYPED_FUNC(GemmConvGradInput, GPU, GemmConvGradInputFunction);
REGISTER_TYPED_FUNC(GemmConvGradFilter, GPU, GemmConvGradFilterFunction);
#endif

}  // namespace paddle
