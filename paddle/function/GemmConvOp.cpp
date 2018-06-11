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
#include "GemmFunctor.h"
#include "Im2Col.h"
#include "paddle/math/MemoryHandle.h"

namespace paddle {

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
    bool needIm2col = isNeedIm2col(filter);

    TensorShape imShape =
        TensorShape({inputChannels / groups_, inputHeight, inputWidth});

    TensorShape colShape;
    real* colData = NULL;

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});
      resizeBuffer<Device>(colShape.getElements());
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Im2ColFunctor<kCFO, Device, real> im2col;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          im2col(inputData + g * inputOffset,
                 imShape,
                 colData,
                 colShape,
                 strideH(),
                 strideW(),
                 paddingH(),
                 paddingW(),
                 dilationH(),
                 dilationW());
        } else {
          colData = inputData + g * inputOffset;
        }
        int M = outputChannels / groups_;
        int N = outputHeight * outputWidth;
        int K = inputChannels / groups_ * filterHeight * filterWidth;
        BlasGemm<Device, real>::compute(false,
                                        false,
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

#ifdef PADDLE_MOBILE_INFERENCE

/*
 * \brief Forward calculation of convolution, optimized for mobile.
 */
template <DeviceType Device>
class GemmConvMobileFunction : public ConvFunctionBase {
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
    real* colData = NULL;
    bool needIm2col = isNeedIm2col(filter);

    TensorShape imShape =
        TensorShape({inputChannels / groups_, inputHeight, inputWidth});
    TensorShape colShape;

    // Max col matrix width 4096, Max col matrix size 4M.
    size_t outputHeightSteps =
        std::min(std::max(4096 / outputWidth, (size_t)1), outputHeight);
    size_t maxColWidth = outputHeightSteps * outputWidth;
    size_t channelSteps =
        std::min(std::max((1048576 / maxColWidth) / filterHeight * filterWidth,
                          (size_t)1),
                 inputChannels / groups_);
    size_t maxColHeight = channelSteps * filterHeight * filterWidth;

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});

      resizeBuffer<Device>(maxColHeight * maxColWidth * sizeof(real));
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Im2ColMobileFunctor<real> im2col;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    int nStride = outputHeight * outputWidth;
    int kStride = inputChannels / groups_ * filterHeight * filterWidth;
    for (size_t i = 0; i < batchSize; i++) {
      filterData = inputs[1].data<real>();
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          real beta_ = beta;
          for (size_t ic = 0; ic < inputChannels / groups_;
               ic += channelSteps) {
            int channels = std::min(inputChannels / groups_ - ic, channelSteps);
            for (size_t oh = 0; oh < outputHeight; oh += outputHeightSteps) {
              int height = std::min(outputHeight - oh, outputHeightSteps);

              int M = outputChannels / groups_;
              int N = height * outputWidth;
              int K = channels * filterHeight * filterWidth;
              // im2col
              im2col(inputData,
                     imShape,
                     colData,
                     colShape,
                     strideH(),
                     strideW(),
                     paddingH(),
                     paddingW(),
                     dilationH(),
                     dilationW(),
                     channels,
                     oh,
                     height,
                     N);

              // gemm
              BlasGemm<Device, real>::compute(
                  false,
                  false,
                  M,
                  N,
                  K,
                  1.0f,
                  filterData + ic * filterHeight * filterWidth,
                  kStride,
                  colData,
                  N,
                  beta_,
                  outputData + oh * outputWidth,
                  nStride);
            }
            beta_ = 1.0;
          }
        } else {
          int M = outputChannels / groups_;
          int N = outputHeight * outputWidth;
          int K = inputChannels / groups_ * filterHeight * filterWidth;
          BlasGemm<Device, real>::compute(false,
                                          false,
                                          M,
                                          N,
                                          K,
                                          1.0f,
                                          filterData,
                                          K,
                                          inputData,
                                          N,
                                          beta,
                                          outputData,
                                          N);
        }
        inputData += inputOffset;
        outputData += outputOffset;
        filterData += filterOffset;
      }
    }

    memory_.reset();
  }
};

#endif

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
    bool needIm2col = isNeedIm2col(filter);

    TensorShape imShape =
        TensorShape({inputChannels / groups_, inputHeight, inputWidth});

    TensorShape colShape;
    real* colData = NULL;

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});
      resizeBuffer<Device>(colShape.getElements());
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Col2ImFunctor<kCFO, Device, real> col2im;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        int K = outputChannels / groups_;
        int N = outputHeight * outputWidth;
        int M = inputChannels / groups_ * filterHeight * filterWidth;
        real scale = 0.0f;
        if (!needIm2col) {
          colData = inputGrad + g * inputOffset;
          scale = 1.0f;
        }
        BlasGemm<Device, real>::compute(true,
                                        false,
                                        M,
                                        N,
                                        K,
                                        1.0f,
                                        filterData + g * filterOffset,
                                        M,
                                        outputGrad + g * outputOffset,
                                        N,
                                        scale,
                                        colData,
                                        N);
        if (needIm2col) {
          col2im(inputGrad + g * inputOffset,
                 imShape,
                 colData,
                 colShape,
                 strideH(),
                 strideW(),
                 paddingH(),
                 paddingW(),
                 dilationH(),
                 dilationW());
        }
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
    bool needIm2col = isNeedIm2col(filter);

    TensorShape imShape =
        TensorShape({inputChannels / groups_, inputHeight, inputWidth});

    TensorShape colShape;
    real* colData = NULL;

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});
      resizeBuffer<Device>(colShape.getElements());
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Im2ColFunctor<kCFO, Device, real> im2col;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;
    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          im2col(inputData + g * inputOffset,
                 imShape,
                 colData,
                 colShape,
                 strideH(),
                 strideW(),
                 paddingH(),
                 paddingW(),
                 dilationH(),
                 dilationW());
        } else {
          colData = inputData + g * inputOffset;
        }
        int M = outputChannels / groups_;
        int K = outputHeight * outputWidth;
        int N = inputChannels / groups_ * filterHeight * filterWidth;
        BlasGemm<Device, real>::compute(false,
                                        true,
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

#ifdef PADDLE_MOBILE_INFERENCE
REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvMobileFunction);
#else
REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvFunction);
#endif
REGISTER_TYPED_FUNC(GemmConvGradInput, CPU, GemmConvGradInputFunction);
REGISTER_TYPED_FUNC(GemmConvGradFilter, CPU, GemmConvGradFilterFunction);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(GemmConv, GPU, GemmConvFunction);
REGISTER_TYPED_FUNC(GemmConvGradInput, GPU, GemmConvGradInputFunction);
REGISTER_TYPED_FUNC(GemmConvGradFilter, GPU, GemmConvGradFilterFunction);
#endif

}  // namespace paddle
