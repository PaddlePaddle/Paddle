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

#include "ConvOp.h"
#include "paddle/math/MathFunctions.h"
#include "paddle/math/MemoryHandle.h"

namespace paddle {

/*
 * imData = [input_channels, input_height, input_width]
 * colData = [input_channels, filter_height, filter_width,
 *            output_height, output_width]
 */
template <class T>
class Im2ColFunctor {
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
      int c_im = c / filterHeight / filterWidth;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          // no c_im*height to Exclude the channel number
          int imgRowIdx = h * strideHeight + hOffset;
          int imgColIdx = w * strideWidth + wOffset;
          if ((imgRowIdx - paddingHeight) < 0 ||
              (imgRowIdx - paddingHeight) >= inputHeight ||
              (imgColIdx - paddingWidth) < 0 ||
              (imgColIdx - paddingWidth) >= inputWidth) {
            colData[(c * outputHeight + h) * outputWidth + w] = T(0);
          } else {
            imgRowIdx += c_im * inputHeight - paddingHeight;
            imgColIdx -= paddingWidth;
            colData[(c * outputHeight + h) * outputWidth + w] =
                imData[imgRowIdx * inputWidth + imgColIdx];
          }
        }
      }
    }
  }
};

/*
 * Function Arguments:
 *
 * \param inputs[0]  Input image data, is NCHW format, where N is batch size,
 *                   C is the number of channels, H and W is the height and
 *                   width of input image.
 * \param inputs[1]  Filter data, is MCHW, where M is the number of output
 *                   channels, C is the number of input channels, H and W
 *                   is height and width of filter.
 * \param outputs[0] Output image data, is NCHW format, where N is batch size,
  *                  C is the number of channels, H and W is the height and
 *                   width of output image.
 */
template <DeviceType Device>
class GemmConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    check(inputs, outputs);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    size_t batchSize = inputs[0].shape()[0];
    size_t inputChannels = inputs[0].shape()[1];
    size_t inputHeight = inputs[0].shape()[2];
    size_t inputWidth = inputs[0].shape()[3];
    size_t filterHeight = inputs[1].shape()[2];
    size_t filterWidth = inputs[1].shape()[2];
    size_t outputChannels = outputs[0].shape()[1];
    size_t outputHeight = outputs[0].shape()[2];
    size_t outputWidth = outputs[0].shape()[3];

    real* inputData = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* outputData = outputs[0].data<real>();

    size_t size =
        inputChannels * filterHeight * filterWidth * outputHeight * outputWidth;
    resizeBuffer(size);
    real* colData = reinterpret_cast<real*>(memory_->getBuf());

    Im2ColFunctor<real> im2col;
    for (size_t i = 0; i < batchSize; i++) {
      im2col(inputData,
             inputChannels,
             inputHeight,
             inputWidth,
             filterHeight,
             filterWidth,
             stride_,
             stride_,
             padding_,
             padding_,
             outputHeight,
             outputWidth,
             colData);

      int M = outputChannels;
      int N = outputHeight * outputWidth;
      int K = inputChannels * filterHeight * filterWidth;
      gemm<real>(CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 1.0f,
                 filterData,
                 K,
                 colData,
                 N,
                 0.0f,
                 outputData,
                 N);
      inputData += inputChannels * inputHeight * inputWidth;
      outputData += outputChannels * outputHeight * outputWidth;
    }
  }

  void resizeBuffer(size_t newSize) {
    if (!memory_ || newSize * sizeof(real) > memory_->getAllocSize()) {
      memory_ = std::make_shared<CpuMemoryHandle>(newSize * sizeof(real));
    }
  }

private:
  CpuMemHandlePtr memory_;
};

REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvFunction);

}  // namespace paddle
