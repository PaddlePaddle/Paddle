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

#include "FunctionTest.h"

namespace paddle {

template <DeviceType DType1, DeviceType DType2>
void forward(Compare2Function<DType1, DType2>& test,
             const TensorShape& input,
             const TensorShape& filter,
             const TensorShape& output) {
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, output));
  test.run();
}

template <DeviceType DType1, DeviceType DType2>
void backward_input(Compare2Function<DType1, DType2>& test,
                    const TensorShape& input,
                    const TensorShape& filter,
                    const TensorShape& output) {
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, input), ADD_TO);
  test.run();
}

template <DeviceType DType1, DeviceType DType2>
void backward_filter(Compare2Function<DType1, DType2>& test,
                     const TensorShape& input,
                     const TensorShape& filter,
                     const TensorShape& output) {
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, filter), ADD_TO);
  test.run();
}

template <DeviceType DType1, DeviceType DType2>
using Function = void (*)(Compare2Function<DType1, DType2>& test,
                          const TensorShape& input,
                          const TensorShape& filter,
                          const TensorShape& output);

/**
 * \brief A basic convolution function test interface.
 *
 * \param conv1         type name of convolution function 1.
 * \param conv2         type name of convolution function 2.
 * \param function      test function, can be one of the forward, backward_input
 *                      backward_filter function.
 * Example:
 * 1. Compare GemmConv's CPU and GPU implementation:
 *   Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
 *      "GemmConv-CPU", "GemmConv-GPU", forward);
 */
template <DeviceType DType1, DeviceType DType2>
void Convolution(const std::string& conv1,
                 const std::string& conv2,
                 Function<DType1, DType2> function) {
  for (size_t batchSize : {1, 5}) {
    for (size_t inputSize : {7, 14, 31}) {
      for (size_t filterSize : {1, 3, 5}) {
        for (size_t inputChannels : {3, 16}) {
          for (size_t outputChannels : {3, 16}) {
            if (outputChannels < inputChannels) continue;
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
                for (size_t dilation : {1, 3}) {
                  if (padding >= filterSize) break;
                  size_t filterS = (filterSize - 1) * dilation + 1;

                  if (inputSize + 2 * padding < filterS) break;

                  if ((conv1 == "NaiveConv-CPU" || conv2 == "NaiveConv-CPU" ||
                       conv1 == "NNPACKConv-CPU" ||
                       conv2 == "NNPACKConv-CPU") &&
                      dilation > 1)
                    break;

                  // NNPACK only supports stride = 1 if batchSize > 1
                  if ((conv1 == "NNPACKConv-CPU" ||
                       conv2 == "NNPACKConv-CPU") &&
                      batchSize > 1 && stride > 1)
                    break;

                  size_t outputSize =
                      (inputSize - filterS + 2 * padding + stride) / stride;
                  VLOG(3) << " batchSize=" << batchSize
                          << " inputChannels=" << inputChannels
                          << " inputHeight=" << inputSize
                          << " inputWidth=" << inputSize
                          << " outputChannels=" << outputChannels
                          << " filterHeight=" << filterSize
                          << " filterWidth=" << filterSize
                          << " outputHeight=" << outputSize
                          << " outputWidth=" << outputSize
                          << " stride=" << stride << " padding=" << padding;

                  std::vector<size_t> paddings = {padding, padding};
                  std::vector<size_t> strides = {stride, stride};
                  std::vector<size_t> dilations = {dilation, dilation};
                  Compare2Function<DType1, DType2> test(
                      conv1,
                      conv2,
                      FuncConfig()
                          .set("paddings", paddings)
                          .set("strides", strides)
                          .set("dilations", dilations)
                          .set("groups", (size_t)1)
                          .set("algo", (std::string) "auto"));

                  TensorShape input{
                      batchSize, inputChannels, inputSize, inputSize};
                  TensorShape filter{
                      outputChannels, inputChannels, filterSize, filterSize};
                  TensorShape output{
                      batchSize, outputChannels, outputSize, outputSize};

                  function(test, input, filter, output);
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * \brief A convolution function test interface for
 *        image height is not equal image width.
 */
template <DeviceType DType1, DeviceType DType2>
void Convolution2(const std::string& conv1,
                  const std::string& conv2,
                  Function<DType1, DType2> function) {
  for (size_t batchSize : {4}) {
    for (size_t inputHeight : {7, 31}) {
      for (size_t inputWidth : {10, 54}) {
        for (size_t filterHeight : {1, 5}) {
          for (size_t filterWidth : {3, 7}) {
            for (size_t inputChannels : {7}) {
              for (size_t outputChannels : {7}) {
                size_t stride = 1;
                size_t padding = 0;
                size_t dilation = 1;
                size_t outputHeight =
                    (inputHeight - filterHeight + 2 * padding + stride) /
                    stride;
                size_t outputWidth =
                    (inputWidth - filterWidth + 2 * padding + stride) / stride;
                VLOG(3) << " batchSize=" << batchSize
                        << " inputChannels=" << inputChannels
                        << " inputHeight=" << inputHeight
                        << " inputWidth=" << inputWidth
                        << " outputChannels=" << outputChannels
                        << " filterHeight=" << filterHeight
                        << " filterWidth=" << filterWidth
                        << " outputHeight=" << outputHeight
                        << " outputWidth=" << outputWidth
                        << " stride=" << stride << " padding=" << padding;

                std::vector<size_t> paddings = {padding, padding};
                std::vector<size_t> strides = {stride, stride};
                std::vector<size_t> dilations = {dilation, dilation};
                Compare2Function<DType1, DType2> test(
                    conv1,
                    conv2,
                    FuncConfig()
                        .set("paddings", paddings)
                        .set("strides", strides)
                        .set("groups", (size_t)1)
                        .set("dilations", dilations)
                        .set("algo", (std::string) "auto"));

                TensorShape input{
                    batchSize, inputChannels, inputHeight, inputWidth};
                TensorShape filter{
                    outputChannels, inputChannels, filterHeight, filterWidth};
                TensorShape output{
                    batchSize, outputChannels, outputHeight, outputWidth};

                function(test, input, filter, output);
              }
            }
          }
        }
      }
    }
  }
}

/**
 * \brief A convolution function test interface for depthwise convolution.
 */
template <DeviceType DType1, DeviceType DType2>
void DepthwiseConvolution(const std::string& conv1,
                          const std::string& conv2,
                          Function<DType1, DType2> function) {
  for (size_t batchSize : {1, 32}) {
    for (size_t inputSize : {7, 14, 54}) {
      for (size_t filterSize : {3, 4}) {
        for (size_t inputChannels : {32}) {
          for (size_t outputChannels : {32, 64}) {
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
                // NNPACK only supports stride = 1 if batchSize > 1,
                // and there has some bug when batchSize > 1 and groups != 1
                if ((conv1 == "NNPACKConv-CPU" || conv2 == "NNPACKConv-CPU") &&
                    batchSize > 1)
                  break;

                size_t outputSize =
                    (inputSize - filterSize + 2 * padding + stride) / stride;
                VLOG(3) << " batchSize=" << batchSize
                        << " inputChannels=" << inputChannels
                        << " inputHeight=" << inputSize
                        << " inputWidth=" << inputSize
                        << " outputChannels=" << outputChannels
                        << " filterHeight=" << filterSize
                        << " filterWidth=" << filterSize
                        << " outputHeight=" << outputSize
                        << " outputWidth=" << outputSize << " stride=" << stride
                        << " padding=" << padding;

                std::vector<size_t> paddings = {padding, padding};
                std::vector<size_t> strides = {stride, stride};
                std::vector<size_t> dilations = {1, 1};
                size_t groups = inputChannels;
                Compare2Function<DType1, DType2> test(
                    conv1,
                    conv2,
                    FuncConfig()
                        .set("paddings", paddings)
                        .set("strides", strides)
                        .set("groups", groups)
                        .set("dilations", dilations)
                        .set("algo", (std::string) "auto"));

                TensorShape input{
                    batchSize, inputChannels, inputSize, inputSize};
                TensorShape filter{groups,
                                   outputChannels / groups,
                                   inputChannels / groups,
                                   filterSize,
                                   filterSize};
                TensorShape output{
                    batchSize, outputChannels, outputSize, outputSize};

                function(test, input, filter, output);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace paddle
