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

#include <gtest/gtest.h>
#include <memory>
#include "Function.h"
#include "FunctionTest.h"

namespace paddle {

enum TestType {
  kForwardTest = 0,
  kBackwardInputTest = 1,
  kBackwardFilterTest = 2,
};

template <DeviceType DType1, DeviceType DType2>
class ConvolutionTest {
public:
  ConvolutionTest(const std::string& conv1,
                  const std::string& conv2,
                  TestType type,
                  std::string algo = "auto") {
    for (size_t batchSize : {1, 32}) {
      for (size_t inputSize : {7, 14, 54}) {
        for (size_t filterSize : {1, 3, 5}) {
          for (size_t inputChannels : {3, 64}) {
            for (size_t outputChannels : {3, 64, 128}) {
              if (inputChannels < outputChannels) break;
              for (size_t stride : {1, 2}) {
                for (size_t padding : {0, 1}) {
                  if (padding >= filterSize) break;
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
                          << " outputWidth=" << outputSize
                          << " stride=" << stride << " padding=" << padding;

                  std::vector<size_t> paddings = {padding, padding};
                  std::vector<size_t> strides = {stride, stride};
                  Compare2Function<DType1, DType2> test(
                      conv1,
                      conv2,
                      FuncConfig()
                          .set("paddings", paddings)
                          .set("strides", strides)
                          .set("groups", (size_t)1)
                          .set("algo", algo));

                  TensorShape input{
                      batchSize, inputChannels, inputSize, inputSize};
                  TensorShape filter{
                      outputChannels, inputChannels, filterSize, filterSize};
                  TensorShape output{
                      batchSize, outputChannels, outputSize, outputSize};

                  if (type == kForwardTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.run();
                  } else if (type == kBackwardInputTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, input), ADD_TO);
                    test.run();
                  } else if (type == kBackwardFilterTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.run();
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// Mainly used to test cases where the height and width (input, filter)
// are not equal.
template <DeviceType DType1, DeviceType DType2>
class ConvolutionTest2 {
public:
  ConvolutionTest2(const std::string& conv1,
                   const std::string& conv2,
                   TestType type,
                   std::string algo = "auto") {
    for (size_t batchSize : {16}) {
      for (size_t inputHeight : {7, 31}) {
        for (size_t inputWidth : {10, 54}) {
          for (size_t filterHeight : {1, 5}) {
            for (size_t filterWidth : {3, 7}) {
              for (size_t inputChannels : {7}) {
                for (size_t outputChannels : {32}) {
                  size_t stride = 1;
                  size_t padding = 0;
                  size_t outputHeight =
                      (inputHeight - filterHeight + 2 * padding + stride) /
                      stride;
                  size_t outputWidth =
                      (inputWidth - filterWidth + 2 * padding + stride) /
                      stride;
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
                  Compare2Function<DType1, DType2> test(
                      conv1,
                      conv2,
                      FuncConfig()
                          .set("paddings", paddings)
                          .set("strides", strides)
                          .set("groups", (size_t)1)
                          .set("algo", algo));

                  TensorShape input{
                      batchSize, inputChannels, inputHeight, inputWidth};
                  TensorShape filter{
                      outputChannels, inputChannels, filterHeight, filterWidth};
                  TensorShape output{
                      batchSize, outputChannels, outputHeight, outputWidth};

                  if (type == kForwardTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.run();
                  } else if (type == kBackwardInputTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, input), ADD_TO);
                    test.run();
                  } else if (type == kBackwardFilterTest) {
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, output));
                    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
                    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, filter));
                    test.run();
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

TEST(Forward, GEMM) {
  ConvolutionTest<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU> test(
      "NaiveConv-CPU", "GemmConv-CPU", kForwardTest);
  ConvolutionTest2<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU> test2(
      "NaiveConv-CPU", "GemmConv-CPU", kForwardTest);
}

#ifndef PADDLE_ONLY_CPU
TEST(Forward, GEMM2) {
  ConvolutionTest<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test(
      "GemmConv-CPU", "GemmConv-GPU", kForwardTest);
  ConvolutionTest2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test2(
      "GemmConv-CPU", "GemmConv-GPU", kForwardTest);
}

TEST(BackwardInput, GEMM) {
  ConvolutionTest<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test(
      "GemmConvGradInput-CPU", "GemmConvGradInput-GPU", kBackwardInputTest);
  ConvolutionTest2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test2(
      "GemmConvGradInput-CPU", "GemmConvGradInput-GPU", kBackwardInputTest);
}

TEST(BackwardFilter, GEMM) {
  ConvolutionTest<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test(
      "GemmConvGradFilter-CPU", "GemmConvGradFilter-GPU", kBackwardFilterTest);
  ConvolutionTest2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> test2(
      "GemmConvGradFilter-CPU", "GemmConvGradFilter-GPU", kBackwardFilterTest);
}
#endif

}  // namespace paddle
