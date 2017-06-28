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
#include "paddle/function/Function.h"
#include "paddle/function/FunctionTest.h"

DEFINE_string(algo,
              "auto",
              "The algorithm (auto, ft8x8, ft16x16, wt8x8, "
              "implicit-gemm, or direct) for computing convolution of NNPACK.");

namespace paddle {

#define IS_NNPACK_SUPPORT(algo, filterSize, stride)        \
  if (algo == "direct" && filterSize != 1) continue;       \
  if (algo == "direct" && batchSize != 1) continue;        \
  if (algo == "wt8x8" && filterSize != 3) continue;        \
  if (algo == "implicit-gemm" && batchSize != 1) continue; \
  if (algo != "auto" && algo != "implicit-gemm" && stride > 1) continue;

class ConvolutionTest {
public:
  ConvolutionTest(const std::string& conv1,
                  const std::string& conv2,
                  std::string algo = "auto") {
    for (size_t batchSize : {1, 32}) {
      for (size_t inputSize : {7, 14, 54}) {
        for (size_t filterSize : {1, 3, 5}) {
          for (size_t inputChannels : {3, 64}) {
            for (size_t outputChannels : {3, 64, 128}) {
              if (inputChannels < outputChannels) break;
              for (size_t stride : {1, 2}) {
                // if batchSize > 1 NNPACKConv only supports stride = 1
                if (batchSize > 1 && stride > 1) break;
                for (size_t padding : {0, 1}) {
                  if (padding >= filterSize) break;
                  size_t outputSize =
                      (inputSize - filterSize + 2 * padding + stride) / stride;
                  IS_NNPACK_SUPPORT(algo, filterSize, stride);
                  LOG(INFO) << " batchSize=" << batchSize
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
                  Compare2Function<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU> test(
                      conv1,
                      conv2,
                      FuncConfig()
                          .set("paddings", paddings)
                          .set("strides", strides)
                          .set("groups", (size_t)1)
                          .set("algo", algo));

                  TensorShape shape0{
                      batchSize, inputChannels, inputSize, inputSize};
                  TensorShape shape1{
                      outputChannels, inputChannels, filterSize, filterSize};
                  TensorShape shape2{
                      batchSize, outputChannels, outputSize, outputSize};
                  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape0));
                  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape1));
                  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape2));
                  test.run();
                }
              }
            }
          }
        }
      }
    }
  }
};

TEST(Convolution, NNPACK) {
  // NNPACK only supports stride = 1
  ConvolutionTest test("GemmConv-CPU", "NNPACKConv-CPU", FLAGS_algo);
}

}  // namespace paddle
