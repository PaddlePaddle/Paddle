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

#include "Im2Col.h"
#include <gtest/gtest.h>
#include "Function.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/tests/TensorCheck.h"

namespace paddle {

TEST(Im2ColFunctor, real) {
  for (size_t channels : {1, 5, 32}) {
    for (size_t inputHeight : {5, 33, 100}) {
      for (size_t inputWidth : {5, 32, 96}) {
        for (size_t filterHeight : {1, 5}) {
          for (size_t filterWidth : {3, 7}) {
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
                if (inputHeight <= filterHeight || inputWidth <= filterWidth)
                  break;
                if (padding >= filterHeight || padding >= filterWidth) break;
                size_t outputHeight =
                    (inputHeight - filterHeight + 2 * padding + stride) /
                    stride;
                size_t outputWidth =
                    (inputWidth - filterWidth + 2 * padding + stride) / stride;

                TensorShape imShape =
                    TensorShape({channels, inputHeight, inputWidth});
                TensorShape colShape1 = TensorShape({channels,
                                                     filterHeight,
                                                     filterWidth,
                                                     outputHeight,
                                                     outputWidth});
                TensorShape colShape2 = TensorShape({outputHeight,
                                                     outputWidth,
                                                     channels,
                                                     filterHeight,
                                                     filterWidth});

                VectorPtr input = Vector::create(imShape.getElements(), false);
                size_t height = channels * filterHeight * filterWidth;
                size_t width = outputHeight * outputWidth;
                MatrixPtr output1 = Matrix::create(height, width, false, false);
                MatrixPtr output2 = Matrix::create(width, height, false, false);
                Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, real> im2col1;
                Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, real> im2col2;

                input->uniform(0.001, 1);
                im2col1(input->getData(),
                        imShape,
                        output1->getData(),
                        colShape1,
                        stride,
                        stride,
                        padding,
                        padding);
                im2col2(input->getData(),
                        imShape,
                        output2->getData(),
                        colShape2,
                        stride,
                        stride,
                        padding,
                        padding);

                MatrixPtr test;
                output2->transpose(test, true);
                autotest::TensorCheckErr(*output1, *test);
              }
            }
          }
        }
      }
    }
  }
}

#if 0
TEST(Col2ImFunctor, real) {
  for (size_t channels : {1, 5, 32}) {
    for (size_t inputHeight : {5, 33, 100}) {
      for (size_t inputWidth : {5, 32, 96}) {
        for (size_t filterHeight : {1, 5}) {
          for (size_t filterWidth : {3, 7}) {
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
              }
            }
          }
        }
      }
    }
  }
}
#endif

}  // namespace paddle
