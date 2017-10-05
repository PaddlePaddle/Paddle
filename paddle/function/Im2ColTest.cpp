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

template <DeviceType Device, class T>
void TestIm2ColFunctor() {
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

                size_t height = channels * filterHeight * filterWidth;
                size_t width = outputHeight * outputWidth;
                VectorPtr input1 = Vector::create(imShape.getElements(), false);
                VectorPtr input2 = Vector::create(imShape.getElements(), false);
                MatrixPtr output1 = Matrix::create(height, width, false, false);
                MatrixPtr output2 = Matrix::create(width, height, false, false);
                input1->uniform(0.001, 1);
                input2->copyFrom(*input1);

                Im2ColFunctor<kCFO, Device, T> im2Col1;
                Im2ColFunctor<kOCF, Device, T> im2Col2;
                im2Col1(input1->getData(),
                        imShape,
                        output1->getData(),
                        colShape1,
                        stride,
                        stride,
                        padding,
                        padding);
                im2Col2(input2->getData(),
                        imShape,
                        output2->getData(),
                        colShape2,
                        stride,
                        stride,
                        padding,
                        padding);

                // The transposition of the result of ColFormat == kCFO
                // is equal to the result of ColFormat == kOCF.
                MatrixPtr test;
                output2->transpose(test, true);
                autotest::TensorCheckErr(*output1, *test);

                Col2ImFunctor<kCFO, Device, T> col2Im1;
                Col2ImFunctor<kOCF, Device, T> col2Im2;
                col2Im1(input1->getData(),
                        imShape,
                        output1->getData(),
                        colShape1,
                        stride,
                        stride,
                        padding,
                        padding);
                col2Im2(input2->getData(),
                        imShape,
                        output2->getData(),
                        colShape2,
                        stride,
                        stride,
                        padding,
                        padding);

                autotest::TensorCheckErr(*input1, *input2);
              }
            }
          }
        }
      }
    }
  }
}

TEST(Im2ColFunctor, CPU) { TestIm2ColFunctor<DEVICE_TYPE_CPU, float>(); }

#ifdef PADDLE_WITH_CUDA

TEST(Im2ColFunctor, GPU) { TestIm2ColFunctor<DEVICE_TYPE_GPU, float>(); }

#endif

}  // namespace paddle
