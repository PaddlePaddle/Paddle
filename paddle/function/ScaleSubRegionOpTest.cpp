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

#include <gtest/gtest.h>
#include "FunctionTest.h"

namespace paddle {

TEST(ScaleSubRegion, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {5, 32}) {
      for (size_t imgSizeH : {5, 33}) {
        for (size_t imgSizeW : {5, 32}) {
          for (real value : {-0.5, 0.0, 0.5}) {
            for (bool firstHalf : {false, true}) {
              VLOG(3) << " numSamples=" << numSamples
                      << " channels=" << channels << " imgSizeH=" << imgSizeH
                      << " imgSizeW=" << imgSizeW;

              for (bool testGrad : {false, true}) {
                CpuGpuFuncCompare compare(
                    testGrad ? "ScaleSubRegionGrad" : "ScaleSubRegion",
                    FuncConfig().set<real>("value", value));

                TensorShape shape{numSamples, channels, imgSizeH, imgSizeW};
                TensorShape indicesShape{numSamples, 6};

                compare.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
                compare.addInputs(BufferArg(VALUE_TYPE_FLOAT, indicesShape));

                compare.registerInitCallback([=](BufferArg& arg, size_t index) {
                  if (index == 1) {
                    real* data = (real*)arg.data();

                    for (size_t i = 0; i < numSamples; ++i) {
                      size_t offset = i * 6;
                      data[offset] = firstHalf ? 1 : channels / 2;
                      data[offset + 1] = firstHalf ? channels / 2 : channels;
                      data[offset + 2] = firstHalf ? 1 : imgSizeH / 2;
                      data[offset + 3] = firstHalf ? imgSizeH / 2 : imgSizeH;
                      data[offset + 4] = firstHalf ? 1 : imgSizeW / 2;
                      data[offset + 5] = firstHalf ? imgSizeW / 2 : imgSizeW;
                    }
                  }
                });

                compare.addOutputs(
                    BufferArg(
                        VALUE_TYPE_FLOAT, shape, testGrad ? ADD_TO : ASSIGN_TO),
                    testGrad ? ADD_TO : ASSIGN_TO);
                compare.run();
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace paddle
