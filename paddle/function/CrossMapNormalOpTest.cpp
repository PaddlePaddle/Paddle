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
#include "FunctionTest.h"

namespace paddle {

TEST(CrossMapNormal, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {1, 5, 32}) {
      for (size_t imgSizeH : {5, 33, 100}) {
        for (size_t imgSizeW : {5, 32, 96}) {
          for (size_t size : {1, 2, 3, 5, 7}) {
            VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                    << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW
                    << " size=" << size;

            FunctionCompare compare("CrossMapNormal",
                                    FuncConfig()
                                        .set("size", size)
                                        .set("scale", (real)1.5)
                                        .set("pow", (real)0.5));
            Dims dims{numSamples, channels, imgSizeH, imgSizeW};
            compare.cmpWithArg({Tensor(nullptr, dims)},
                               {Tensor(nullptr, dims), Tensor(nullptr, dims)},
                               {});
          }
        }
      }
    }
  }
}

TEST(CrossMapNormalGrad, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {1, 5, 32}) {
      for (size_t imgSizeH : {5, 33, 100}) {
        for (size_t imgSizeW : {5, 32, 96}) {
          for (size_t size : {1, 2, 3, 5, 7}) {
            VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                    << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW
                    << " size=" << size;

            FunctionCompare compare("CrossMapNormalGrad",
                                    FuncConfig()
                                        .set("size", size)
                                        .set("scale", (real)1.5)
                                        .set("pow", (real)0.5));
            Dims dims{numSamples, channels, imgSizeH, imgSizeW};
            compare.cmpWithArg({Tensor(nullptr, dims),
                                Tensor(nullptr, dims),
                                Tensor(nullptr, dims),
                                Tensor(nullptr, dims)},
                               {Tensor(nullptr, dims)},
                               {});
          }
        }
      }
    }
  }
}

}  // namespace paddle
