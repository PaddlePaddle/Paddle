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

TEST(Crop, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {5, 5, 32}) {
      for (size_t imgSizeH : {5, 33, 100}) {
        for (size_t imgSizeW : {5, 32, 96}) {
          VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                  << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW;
          for (bool test_grad : {false, true}) {
            CpuGpuFuncCompare compare(
                test_grad ? "CropGrad" : "Crop",
                FuncConfig()
                    .set<std::vector<uint32_t>>("crop_corner", {0, 1, 1, 1})
                    .set<std::vector<uint32_t>>("crop_shape", {0, 2, 3, 3}));
            TensorShape inDims{numSamples, channels, imgSizeH, imgSizeW};
            TensorShape outDims{numSamples, 2, 3, 3};
            compare.addInputs(
                BufferArg(VALUE_TYPE_FLOAT, test_grad ? outDims : inDims));
            compare.addOutputs(BufferArg(VALUE_TYPE_FLOAT,
                                         test_grad ? inDims : outDims,
                                         test_grad ? ADD_TO : ASSIGN_TO),
                               test_grad ? ADD_TO : ASSIGN_TO);
            compare.run();
          }
        }
      }
    }
  }
}

}  // namespace paddle
