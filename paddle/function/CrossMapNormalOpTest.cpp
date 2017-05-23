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
            topology::Function config;
            config.type = "CrossMapNormalFwd";
            config.attributes.set("size", size)
                .set<double>("scale", 1.5)
                .set<double>("pow", 0.5);
            FunctionCompare test(config);

            // prepare input arguments
            TensorShape shape{numSamples, channels, imgSizeH, imgSizeW};
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            // run Function
            test.run();
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

            topology::Function config;
            config.type = "CrossMapNormalBwd";
            config.attributes.set("size", size)
                .set<double>("scale", 1.5)
                .set<double>("pow", 0.5);
            FunctionCompare test(config);

            TensorShape shape{numSamples, channels, imgSizeH, imgSizeW};
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape));
            // run Function
            test.run();
          }
        }
      }
    }
  }
}

}  // namespace paddle
