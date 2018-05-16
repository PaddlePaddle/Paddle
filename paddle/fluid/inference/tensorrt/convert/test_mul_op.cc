/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(MulOpConverter, main) {
  TRTConvertValidation validator(10);
  validator.DeclVar("x0", nvinfer1::Dims3(10, 6, 3));
  validator.DeclVar("x1", nvinfer1::Dims3(10, 3, 12));
  validator.DeclVar("y", nvinfer1::Dims3(10, 6, 12));
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
