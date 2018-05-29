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
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(OpConverter, ConvertBlock) {
  framework::ProgramDesc prog;
  auto* block = prog.MutableBlock(0);
  auto* conv2d_op = block->AppendOp();
  conv2d_op->SetType("conv2d");

  OpConverter converter;
  converter.ConvertBlock(*block->Proto(), nullptr /*TensorRTEngine*/);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
