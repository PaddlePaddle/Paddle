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
#include <fstream>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void test_pool2d(bool global_pooling, bool ceil_mode,
                 std::string pool_type = "max") {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);

  // The ITensor's Dims should not contain the batch size.
  // So, the ITensor's Dims of input and output should be C * H * W.
  validator.DeclInputVar("pool2d-X", nvinfer1::Dims3(3, 6, 7));
  if (global_pooling)
    validator.DeclOutputVar("pool2d-Out", nvinfer1::Dims3(3, 1, 1));
  else if (ceil_mode)
    validator.DeclOutputVar("pool2d-Out", nvinfer1::Dims3(3, 3, 4));
  else
    validator.DeclOutputVar("pool2d-Out", nvinfer1::Dims3(3, 3, 3));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d-X"});
  desc.SetOutput("Out", {"pool2d-Out"});

  std::vector<int> ksize({2, 2});
  std::vector<int> strides({2, 2});
  std::vector<int> paddings({0, 0});
  std::string pooling_t = pool_type;

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("global_pooling", global_pooling);
  desc.SetAttr("ceil_mode", ceil_mode);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(3);
}

TEST(Pool2dOpConverter, normal) { test_pool2d(false, false); }
TEST(Pool2dOpConverter, test_global_pooling) { test_pool2d(true, false); }

TEST(Pool2dOpConverter, max_ceil_test) { test_pool2d(false, true); }
TEST(Pool2dOpConverter, avg_ceil_test) { test_pool2d(false, true, "avg"); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP_ITSELF(pool2d);
