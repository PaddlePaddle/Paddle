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

TEST(Pool2dOpConverter, main) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("pool2d-X", nvinfer1::Dims4(10, 3, 2, 2));
  validator.DeclOutputVar("pool2d-Out", nvinfer1::Dims4(10, 3, 1, 1));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d-X"});
  desc.SetOutput("Out", {"pool2d-Out"});

  std::vector<int> ksize({2, 2});
  std::vector<int> strides({1, 1});
  std::vector<int> paddings({0, 0});
  std::string pooling_t = "max";

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  // std::string temp = "";
  // (*desc.Proto()).SerializeToString(&temp);

  // std::cout << temp << std::endl;
  // std::ofstream f("__temp__", std::ios::out);
  // f << temp;

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(10);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(pool2d);
