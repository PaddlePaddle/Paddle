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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void test_activation(std::string act_type) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("act-X", nvinfer1::Dims2(10, 6));
  validator.DeclOutputVar("act-Out", nvinfer1::Dims2(10, 6));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType(act_type);
  desc.SetInput("X", {"act-X"});
  desc.SetOutput("Out", {"act-Out"});

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(5);
}

TEST(ReluOpConverter, main) { test_activation("relu"); }

TEST(SigmoidOpConverter, main) { test_activation("sigmoid"); }

TEST(TanhOpConverter, main) { test_activation("tanh"); }

TEST(Relu6OpConverter, main) { test_activation("relu6"); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP_ITSELF(relu);
USE_OP_ITSELF(sigmoid);
USE_OP_ITSELF(tanh);
USE_OP(relu6);
