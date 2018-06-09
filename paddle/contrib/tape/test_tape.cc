// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "paddle/contrib/tape/function.h"

TEST(Tape, TestMLP) {
  LOG(INFO) << "TestMLP";
  paddle::tape::Linear linear1(3, 3, "relu");
  paddle::tape::Linear linear2(3, 3, "relu");
  paddle::tape::Mean mean;

  paddle::tape::SGD sgd(0.001);

  for (int i = 0; i < 2; ++i) {
    paddle::tape::reset_global_tape();

    paddle::tape::VariableHandle input(new paddle::tape::Variable("input"));
    std::string initializer = "fill_constant";
    paddle::framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{3, 3};
    attrs["value"] = 1.0f;
    paddle::tape::Fill filler(initializer, attrs);
    filler(input);

    auto loss = mean(linear2(linear1(input)));

    paddle::tape::get_global_tape().Backward(loss);

    for (auto w : linear1.Params()) {
      sgd(w);
    }
    for (auto w : linear2.Params()) {
      sgd(w);
    }
  }
}

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
