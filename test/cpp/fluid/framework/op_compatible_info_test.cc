// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_compatible_info.h"

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest.h"
#include "gtest/gtest_pred_impl.h"

namespace paddle {
namespace framework {

TEST(test_op_compatible_info, test_op_compatible) {
  auto comp_map = OpCompatibleMap();
  comp_map.InitOpCompatibleMap();

  ASSERT_NE(comp_map.GetDefaultRequiredVersion(), std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("reshape").required_version_,
            std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("layer_norm").required_version_,
            std::string());
  ASSERT_NE(comp_map.GetOpCompatibleInfo("layer_xx").required_version_,
            std::string());

  ASSERT_EQ(comp_map.IsRequireMiniVersion("unkop", "2.0.0"),
            OpCompatibleType::compatible);
  ASSERT_EQ(comp_map.IsRequireMiniVersion("unkop", "0.7.0"),
            OpCompatibleType::definite_not);

  ASSERT_EQ(comp_map.IsRequireMiniVersion("slice", "0.7.0"),
            OpCompatibleType::possible);
  ASSERT_EQ(comp_map.IsRequireMiniVersion("slice", "1.6.0"),
            OpCompatibleType::compatible);
}

}  // namespace framework
}  // namespace paddle
