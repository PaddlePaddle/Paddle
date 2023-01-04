/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/op_version_registry.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace compatible {

TEST(test_operator_version, test_operator_version) {
  REGISTER_OP_VERSION(op_name__)
      .AddCheckpoint(
          R"ROC(Fix the bug of reshape op, support the case of axis < 0)ROC",
          framework::compatible::OpVersionDesc().BugfixWithBehaviorChanged(
              "Support the case of axis < 0"))
      .AddCheckpoint(
          R"ROC(
        Upgrade reshape, modified one attribute [axis] and add a new attribute [size].
      )ROC",
          framework::compatible::OpVersionDesc()
              .ModifyAttr(
                  "axis", "Increased from the original one method to two.", -1)
              .NewAttr("size",
                       "In order to represent a two-dimensional rectangle, the "
                       "parameter size is added.",
                       0))
      .AddCheckpoint(
          R"ROC(
        Add a new attribute [height]
      )ROC",
          framework::compatible::OpVersionDesc().NewAttr(
              "height",
              "In order to represent a two-dimensional rectangle, the "
              "parameter height is added.",
              0))
      .AddCheckpoint(
          R"ROC(
        Add a input [X2] and a output [Y2]
      )ROC",
          framework::compatible::OpVersionDesc()
              .NewInput("X2", "The second input.")
              .NewOutput("Y2", "The second output."));

  REGISTER_OP_VERSION(op_name_0__)
      .AddCheckpoint(
          R"ROC(
        Incompatible upgrade of attribute [height], input [X2] and output [Y2]
      )ROC",
          framework::compatible::OpVersionDesc()
              .DeleteAttr("height",
                          "Parameters deleted due to interface alignment.")
              .ModifyInput("X2", "Modify input due to interface alignment.")
              .ModifyOutput("Y2", "Modify output due to interface alignment.")
              .DeleteInput("X2", "Delete input due to interface alignment.")
              .DeleteOutput("Y2", "Delete output due to interface alignment."));
}

TEST(test_pass_op_version_checker, test_pass_op_version_checker) {
  const std::string fake_op_name{"op_name__"};
  ASSERT_FALSE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "no_registered_capability_pass"));

  REGISTER_PASS_CAPABILITY(no_bind_pass);
  ASSERT_TRUE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "no_bind_pass"));

  REGISTER_PASS_CAPABILITY(test_pass1)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .LE("mul", 1)
              .EQ("fc", 0));
  ASSERT_TRUE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass1"));

  REGISTER_PASS_CAPABILITY(test_pass2)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .GE("mul", 0)
              .NE("fc", 0));
  ASSERT_FALSE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass2"));

  REGISTER_PASS_CAPABILITY(test_pass3)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .GE("mul", 0)
              .NE("fc", 0))
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .LE("mul", 1)
              .EQ("fc", 0));
  ASSERT_TRUE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass3"));

  REGISTER_PASS_CAPABILITY(test_pass4)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .GE(fake_op_name, 5)
              .EQ("fc", 0));
  ASSERT_FALSE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass4"));

  REGISTER_PASS_CAPABILITY(test_pass5)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .GE(fake_op_name, 4)
              .EQ("fc", 0));
  ASSERT_TRUE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass5"));

  REGISTER_PASS_CAPABILITY(test_pass6)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .EQ(fake_op_name, 4)
              .EQ("fc", 0));
  ASSERT_TRUE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass6"));

  REGISTER_PASS_CAPABILITY(test_pass7)
      .AddCombination(
          paddle::framework::compatible::OpVersionComparatorCombination()
              .NE(fake_op_name, 4)
              .EQ("fc", 0));
  ASSERT_FALSE(PassVersionCheckerRegistrar::GetInstance().IsPassCompatible(
      "test_pass7"));
}

}  // namespace compatible
}  // namespace framework
}  // namespace paddle
