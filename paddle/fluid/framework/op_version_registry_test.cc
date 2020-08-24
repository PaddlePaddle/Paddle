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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace compatible {

TEST(test_operator_version, test_operator_version) {
  REGISTER_OP_VERSION(test__)
      .AddCheckpoint(
          R"ROC(
        Upgrade reshape, modified one attribute [axis] and add a new attribute [size].
      )ROC",
          framework::compatible::OpVersionDesc()
              .ModifyAttr("axis",
                          "Increased from the original one method to two.", -1)
              .NewAttr("size",
                       "In order to represent a two-dimensional rectangle, the "
                       "parameter size is added."))
      .AddCheckpoint(
          R"ROC(
        Add a new attribute [height]
      )ROC",
          framework::compatible::OpVersionDesc().NewAttr(
              "height",
              "In order to represent a two-dimensional rectangle, the "
              "parameter height is added."));
}
}  // namespace compatible
}  // namespace framework
}  // namespace paddle
