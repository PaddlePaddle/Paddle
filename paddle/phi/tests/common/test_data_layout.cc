/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <sstream>

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/common/layout.h"

namespace phi {
namespace tests {

TEST(DataLayout, OStream) {
  std::ostringstream oss;
  oss << phi::DataLayout::UNDEFINED;
  EXPECT_EQ(oss.str(), "Undefined(AnyLayout)");
  oss.str("");
  oss << phi::DataLayout::ANY;
  EXPECT_EQ(oss.str(), "Undefined(AnyLayout)");
  oss.str("");
  oss << phi::DataLayout::NHWC;
  EXPECT_EQ(oss.str(), "NHWC");
  oss.str("");
  oss << phi::DataLayout::NCHW;
  EXPECT_EQ(oss.str(), "NCHW");
  oss.str("");
  oss << phi::DataLayout::MKLDNN;
  EXPECT_EQ(oss.str(), "MKLDNN");
  oss.str("");
  try {
    oss << phi::DataLayout::NUM_DATA_LAYOUTS;
  } catch (const std::exception& exception) {
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find("Unknown Data Layout type") != std::string::npos);
  }
}

}  // namespace tests
}  // namespace phi
