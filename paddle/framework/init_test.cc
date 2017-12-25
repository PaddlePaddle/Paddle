/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include "gtest/gtest.h"

#include "paddle/framework/init.h"

TEST(Init, InitDevices) {
  using paddle::framework::InitDevices;
  std::vector<std::string> ds1 = {"CPU"};
  ASSERT_EQ(InitDevices(ds1), true);

#ifdef PADDLE_WITH_CUDA
  std::vector<std::string> ds2 = {"CPU", "GPU:0", "GPU:1"};
  ASSERT_EQ(InitDevices(ds2), true);

  // test re-init
  std::vector<std::string> ds3 = {"GPU:0", "GPU:1"};
  ASSERT_EQ(InitDevices(ds3), true);
#endif
}
