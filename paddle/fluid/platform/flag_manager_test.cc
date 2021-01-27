// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/flag_manager.h"
#include <cstring>
#include <utility>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_int32(file_manager_test_int32__, -1, "file manager test int32.");
DEFINE_double(file_manager_test_double__, -1.0, "file manager test float.");
DEFINE_double(file_manager_test_double_0__, -1.0, "file manager test float.");
DEFINE_double(file_manager_test_double_1__, -1.0, "file manager test float.");
DEFINE_bool(file_manager_test_bool__, false, "file manager test bool.");

namespace paddle {
namespace platform {

TEST(FlagManager, FlagManagerTest) {
  FlagRegistrar::Get().Insert("file_manager_test_int32__", 1, __FILE__);
  FlagRegistrar::Get().Insert("file_manager_test_double__", .23, __FILE__);
  FlagRegistrar::Get().Insert("file_manager_test_double_0__", 3.12, __FILE__);
  // override the old value of `file_manager_test_double_0__`
  FlagRegistrar::Get().Insert("file_manager_test_double_0__", 2., __FILE__);

  // Backward compatible
  std::vector<std::string> flags{"dummy", "--file_manager_test_bool__=true",
                                 "--file_manager_test_double_1__=4.12",
                                 "--file_manager_test_double_2__=123"};
  FlagRegistrar::Get().Insert(std::move(flags));

  auto fake_ParseCommandLineFlags = [](int* argc, char*** argv,
                                       bool remove_flags) -> uint32_t {
    CHECK_EQ(std::strcmp(*(*argv + 0), "dummy"), 0);
    CHECK_EQ(std::strcmp(*(*argv + 1), "--file_manager_test_double_2__=123"),
             0);
    return 0;
  };
  FlagRegistrar::Get().SyncFlagsOnce(fake_ParseCommandLineFlags);

  CHECK_EQ(FLAGS_file_manager_test_int32__, 1);
  CHECK_NEAR(FLAGS_file_manager_test_double__, 0.23, 0.0001);
  CHECK_NEAR(FLAGS_file_manager_test_double_0__, 2.0, 0.0001);
  CHECK_NEAR(FLAGS_file_manager_test_double_1__, 4.12, 0.0001);
}

}  // namespace platform
}  // namespace paddle
