//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/platform/os_info.h"
#include <thread>
#include "gtest/gtest.h"

TEST(ThreadInfo, TestThreadIdUtils) {
  using paddle::platform::GetCurrentThreadStdId;
  using paddle::platform::GetCurrentThreadId;
  using paddle::platform::GetAllThreadIds;
  EXPECT_EQ(std::hash<std::thread::id>()(std::this_thread::get_id()),
            GetCurrentThreadId().std_tid);
  auto ids = GetAllThreadIds();
  EXPECT_TRUE(ids.find(GetCurrentThreadStdId()) != ids.end());
}

TEST(ThreadInfo, TestThreadNameUtils) {
  using paddle::platform::GetCurrentThreadStdId;
  using paddle::platform::GetCurrentThreadName;
  using paddle::platform::SetCurrentThreadName;
  using paddle::platform::GetAllThreadNames;
  SetCurrentThreadName("MainThread");
  EXPECT_FALSE(SetCurrentThreadName("MainThread"));
  auto names = GetAllThreadNames();
  EXPECT_TRUE(names.find(GetCurrentThreadStdId()) != names.end());
  EXPECT_EQ("MainThread", names[GetCurrentThreadStdId()]);
  EXPECT_EQ("MainThread", GetCurrentThreadName());
}
