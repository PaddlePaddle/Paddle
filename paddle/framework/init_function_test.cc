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

#include <gtest/gtest.h>
#include <paddle/framework/init_function.h>

static bool InitializeFlag = false;

static paddle::framework::InitFunction __init_func1__([] {
  InitializeFlag = true;
});

static int InitFlag = 0;

static paddle::framework::InitFunction __init_func2__([] { InitFlag = 1; });

static paddle::framework::InitFunction __init_func3__([] { InitFlag = 10; },
                                                      -1);

TEST(InitFunction, init_func) {
  paddle::framework::RunInitFunctions();
  ASSERT_TRUE(InitializeFlag);
  ASSERT_EQ(10, InitFlag);
}