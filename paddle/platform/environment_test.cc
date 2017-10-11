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

#include "paddle/platform/environment.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(ENVIRONMENT, ACCESS) {
  namespace platform = paddle::platform;
  namespace string = paddle::string;

  platform::SetEnvVariable("PADDLE_USE_ENV", "TRUE");

  EXPECT_TRUE(platform::IsEnvVarDefined("PADDLE_USE_ENV"));
  EXPECT_EQ(platform::GetEnvValue("PADDLE_USE_ENV"), "TRUE");

  platform::UnsetEnvVariable("PADDLE_USE_ENV");
  EXPECT_FALSE(platform::IsEnvVarDefined("PADDLE_USE_ENV"));

  platform::SetEnvVariable("PADDLE_USE_ENV1", "Hello ");
  platform::SetEnvVariable("PADDLE_USE_ENV2", "World, ");
  platform::SetEnvVariable("PADDLE_USE_ENV3", "PaddlePaddle!");

  std::string env_info;
  auto vars = platform::GetAllEnvVariables();
  for_each(vars.begin(), vars.end(), [&](const std::string& var) {
    env_info += platform::GetEnvValue(var);
  });

  EXPECT_TRUE(string::Contains(env_info, "Hello World, PaddlePaddle!"));
  platform::UnsetEnvVariable("PADDLE_USE_ENV1");
  platform::UnsetEnvVariable("PADDLE_USE_ENV2");
  platform::UnsetEnvVariable("PADDLE_USE_ENV3");

  env_info.clear();
  vars = platform::GetAllEnvVariables();
  for_each(vars.begin(), vars.end(), [&](const std::string& var) {
    env_info += platform::GetEnvValue(var);
  });

  EXPECT_FALSE(string::Contains(env_info, "Hello World, PaddlePaddle!"));
  EXPECT_FALSE(platform::IsEnvVarDefined("PADDLE_USE_ENV1"));
  EXPECT_FALSE(platform::IsEnvVarDefined("PADDLE_USE_ENV2"));
  EXPECT_FALSE(platform::IsEnvVarDefined("PADDLE_USE_ENV3"));
}
