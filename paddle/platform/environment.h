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

#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "paddle/platform/enforce.h"
#include "paddle/string/piece.h"

extern char** environ;  // for environment variables

namespace paddle {
namespace platform {

inline void SetEnvVariable(const std::string& name, const std::string& value) {
  PADDLE_ENFORCE_NE(setenv(name.c_str(), value.c_str(), 1), -1,
                    "Failed to set environment variable %s=%s", name, value);
}

inline void UnsetEnvVariable(const std::string& name) {
  PADDLE_ENFORCE_NE(unsetenv(name.c_str()), -1,
                    "Failed to unset environment variable %s", name);
}

inline bool IsEnvVarDefined(const std::string& name) {
  return std::getenv(name.c_str()) != nullptr;
}

inline std::string GetEnvValue(const std::string& name) {
  PADDLE_ENFORCE(IsEnvVarDefined(name),
                 "Tried to access undefined environment variable %s", name);
  return std::getenv(name.c_str());
}

inline std::vector<std::string> GetAllEnvVariables() {
  std::vector<std::string> vars;
  for (auto var = environ; *var != nullptr; ++var) {
    auto tail = string::Index(*var, "=");
    auto name = string::SubStr(*var, 0, tail).ToString();
    vars.push_back(name);
  }
  return vars;
}

}  // namespace platform
}  // namespace paddle
