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

#pragma once

#include <cstdlib>
#include <string>
#include "boost/lexical_cast.hpp"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

template <typename T>
T GetEnv(const std::string &env_var, const T &default_value) {
  const char *str_val = std::getenv(env_var.c_str());
  if (str_val == nullptr) return default_value;
  try {
    return boost::lexical_cast<T>(std::string(str_val));
  } catch (boost::bad_lexical_cast &) {
    PADDLE_THROW("Cannot parse environment variable %s=%s to type %s", env_var,
                 str_val, typeid(T).name());
  }
}

}  // namespace platform
}  // namespace paddle
