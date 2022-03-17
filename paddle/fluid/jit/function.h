// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/jit/ivalue.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace jit {

class Argument {
 public:
  Argument(const std::string& name,
           paddle::optional default_val = paddle::none_t, bool is_out = false)
      : name_(name), default_val_(default_val), is_out_(is_out) {}

 private:
  std::string name_;
  paddle::optional<IValue> default_val_;
  bool is_output_;
};

class FunctionSchema {
 public:
  FunctionSchema();

 private:
  std::vector<Argument> input_args;
  std::vector<Argument> output_args;
};

class Function {
 public:
  Function();
  ~Function() {}

 private:
  FunctionSchema schema_;
};

}  // namespace jit
}  // namespace paddle
