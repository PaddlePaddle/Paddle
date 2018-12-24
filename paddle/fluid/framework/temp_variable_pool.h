// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <deque>
#include <mutex>  // NOLINT
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

/*
 * temporary variables stored inside scope
 */
constexpr const char kTempVariablePool[] = "@TEMP_VAR_POOL@";

class TempVariablePool {
 public:
  TempVariablePool() = default;

  void Push(std::unique_ptr<Variable> &&var) {
// Not quite sure whether to add this macro guard
#ifndef PADDLE_ON_INFERENCE
    std::lock_guard<std::mutex> guard(mtx_);
#endif
    vars_.emplace_back(std::move(var));
  }

  void Clear() {
#ifndef PADDLE_ON_INFERENCE
    std::lock_guard<std::mutex> guard(mtx_);
#endif
    vars_.clear();
  }

 private:
  std::deque<std::unique_ptr<Variable>> vars_;
#ifndef PADDLE_ON_INFERENCE
  std::mutex mtx_;
#endif
};

}  // namespace framework
}  // namespace paddle
