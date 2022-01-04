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

#pragma once

#include <type_traits>
#include <utility>
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

template <typename ReleaseCallback>
class ScopeGuard {
 public:
  explicit ScopeGuard(const ReleaseCallback &callback) : callback_(callback) {}

  ~ScopeGuard() { callback_(); }

 private:
  DISABLE_COPY_AND_ASSIGN(ScopeGuard);

 private:
  ReleaseCallback callback_;
};

// Two macros are needed here.
// See:
// https://stackoverflow.com/questions/10379691/creating-macro-using-line-for-different-variable-names
#define _PADDLE_CONCAT_TOKEN(x, y) x##y
#define PADDLE_CONCAT_TOKEN(x, y) _PADDLE_CONCAT_TOKEN(x, y)

#define DEFINE_PADDLE_SCOPE_GUARD(...)                                     \
  auto PADDLE_CONCAT_TOKEN(__scope_guard_func, __LINE__) = __VA_ARGS__;    \
  ::paddle::framework::ScopeGuard<typename std::remove_reference<decltype( \
      PADDLE_CONCAT_TOKEN(__scope_guard_func, __LINE__))>::type>           \
      PADDLE_CONCAT_TOKEN(__scope_guard, __LINE__)(                        \
          PADDLE_CONCAT_TOKEN(__scope_guard_func, __LINE__))

}  // namespace framework
}  // namespace paddle
