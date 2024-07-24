// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>

namespace paddle_ctor {

struct StaticGlobalWrapper {
  explicit StaticGlobalWrapper(const std::function<void()>& f) { f(); }
  StaticGlobalWrapper(const StaticGlobalWrapper&) = default;
  StaticGlobalWrapper(StaticGlobalWrapper&&) = default;
};

}  // namespace paddle_ctor

#define PD_CALL_BEFORE_MAIN(f) \
  ::paddle_ctor::StaticGlobalWrapper static_global_wrapper_##__LINE__(f)
