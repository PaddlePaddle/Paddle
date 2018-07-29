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
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace platform {

// helper function for kenrels do not support fp16 at cuda device
template <typename T, typename V>
struct CastFunctor {
  HOSTDEVICE V operator()(const T& a) { return static_cast<V>(a); }
};

}  // namespace platform
}  // namespace paddle
