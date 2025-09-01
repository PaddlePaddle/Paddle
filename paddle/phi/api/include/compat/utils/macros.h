// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

namespace compat {
#ifndef TORCH_EXTENSION_NAME
#define _EXPAND(x) x
#define TORCH_EXTENSION_NAME _EXPAND(PADDLE_EXTENSION_NAME)
#undef _EXPAND
#endif
#define UNSUPPORTED_FEATURE_IN_PADDLE(feature) \
  std::cerr << "Unsupported feature in Paddle: " << feature << std::endl;
}  // namespace compat
