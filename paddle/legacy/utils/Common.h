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

#include "Excepts.h"

/**
 * Disable copy macro.
 */
#define DISABLE_COPY(class_name)                \
  class_name(class_name &&) = delete;           \
  class_name(const class_name &other) = delete; \
  class_name &operator=(const class_name &other) = delete

namespace paddle {

#ifdef PADDLE_TYPE_DOUBLE
using real = double;
#else
using real = float;
#endif

}  // namespace paddle
