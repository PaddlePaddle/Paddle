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

#include "PaddleAPI.h"

#include <algorithm>
#include <vector>

template <typename T1, typename T2>
void staticCastVector(std::vector<T2>* dest, const std::vector<T1>& src) {
  dest->resize(src.size());
  std::transform(src.begin(), src.end(), dest->begin(), [](T1 t) {
    return static_cast<T2>(t);
  });
}
