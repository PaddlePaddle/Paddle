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

#include "paddle/framework/lod_tensor.h"

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <glog/logging.h>

namespace paddle {
namespace framework {

LoD ToAbsOffset(const LoD& in) {
  // the lowest level stores relative offsets
  LoD result = in;
  for (size_t i = 1; i < result.size(); ++i) {
    size_t level = result.size() - i - 1;  // reversely
    for (auto& ele : result[level]) {
      if (ele == 0) continue;
      ele = result[level + 1][ele - 1];
    }
  }
  return result;
}

bool operator==(const LoD& a, const LoD& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto& a_level = a[i];
    const auto& b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace framework
}  // namespace paddle
