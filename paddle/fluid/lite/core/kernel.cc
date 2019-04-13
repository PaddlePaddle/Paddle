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

#include "paddle/fluid/lite/core/kernel.h"

namespace paddle {
namespace lite {

bool operator==(const Place &a, const Place &b) {
  return a.target == b.target && a.precision == b.precision &&
         a.layout == b.layout;
}

bool operator<(const Place &a, const Place &b) {
  if (a.target != b.target)
    return a.target < b.target;
  else if (a.precision != b.precision)
    return a.precision < b.precision;
  else if (a.layout != b.layout)
    return a.layout < b.layout;
  return true;
}

bool ParamTypeRegistry::KeyCmp::operator()(
    const ParamTypeRegistry::key_t &a,
    const ParamTypeRegistry::key_t &b) const {
  if (a.kernel_type != b.kernel_type)
    return a.kernel_type < b.kernel_type;
  else if (a.io != b.io)
    return a.io < b.io;
  else if (a.offset != b.offset)
    return a.offset < b.offset;
  else if (!(a.place == b.place)) {
    return a.place < b.place;
  }
  return true;
}

}  // namespace lite
}  // namespace paddle