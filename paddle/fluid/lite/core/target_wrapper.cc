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

#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

size_t Place::hash() const {
  std::hash<int> h;
  size_t hash = h(static_cast<int>(target));
  hash = hash_combine(hash, static_cast<int>(precision));
  hash = hash_combine(hash, static_cast<int>(layout));
  hash = hash_combine(hash, static_cast<int>(device));
  return hash;
}

bool operator<(const Place &a, const Place &b) {
  if (a.target != b.target) return a.target < b.target;
  if (a.precision != b.precision) return a.precision < b.precision;
  if (a.layout != b.layout) return a.layout < b.layout;
  if (a.device != b.device) return a.device < b.device;
  return true;
}

std::string Place::DebugString() const {
  std::stringstream os;
  os << TargetToStr(target) << "/" << PrecisionToStr(precision) << "/"
     << DataLayoutToStr(layout);
  return os.str();
}

}  // namespace lite
}  // namespace paddle