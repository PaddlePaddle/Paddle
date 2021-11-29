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

#include "paddle/infrt/common/axis.h"

#include <glog/logging.h>

#include <set>

#include "paddle/infrt/common/common.h"

namespace infrt {
namespace common {

const std::vector<std::string> kAxises({
    "i",  // level 0
    "j",  // level 1
    "k",  // level 2
    "a",  // level 3
    "b",  // level 4
    "c",  // level 5
    "d",  // level 6
    "e",  // level 7
    "f",  // level 8
    "g",  // level 9
    "h"   // level 10
});

static std::set<std::string> axis_set() {
  static std::set<std::string> x(kAxises.begin(), kAxises.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) { return axis_set().count(x); }

const std::string& axis_name(int level) {
  CHECK_LT(static_cast<size_t>(level), kAxises.size());
  return kAxises[level];
}

}  // namespace common
}  // namespace infrt
