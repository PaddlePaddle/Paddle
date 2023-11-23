// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/print_utils/print_loop_size.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

namespace cinn::adt {

std::string ToTxtString(const LoopSize& loop_size) {
  return std::to_string(loop_size.Get<std::int64_t>());
}

std::string ToTxtString(const List<LoopSize>& loop_sizes) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < loop_sizes->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(loop_sizes.Get(idx));
  }
  ret += "]";
  return ret;
}

}  // namespace cinn::adt
