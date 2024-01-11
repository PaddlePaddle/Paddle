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

#include "paddle/pir/pass/utils.h"

namespace pir {
namespace detail {

void PrintHeader(const std::string &header, std::ostream &os) {
  const size_t padding = 8;
  size_t mid_len = header.size() + ((padding - 3) * 2);
  os << "===" << std::string(mid_len, '-') << "===\n";
  os << std::string(padding, ' ') << header << "\n";
  os << "===" << std::string(mid_len, '-') << "===\n";
}

}  // namespace detail
}  // namespace pir
