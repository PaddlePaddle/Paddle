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

#include "paddle/fluid/lite/core/types.h"

namespace paddle {
namespace lite {
namespace core {

KernelPickFactor& KernelPickFactor::ConsiderDataLayout() {
  data_ |= static_cast<int>(Factor::DataLayoutFirst);
  return *this;
}
KernelPickFactor& KernelPickFactor::ConsiderPrecision() {
  data_ |= static_cast<int>(Factor::PrecisionFirst);
  return *this;
}
KernelPickFactor& KernelPickFactor::ConsiderTarget() {
  data_ |= static_cast<int>(Factor::TargetFirst);
  return *this;
}
KernelPickFactor& KernelPickFactor::ConsiderDevice() {
  data_ |= static_cast<int>(Factor::DeviceFirst);
  return *this;
}
bool KernelPickFactor::IsPrecisionConsidered() const {
  return data_ & static_cast<int>(Factor::PrecisionFirst);
}
bool KernelPickFactor::IsTargetConsidered() const {
  return data_ & static_cast<int>(Factor::TargetFirst);
}
bool KernelPickFactor::IsDataLayoutConsidered() const {
  return data_ & static_cast<int>(Factor::DataLayoutFirst);
}
bool KernelPickFactor::IsDeviceConsidered() const {
  return data_ & static_cast<int>(Factor::DeviceFirst);
}

std::ostream& operator<<(std::ostream& os, const KernelPickFactor& k) {
  std::stack<bool> bits;
  auto data = k.data_;
  while (data) {
    bits.push(data % 2);
    data /= 2;
  }
  int nbits = bits.size();
  for (size_t i = 0; i < sizeof(data) * 8 - nbits; i++) {
    os << 0;
  }
  while (!bits.empty()) {
    os << bits.top();
    bits.pop();
  }
  return os;
}

}  // namespace core
}  // namespace lite
}  // namespace paddle
