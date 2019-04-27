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

#pragma once

#include <stack>
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace core {

// Factors that impact the kernel picking strategy. Multiple factors can be
// considered together by using statement like 'factor1 | factor2'
class KernelPickFactor {
 public:
  using value_type = unsigned char;
  enum class Factor : int {
    // The following factors are sorted by priority.
    TargetFirst = 1,
    PrecisionFirst = 1 << 1,
    DataLayoutFirst = 1 << 2,
    DeviceFirst = 1 << 3,
  };

  // Has any factors considered.
  bool AnyFactorConsidered() const { return data_; }

  KernelPickFactor& ConsiderTarget();
  // Perfer a specific target, e.g. prefer CUDA kernels.
  KernelPickFactor& ConsiderPrecision();
  KernelPickFactor& ConsiderDataLayout();
  KernelPickFactor& ConsiderDevice();

  bool IsTargetConsidered() const;
  bool IsPrecisionConsidered() const;
  bool IsDataLayoutConsidered() const;
  bool IsDeviceConsidered() const;

  friend std::ostream& operator<<(std::ostream& os, const KernelPickFactor& k) {
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

 private:
  unsigned char data_{};
  TargetType target_{TARGET(kUnk)};
};

struct dim2 {
  int x{};
  int y{};

  dim2(int x, int y) : x(x), y(y) {}
};

struct dim3 {
  int x{};
  int y{};
  int z{};

  dim3(int x, int y, int z) : x(x), y(y), z(z) {}
};

}  // namespace core
}  // namespace lite
}  // namespace paddle
