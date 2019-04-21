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
  KernelPickFactor& ConsiderPrecision();
  KernelPickFactor& ConsiderDataLayout();
  KernelPickFactor& ConsiderDevice();

  bool IsTargetConsidered() const;
  bool IsPrecisionConsidered() const;
  bool IsDataLayoutConsidered() const;
  bool IsDeviceConsidered() const {
    return data_ & static_cast<int>(Factor::DeviceFirst);
  }

 private:
  unsigned char data_{};
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
