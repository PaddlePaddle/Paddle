// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>

#include <string>

namespace paddle {
namespace framework {
namespace details {

struct BuildStrategy {
  enum class ReduceStrategy {
    kAllReduce = 0x0000,
    kReduce = 0x0001,

    kOperationMask = 0x00FF,
    kFusedBit = 0x0100,

    kFusedAllReduce = kFusedBit | kAllReduce,
    kFusedReduce = kFusedBit | kReduce,
  };

  ReduceStrategy ReduceOperation() const {
    return static_cast<ReduceStrategy>(
        static_cast<uint16_t>(reduce_) &
        static_cast<uint16_t>(ReduceStrategy::kOperationMask));
  }

  enum class GradientScaleStrategy {
    kCoeffNumDevice = 0,
    kOne = 1,
    kCustomized = 2,
  };

  ReduceStrategy reduce_{ReduceStrategy::kAllReduce};
  GradientScaleStrategy gradient_scale_{GradientScaleStrategy::kCoeffNumDevice};

  std::string debug_graphviz_path_{""};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
