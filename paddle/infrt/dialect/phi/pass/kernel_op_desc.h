// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/infrt/dialect/infrt/common/types.h"

namespace infrt {

struct PhiKernelDesc {
  std::vector<Place> input_types;   // kernel input place
  std::vector<Place> output_types;  // kernel output place
  Place kernel_type;                // kernel place
};

std::string getPhiTargetPrefix(TargetType target);
std::string getPhiPrecisionSuffix(PrecisionType precision);
std::string getPhiLayoutSuffix(LayoutType layout);

std::vector<PhiKernelDesc> GetCandidateKernels(
    std::string name, const std::vector<Place>& valid_palces);

}  // namespace infrt
