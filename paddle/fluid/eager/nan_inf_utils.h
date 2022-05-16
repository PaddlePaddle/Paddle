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
#include <tuple>
#include <vector>

#include "paddle/fluid/eager/type_defs.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/small_vector.h"

namespace egr {

using paddle::experimental::Tensor;
using TwoTensorTuple = std::tuple<Tensor, Tensor>;
using ThreeTensorTuple = std::tuple<Tensor, Tensor, Tensor>;
using FourTensorTuple = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using FiveTensorTuple = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using SixTensorTuple =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

void CheckTensorHasNanOrInf(const std::string& api_name, const Tensor& tensor);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TwoTensorTuple& tensors);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const ThreeTensorTuple& tensors);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const FourTensorTuple& tensors);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const FiveTensorTuple& tensors);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const SixTensorTuple& tensors);

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const std::vector<Tensor>& tensors);

void CheckTensorHasNanOrInf(
    const std::string& api_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>& tensors);

}  // namespace egr
