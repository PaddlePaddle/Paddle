// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include <random>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {

/**
 * @brief  Fill an int Tensor with random data, which is going to be [low,
 * high).
 *
 * @param tensor  A Tensor that needs to be filled with data has to be of type
 * Int.
 * @param target  The type of device that tensor need.
 * @param seed    Random number seed. Default setting is -1.
 * @param low     Set the lower bound of the data range, which is represented as
 * [low, high).
 * @param high    Set the upper bound of the data range, which is represented as
 * [low, high).
 */
void SetRandInt(hlir::framework::Tensor tensor,
                const common::Target& target,
                int seed = -1,
                int low = 0,
                int high = 11);

template <typename T>
void SetRandData(hlir::framework::Tensor tensor,
                 const common::Target& target,
                 int seed = -1);

template <typename T>
std::vector<T> GetTensorData(const hlir::framework::Tensor& tensor,
                             const common::Target& target);

}  // namespace cinn
