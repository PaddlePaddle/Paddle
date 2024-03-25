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
#include "paddle/utils/optional.h"
#include "paddle/utils/small_vector.h"

namespace egr {

using paddle::Tensor;
using TupleOfTwoTensors = std::tuple<Tensor, Tensor>;
using TupleOfThreeTensors = std::tuple<Tensor, Tensor, Tensor>;
using TupleOfFourTensors = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using TupleOfFiveTensors = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using TupleOfSixTensors =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using TupleOfTensorAndVector =
    std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>;

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const Tensor& tensor);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const paddle::optional<Tensor>& tensor);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfTwoTensors& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfThreeTensors& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfFourTensors& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfFiveTensors& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfSixTensors& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const std::vector<Tensor>& tensors);

TEST_API void CheckTensorHasNanOrInf(
    const std::string& api_name,
    const paddle::optional<std::vector<Tensor>>& tensors);

TEST_API void CheckTensorHasNanOrInf(const std::string& api_name,
                                     const TupleOfTensorAndVector& tensors);

void SetCheckOpList(const std::string& check_op_list);

void SetSkipOpList(const std::string& skip_op_list);

TEST_API void CheckTensorHasNanOrInf(
    const std::string& api_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>& tensors);

template <typename TupleT, size_t N, size_t Last>
struct NanInfChecker {
  void operator()(const std::string& api_name, const TupleT& tensors) {
    CheckTensorHasNanOrInf(api_name, std::get<N>(tensors));
    NanInfChecker<TupleT, N + 1, Last>()(api_name, tensors);
  }
};

template <typename TupleT, size_t N>
struct NanInfChecker<TupleT, N, N> {
  void operator()(const std::string& api_name, const TupleT& tensors) {
    CheckTensorHasNanOrInf(api_name, std::get<N>(tensors));
  }
};

template <typename TupleT>
void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleT& tensors) {
  constexpr size_t size = std::tuple_size<TupleT>::value;
  NanInfChecker<TupleT, 0, size - 1>()(api_name, tensors);
}
}  // namespace egr
