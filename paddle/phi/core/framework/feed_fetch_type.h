// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {
using FeedType = phi::DenseTensor;
using FetchType = paddle::variant<phi::DenseTensor,
                                  phi::TensorArray,
                                  phi::Vocab,
                                  phi::SparseCooTensor>;

template <>
struct PhiVectorType<FeedType> {
  const char *type_name = "PhiVectorFeedType";
};

template <>
struct PhiVectorType<FetchType> {
  const char *type_name = "PhiVectorFetchType";
};

using FeedList = PhiVector<FeedType>;
using FetchList = PhiVector<FetchType>;
using FetchUnmergedList = std::vector<std::vector<FetchType>>;

}  // namespace phi

namespace paddle {
namespace framework {
using FeedType = phi::FeedType;
using FetchType = phi::FetchType;
using FeedList = phi::FeedList;
using FetchList = phi::FetchList;
using FetchUnmergedList = phi::FetchUnmergedList;
}  // namespace framework
}  // namespace paddle
