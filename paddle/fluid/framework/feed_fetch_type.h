/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/string_array.h"

namespace paddle {
namespace framework {
<<<<<<< HEAD
using FeedType =
    paddle::variant<phi::DenseTensor, Strings, phi::SparseCooTensor>;
using FeedList = std::vector<FeedType>;

using FetchType = paddle::variant<phi::DenseTensor,
                                  LoDTensorArray,
                                  framework::Vocab,
                                  phi::SparseCooTensor>;
=======
using FeedType = paddle::variant<LoDTensor, Strings>;
using FeedList = std::vector<FeedType>;

using FetchType = paddle::variant<LoDTensor, LoDTensorArray, framework::Vocab>;
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
using FetchList = std::vector<FetchType>;

using FetchUnmergedList = std::vector<std::vector<FetchType>>;
using FetchResultType = paddle::variant<FetchList, FetchUnmergedList>;

inline bool data_is_lod_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::DenseTensor)) {
    return true;
  }
  return false;
}

inline bool data_is_lod_tensor_array(const FetchType &data) {
  if (data.type() == typeid(LoDTensorArray)) {
    return true;
  }
  return false;
}

inline bool data_is_string_tensor(const FeedType &data) {
  if (data.type() == typeid(Strings)) {
    return true;
  }
  return false;
}

inline bool data_is_sparse_coo_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::SparseCooTensor)) {
    return true;
  }
  return false;
}

static const char kFeedOpType[] = "feed";
static const char kFetchOpType[] = "fetch";

}  // namespace framework
}  // namespace paddle
