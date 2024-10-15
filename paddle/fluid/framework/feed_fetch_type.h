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
#include "paddle/phi/core/extended_tensor.h"

namespace phi {
using FeedType = paddle::
    variant<phi::DenseTensor, paddle::framework::Strings, phi::SparseCooTensor>;
using FetchType = paddle::variant<phi::DenseTensor,
                                  phi::TensorArray,
                                  paddle::framework::Vocab,
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
}  // namespace phi

namespace paddle {
namespace framework {
using FeedType = phi::FeedType;
using FetchType = phi::FetchType;
using FeedList = phi::FeedList;
using FetchList = phi::FetchList;
using FetchUnmergedList = std::vector<std::vector<FetchType>>;

inline bool data_is_lod_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::DenseTensor)) {
    return true;
  }
  return false;
}

inline bool data_is_lod_tensor_array(const FetchType &data) {
  if (data.type() == typeid(phi::TensorArray)) {
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
