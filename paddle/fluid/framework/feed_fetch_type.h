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
#include "paddle/phi/core/framework/feed_fetch_type.h"

namespace paddle {
namespace framework {

inline bool data_is_dense_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::DenseTensor)) {
    return true;
  }
  return false;
}

inline bool data_is_dense_tensor_array(const FetchType &data) {
  if (data.type() == typeid(phi::TensorArray)) {
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
