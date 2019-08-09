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
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

using FeedFetchList = std::vector<lite::Tensor>;

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    return blob_.get<T>();
  }

  template <typename T>
  T* GetMutable() {
    if (!blob_.is<T>()) blob_.set<T>();
    return blob_.get_mutable<T>();
  }

  template <typename T>
  bool IsType() {
    return blob_.type() == typeid(T).hash_code();
  }

 private:
  // variant<int, float, std::string, lite::Tensor> blob_;
  variant<int, float, std::string, lite::Tensor, std::vector<lite::Tensor>>
      blob_;
};

}  // namespace lite
}  // namespace paddle
