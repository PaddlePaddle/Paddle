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
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

class Variable {
 public:
  template <typename T>
  T& Get() {
    return blob_;
  }

  template <typename T>
  T* GetMutable() {
    return any_cast<T>(&blob_);
  }

 private:
  any blob_;
};

}  // namespace lite
}  // namespace paddle
