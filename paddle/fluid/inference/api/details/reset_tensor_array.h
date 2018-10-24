// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace details {

// Clean the TensorArray each batch to make the behavior the same with the
// training phase.
struct TensorArrayBatchCleaner {
  // Fix the tensor array not clear in the inference scenarios.
  void CollectTensorArrays(framework::Scope *scope);
  void ResetTensorArray();

 private:
  bool flag_{true};
  std::vector<framework::LoDTensorArray *> arrays_;
};

}  // namespace details
}  // namespace paddle
