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

#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
class SelectedRows;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace details {

// Clean the TensorArray each batch to make the behavior the same with the
// training phase.
struct TensorArrayBatchCleaner {
  TensorArrayBatchCleaner() {
    constexpr auto kTensorId = framework::VarTypeTrait<phi::DenseTensor>::kId;
    constexpr auto kLoDTensorId =
        framework::VarTypeTrait<framework::LoDTensor>::kId;
    constexpr auto kSelectedRowsId =
        framework::VarTypeTrait<phi::SelectedRows>::kId;
    constexpr auto kFetchListId =
        framework::VarTypeTrait<framework::FetchList>::kId;
    valid_types_.insert(kTensorId);
    valid_types_.insert(kLoDTensorId);
    valid_types_.insert(kSelectedRowsId);
    valid_types_.insert(kFetchListId);
  }
  // Collect the variables that are not Tensor or LoDTensor, and reset them to a
  // bool(trick), because some of them are containers, and some operators just
  // keep inserting new items without clearing the containers first; So the
  // memory grow larger and larger in inference service deployed online.
  void CollectNoTensorVars(framework::Scope *scope);
  void ResetNoTensorVars();

  // Fix the tensor array not clear in the inference scenarios.
  void CollectTensorArrays(framework::Scope *scope);
  void ResetTensorArray();

 private:
  bool flag_{true};
  bool no_tensor_flag_{true};
  std::vector<framework::LoDTensorArray *> arrays_;

  std::unordered_set<int> valid_types_;
  std::unordered_set<framework::Variable *> no_tensor_vars_;
};

}  // namespace details
}  // namespace paddle
