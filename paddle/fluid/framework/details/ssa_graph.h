//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"

namespace paddle {
namespace framework {
namespace details {

// A SSA graph used by parallel executor.
struct SSAGraph {
  // all variable in each devices.
  // The outside vector is the device vector. Each element of this vector is a
  // map from variable name to variables. The variables, who have the same name,
  // will have a different version. The offset in the
  // `std::vector<std::unique_ptr<VarHandle>>` is the version of varaibles.
  std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>
      vars_;

  // aux variables to represent dependency. Useful to resolve data hazard.
  std::unordered_set<std::unique_ptr<VarHandleBase>> dep_vars_;

  // all operators. THIS FIELD IS UNORDERED.
  // NOTE: because C++ 11 lacks of `set.extract` method to remove a move-only
  // item from a set, here we use a vector to simulate a set.
  std::vector<std::unique_ptr<OpHandleBase>> ops_;

  std::unique_ptr<OpHandleBase> ExtractOp(size_t i) {
    std::unique_ptr<OpHandleBase> res = std::move(ops_[i]);
    if (i != ops_.size() - 1) {
      std::swap(ops_[i], ops_.back());
    }
    ops_.pop_back();
    return res;
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
