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

#include "paddle/fluid/framework/details/ssa_graph.h"

namespace paddle {
namespace framework {
namespace details {

VarHandle* SSAGraph::InsertVariable(size_t position, const std::string& name,
                                    size_t scope_index, platform::Place place) {
  auto& var_vec = vars_.at(scope_index).at(name);
  PADDLE_ENFORCE_LT(position, var_vec.size());
  for (auto i = position; i < var_vec.size(); ++i) {
    ++var_vec[i]->version_;
  }
  auto* new_var = new VarHandle(position, scope_index, name, place);
  var_vec.insert(var_vec.begin() + position,
                 std::unique_ptr<VarHandle>(new_var));
  return new_var;
}
std::unique_ptr<VarHandle> SSAGraph::ExtractVariable(size_t position,
                                                     const std::string& name,
                                                     size_t scope_index) {
  auto& var_vec = vars_.at(scope_index).at(name);
  PADDLE_ENFORCE_LT(position, var_vec.size());
  for (auto i = position + 1; i < var_vec.size(); ++i) {
    --var_vec[i]->version_;
  }

  std::unique_ptr<VarHandle> res;
  std::swap(res, var_vec[position]);
  var_vec.erase(var_vec.begin() + position);
  return std::move(res);
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
