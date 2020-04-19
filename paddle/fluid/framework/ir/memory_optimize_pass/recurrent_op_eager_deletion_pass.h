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

#include <unordered_map>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/operators/controlflow/op_variant.h"
#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"

namespace paddle {
namespace framework {
namespace ir {

// Pass class set skip eager deletion vars for recurrent ops
class RecurrentOpEagerDeletionPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override;

 private:
  // Returns a std::unordered_map mapping from the device id to recurrent op and
  // grad op pair
  std::unordered_map<size_t, paddle::operators::OpAndGradOpPair>
  DeviceIdToRecurrentAndRecurrentGradOp(const Graph &graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
