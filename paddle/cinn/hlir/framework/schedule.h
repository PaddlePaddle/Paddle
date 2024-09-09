// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * \brief Global schedule container
 *  For operations and all the operations they depend on.
 *  The schedule per Operation is named as stage.
 */
class Schedule : public cinn::common::Object {
 public:
  const char* type_info() const override { return __type_info__; }

  /**
   * \brief Get the stage corresponds to the op
   * @param op The operation.
   */
  ir::Tensor operator[](const ir::Operation& op) {
    auto it = stage_map.find(op.name);
    PADDLE_ENFORCE(
        it != stage_map.end(),
        ::common::errors::NotFound(
            "Cannot find Stage for operator in the schedule", op.name));
    return it->second;
  }

  //! The output operations in original data flow graph
  std::vector<ir::Operation> outputs;
  /**
   * \brief list of all stages for ops.
   * The stages are sorted in dependency order.
   */
  std::vector<poly::Stage> stages;

  //! map of original operation to the stages
  absl::flat_hash_map<std::string, ir::Tensor> stage_map;

 private:
  static constexpr char* __type_info__ = "CINNSchedule";
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
