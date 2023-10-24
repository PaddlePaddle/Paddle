// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <atomic>
#include <vector>

#include "paddle/cinn/auto_schedule/cost_model/xgb_cost_model.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * A C++ cost model which trains and predicts on ir::Expr
 *
 */
class ExprCostModel : public XgbCostModel {
 public:
  virtual float Predict(const ir::ModuleExpr& sample,
                        const common::Target& target) const;
  void Train(const std::vector<const ir::ModuleExpr*>& samples,
             const std::vector<float>& labels,
             const common::Target& target);
  void Update(const std::vector<const ir::ModuleExpr*>& samples,
              const std::vector<float>& labels,
              const common::Target& target);

 private:
  std::atomic<int> trained_times_{0};
};

}  // namespace auto_schedule
}  // namespace cinn
