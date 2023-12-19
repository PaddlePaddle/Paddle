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

#include <pybind11/embed.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "paddle/cinn/common/cost_model.h"

namespace cinn {
namespace auto_schedule {

/**
 * A C++ cost model which calls Python xgboost via pybind
 *
 * Note: this class handles Python interpreter life time in class.
 * If you have to call other Python functions out of this class so that meet
 * life time conflict, you can check cinn::common::PythonInterpreterGuard
 *
 * For cinn::common::PythonInterpreterGuard, see:
 *   cinn/common/python_interpreter_guard.h .cc
 *
 * For pybind interpreter lifetime management, see:
 *
 *   https://pybind11.readthedocs.io/en/stable/advanced/embedding.html#interpreter-lifetime
 *   https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv422initialize_interpreterbiPPCKcb
 */
class XgbCostModel : public CostModel {
 public:
  XgbCostModel();
  ~XgbCostModel() = default;

  void Train(const std::vector<std::vector<float>>& samples,
             const std::vector<float>& labels) override;

  std::vector<float> Predict(
      const std::vector<std::vector<float>>& samples) const override;

  void Update(const std::vector<std::vector<float>>& samples,
              const std::vector<float>& labels) override;

  void Save(const std::string& path) override;

  void Load(const std::string& path) override;

 private:
  // Python xgboost module
  pybind11::module xgb_module_;
  // Object points to Python xgb.Booster()
  pybind11::object xgb_booster_;
  // atomic int to handle python interpreter lifetime and package dependency
  static std::atomic<int> xgb_cost_model_count_;
  // Default train rounds
  static constexpr int kTrainRound_ = 10;

  std::vector<std::vector<float>> update_samples_;
  std::vector<float> update_labels_;
};

}  // namespace auto_schedule
}  // namespace cinn
