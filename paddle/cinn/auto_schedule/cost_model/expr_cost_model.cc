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

#include "paddle/cinn/auto_schedule/cost_model/expr_cost_model.h"

#include <glog/logging.h>

#include <atomic>
#include <vector>

#include "paddle/cinn/auto_schedule/cost_model/feature.h"
#include "paddle/cinn/auto_schedule/cost_model/feature_extractor.h"
#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

float ExprCostModel::Predict(const ir::ModuleExpr& sample,
                             const common::Target& target) const {
  if (trained_times_.load() == 0) {
    return SearchState::NOT_INIT_COST;
  }
  FeatureExtractor extractor;
  Feature feature = extractor.Extract(sample, target);
  std::vector<float> feature_numbers = feature.ToFixedSizeVector();
  std::vector<float> pred = XgbCostModel::Predict({feature_numbers});
  return pred[0];
}

void ExprCostModel::Train(const std::vector<const ir::ModuleExpr*>& samples,
                          const std::vector<float>& labels,
                          const common::Target& target) {
  trained_times_.store(1);
  size_t total_size = samples.size();
  CHECK_EQ(total_size, labels.size())
      << "Samples must have same size as labels";
  std::vector<std::vector<float>> train_feature_numbers(total_size);
  FeatureExtractor extractor;
  for (size_t i = 0; i < total_size; ++i) {
    CHECK(samples[i] != nullptr) << "Train samples cannot be nullptr";
    Feature feature = extractor.Extract(*samples[i], target);
    train_feature_numbers[i] = feature.ToFixedSizeVector();
  }

  XgbCostModel::Train(train_feature_numbers, labels);
}

void ExprCostModel::Update(const std::vector<const ir::ModuleExpr*>& samples,
                           const std::vector<float>& labels,
                           const common::Target& target) {
  ++trained_times_;
  size_t total_size = samples.size();
  CHECK_EQ(total_size, labels.size())
      << "Samples must have same size as labels";
  std::vector<std::vector<float>> train_feature_numbers(total_size);
  FeatureExtractor extractor;
  for (size_t i = 0; i < total_size; ++i) {
    CHECK(samples[i] != nullptr) << "Train samples cannot be nullptr";
    Feature feature = extractor.Extract(*samples[i], target);
    train_feature_numbers[i] = feature.ToFixedSizeVector();
  }

  XgbCostModel::Update(train_feature_numbers, labels);
}

}  // namespace auto_schedule
}  // namespace cinn
