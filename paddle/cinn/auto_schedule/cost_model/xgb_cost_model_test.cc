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

#include "paddle/cinn/auto_schedule/cost_model/xgb_cost_model.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace cinn {
namespace auto_schedule {

TEST(CostModel, Basic) {
  XgbCostModel cost_model;

  srand(time(NULL));

  int batch_size = 16;
  int feature_size = 8;
  std::vector<float> labels(batch_size, 1.0);
  std::vector<std::vector<float>> samples(batch_size,
                                          std::vector<float>(feature_size));
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < feature_size; ++j) {
      samples[i][j] = rand() % 10;  // NOLINT
    }
  }

  cost_model.Train(samples, labels);
  std::vector<float> pred = cost_model.Predict(samples);

  std::string path = "./test_cost_model.cpp_save_model";
  cost_model.Save(path);

  XgbCostModel load_cost_model;
  load_cost_model.Load(path);
  std::vector<float> load_pred = cost_model.Predict(samples);

  ASSERT_EQ(pred.size(), load_pred.size());
  for (size_t i = 0; i < pred.size(); ++i) {
    ASSERT_FLOAT_EQ(pred[i], load_pred[i]);
    VLOG(6) << "pred[" << i << "] = " << pred[i];
  }
  std::remove(path.c_str());

  cost_model.Update(samples, labels);
  pred = cost_model.Predict(samples);
  for (size_t i = 0; i < pred.size(); ++i) {
    VLOG(6) << "pred[" << i << "] = " << pred[i];
  }
}

}  // namespace auto_schedule
}  // namespace cinn
