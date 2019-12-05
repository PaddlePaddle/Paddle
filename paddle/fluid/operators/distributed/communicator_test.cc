//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "paddle/fluid/operators/distributed/communicator.h"

namespace paddle {
namespace operators {
namespace distributed {

using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

TEST(communicator, merge_lod_tensors) {
  auto cpu_place = platform::CPUPlace();
  auto dims = framework::make_ddim({2, 3});
  std::vector<std::shared_ptr<framework::Variable>> in_vars;
  float out_value = 0;
  for (auto i = 0; i < 10; ++i) {
    auto var = std::make_shared<Variable>();
    in_vars.emplace_back(var);
    auto *tensor = var->GetMutable<LoDTensor>();
    auto *data = tensor->mutable_data<float>(dims, cpu_place);
    for (auto j = 0; j < tensor->numel(); ++j) {
      data[j] = static_cast<float>(i);
    }
    out_value += static_cast<float>(i);
  }
  const std::string out_name = "Out";
  std::unique_ptr<framework::Scope> scope;
  scope.reset(new framework::Scope());
  scope->Var(out_name);
  for (auto i = 0; i < 10; ++i) {
    MergeVars<float>(out_name, in_vars, scope.get());
  }
  auto &out_tensor = scope->FindVar(out_name)->Get<LoDTensor>();
  auto *out_data = out_tensor.data<float>();
  ASSERT_EQ(out_tensor.dims(), dims);
  for (auto i = 0; i < out_tensor.numel(); ++i) {
    ASSERT_EQ(out_data[i], out_value);
  }
}

TEST(communicator, merge_selected_rows) {
  auto cpu_place = platform::CPUPlace();
  int64_t width = 10;
  std::vector<std::shared_ptr<framework::Variable>> in_vars;
  const int64_t height = 100;
  for (auto i = 0; i < 10; ++i) {
    std::vector<int64_t> rows;
    for (auto k = 0; k <= i; ++k) {
      rows.push_back(k);
    }
    auto var = std::make_shared<Variable>();
    in_vars.emplace_back(var);
    auto *slr = var->GetMutable<SelectedRows>();
    slr->set_height(height);
    slr->set_rows(rows);
    auto dims =
        framework::make_ddim({static_cast<int64_t>(rows.size()), width});
    auto *data = slr->mutable_value()->mutable_data<float>(dims, cpu_place);
    for (size_t i = 0; i < rows.size(); ++i) {
      for (auto j = 0; j < width; ++j) {
        data[i * width + j] = static_cast<float>(rows[i]);
      }
    }
  }
  const std::string out_name = "Out";
  std::unique_ptr<framework::Scope> scope;
  scope.reset(new framework::Scope());
  scope->Var(out_name);
  for (auto i = 0; i < 10; ++i) {
    MergeVars<float>(out_name, in_vars, scope.get());
  }
  auto &out_slr = scope->FindVar(out_name)->Get<SelectedRows>();
  auto &out_t = out_slr.value();
  auto *out_data = out_t.data<float>();
  ASSERT_EQ(out_t.dims(), framework::make_ddim({10, width}));
  std::vector<float> out_values;
  out_values.reserve(10);
  for (auto i = 0; i < 10; ++i) {
    out_values.push_back(static_cast<float>(i * (10 - i)));
  }
  for (size_t i = 0; i < out_slr.rows().size(); ++i) {
    ASSERT_EQ(out_slr.rows()[i], static_cast<int>(i));
    for (auto j = 0; j < width; ++j) {
      ASSERT_EQ(out_data[i * width + j], out_values[i]);
    }
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
