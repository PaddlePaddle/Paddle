/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <ThreadPool.h>

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/table/common_dense_table.h"
#include "paddle/fluid/distributed/table/common_sparse_table.h"
#include "paddle/fluid/distributed/table/sparse_geo_table.h"
#include "paddle/fluid/distributed/table/table.h"

namespace paddle {
namespace distributed {

// SparseGeoTable + SSUM
TEST(SparseGeoTable, SSUM) {
  int emb_dim = 10;
  int trainers = 2;

  TableParameter table_config;
  table_config.set_table_class("SparseGeoTable");
  FsClientParameter fs_config;
  Table *table = new SparseGeoTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  common_config->set_name("sum");
  common_config->set_table_name("ssum_test_table");
  common_config->set_trainer_num(trainers);
  common_config->add_params("Param");
  common_config->add_dims(emb_dim);
  common_config->add_initializers("fill_constant&1.0");

  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, 0);

  // test push_sparse_param, and create params
  std::vector<uint64_t> init_keys = {0, 1, 2, 3, 4};
  std::vector<float> init_values;
  for (size_t i = 0; i < init_keys.size() * emb_dim; i++) {
    init_values.push_back(0.0);
  }
  table->push_sparse_param(init_keys.data(), init_values.data(),
                           init_keys.size());
  std::vector<float> pull_values(init_values.size());
  table->pull_sparse(pull_values.data(), init_keys.data(), init_keys.size());
  for (size_t i = 0; i < init_keys.size() * emb_dim; i++) {
    ASSERT_TRUE(abs(pull_values[i] - init_values[i]) < 1e-5);
  }

  std::vector<std::vector<uint64_t>> trainer_keys;
  std::vector<std::vector<float>> trainer_values;
  trainer_keys.resize(trainers);
  trainer_values.resize(trainers);
  float start = 0.0;
  for (int i = 0; i < trainers; i++) {
    trainer_keys[i] = init_keys;
    for (size_t j = 0; j < trainer_keys[i].size(); j++) {
      auto id = trainer_keys[i][j];
      for (int k = 0; k < emb_dim; k++) {
        trainer_values[i].push_back(start);
        pull_values[id * emb_dim + k] += start;
        start += 0.1;
      }
    }
  }

  std::shared_ptr<::ThreadPool> pool_ =
      std::make_shared<::ThreadPool>(trainers);
  std::vector<std::future<void>> task_status;
  for (int i = 0; i < trainers; i++) {
    auto &push_keys = trainer_keys[i];
    auto &push_values = trainer_values[i];
    auto task = [table, &push_keys, &push_values] {
      table->push_sparse(push_keys.data(), push_values.data(),
                         push_keys.size());
    };
    task_status.push_back(pool_->enqueue(std::move(task)));
  }
  for (auto &status : task_status) {
    status.wait();
  }

  std::vector<std::vector<uint64_t>> geo_pull_ids;
  std::vector<std::vector<float>> geo_pull_values;
  geo_pull_ids.resize(trainers);
  geo_pull_values.resize(trainers);
  for (int i = 0; i < trainers; i++) {
    table->pull_geo_param(i, &geo_pull_values[i], &geo_pull_ids[i]);
    ASSERT_EQ(geo_pull_values[i].size(), geo_pull_ids[i].size() * emb_dim);
    for (size_t j = 0; j < geo_pull_ids[i].size(); ++j) {
      auto id = geo_pull_ids[i][j];
      for (int k = 0; k < emb_dim; k++) {
        ASSERT_TRUE(abs(geo_pull_values[i][j * emb_dim + k] -
                        pull_values[id * emb_dim + k]) < 1e-5);
      }
    }
  }
}

}  // namespace distributed
}  // namespace paddle
