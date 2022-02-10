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
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

TEST(MemorySparseTable, SGD) {
  int emb_dim = 8;
  int trainers = 2;

  TableParameter table_config;
  table_config.set_table_class("MemorySparseTable");
  table_config.set_shard_num(10);
  FsClientParameter fs_config;
  Table *table = new MemorySparseTable();
  table->set_shard(0, 1);

  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CtrCommonAccessor");
  accessor_config->set_fea_dim(11);
  accessor_config->set_embedx_dim(8);
  accessor_config->set_embedx_threshold(5);
  accessor_config->mutable_ctr_accessor_param()->set_nonclk_coeff(0.2);
  accessor_config->mutable_ctr_accessor_param()->set_click_coeff(1);
  accessor_config->mutable_ctr_accessor_param()->set_base_threshold(0.5);
  accessor_config->mutable_ctr_accessor_param()->set_delta_threshold(0.2);
  accessor_config->mutable_ctr_accessor_param()->set_delta_keep_days(16);
  accessor_config->mutable_ctr_accessor_param()->set_show_click_decay_rate(
      0.99);

  accessor_config->mutable_embed_sgd_param()->set_name("SparseNaiveSGDRule");
  auto *naive_param =
      accessor_config->mutable_embed_sgd_param()->mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);

  accessor_config->mutable_embedx_sgd_param()->set_name("SparseNaiveSGDRule");
  naive_param = accessor_config->mutable_embedx_sgd_param()->mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);

  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, 0);

  // pull parameters for create and check
  std::vector<uint64_t> init_keys = {0, 1, 2, 3, 4};
  std::vector<uint32_t> init_fres = {1, 1, 1, 1, 1};

  std::vector<float> init_values;
  init_values.resize(init_keys.size() * (emb_dim + 1));
  auto value = PullSparseValue(init_keys, init_fres, emb_dim);
  table->pull_sparse(init_values.data(), value);

  // for check
  std::vector<float> total_gradients;
  total_gradients.resize(init_keys.size() * (4 + emb_dim));
  memset(total_gradients.data(), 0, sizeof(float) * total_gradients.size());

  // push gradient
  std::vector<std::vector<uint64_t>> trainer_keys;
  std::vector<std::vector<float>> trainer_gradient_values;
  trainer_keys.resize(trainers);
  trainer_gradient_values.resize(trainers);
  float start = 0.0;
  for (int i = 0; i < trainers; i++) {
    start = 0.0;
    trainer_keys[i] = init_keys;
    for (size_t j = 0; j < trainer_keys[i].size(); j++) {
      auto id = trainer_keys[i][j];
      for (int k = 0; k < emb_dim + 4; k++) {
        trainer_gradient_values[i].push_back(start);
        total_gradients[id * (emb_dim + 4) + k] += start;
        start += 0.1;
      }
    }
  }

  std::shared_ptr<::ThreadPool> pool_ =
      std::make_shared<::ThreadPool>(trainers);
  std::vector<std::future<void>> task_status;
  for (int i = 0; i < trainers; i++) {
    auto &push_keys = trainer_keys[i];
    auto &push_values = trainer_gradient_values[i];
    auto task = [table, &push_keys, &push_values] {
      table->push_sparse(push_keys.data(), push_values.data(),
                         push_keys.size());
    };
    task_status.push_back(pool_->enqueue(std::move(task)));
  }
  for (auto &status : task_status) {
    status.wait();
  }

  std::vector<float> pull_values;
  pull_values.resize(init_keys.size() * (emb_dim + 1));
  table->pull_sparse(pull_values.data(), value);

  for (size_t i = 0; i < init_keys.size(); ++i) {
    for (size_t j = 0; j < emb_dim + 1; ++j) {
      auto update_val = init_values[i * (emb_dim + 1) + j] -
                        0.1 * total_gradients[3 + i * (emb_dim + 4) + j];
      VLOG(3) << total_gradients[i * (emb_dim + 4) + j + 3] << ":"
              << init_values[i * (emb_dim + 1) + j];
      VLOG(3) << update_val << ": " << pull_values[i * (emb_dim + 1) + j];
    }
  }

  MemorySparseTable *ctr_table = dynamic_cast<MemorySparseTable *>(table);
  ctr_table->save_local_fs("./work/table.save", "0", "test");
}

}  // namespace distributed
}  // namespace paddle
