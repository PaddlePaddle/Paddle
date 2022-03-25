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
#include "paddle/fluid/distributed/ps/table/common_dense_table.h"
#include "paddle/fluid/distributed/ps/table/common_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/sparse_geo_table.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

// CommonSparseTable + SSGD
TEST(CommonSparseTable, SGD) {
  int emb_dim = 10;
  int trainers = 2;

  TableParameter table_config;
  table_config.set_table_class("CommonSparseTable");
  FsClientParameter fs_config;
  Table *table = new CommonSparseTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  common_config->set_name("sgd");
  common_config->set_table_name("sgd_test_table");
  common_config->set_trainer_num(trainers);
  common_config->add_params("Param");
  common_config->add_dims(emb_dim);
  common_config->add_initializers("uniform_random&0&-1.0&1.0");  // param
  common_config->add_params("LearningRate");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");  // learning_rate
  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, 0);

  // pull parameters for create and check
  std::vector<uint64_t> init_keys = {0, 1, 2, 3, 4};
  std::vector<uint32_t> init_fres = {1, 1, 1, 1, 1};

  std::vector<float> init_values;
  init_values.resize(init_keys.size() * emb_dim);

  std::vector<float> pull_values(init_values.size());
  auto value = PullSparseValue(init_keys, init_fres, emb_dim);
  table->pull_sparse(init_values.data(), value);

  // for check
  std::vector<float> total_gradients;
  total_gradients.resize(init_keys.size() * emb_dim);
  memset(total_gradients.data(), 0, sizeof(float) * total_gradients.size());

  // push gradient
  std::vector<std::vector<uint64_t>> trainer_keys;
  std::vector<std::vector<float>> trainer_gradient_values;
  trainer_keys.resize(trainers);
  trainer_gradient_values.resize(trainers);
  float start = 0.0;
  for (int i = 0; i < trainers; i++) {
    trainer_keys[i] = init_keys;
    for (size_t j = 0; j < trainer_keys[i].size(); j++) {
      auto id = trainer_keys[i][j];
      for (int k = 0; k < emb_dim; k++) {
        trainer_gradient_values[i].push_back(start);
        total_gradients[id * emb_dim + k] += start;
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
  pull_values.resize(init_keys.size() * emb_dim);
  table->pull_sparse(init_values.data(), value);

  for (size_t i = 0; i < init_values.size(); ++i) {
    auto update_val = init_values[i] - 1.0 * total_gradients[i];
    ASSERT_TRUE(abs(update_val - pull_values[i]) < 1e-5);
  }
}

// CommonSparseTable + Adam
TEST(CommonSparseTable, Adam) {
  int emb_dim = 10;
  int trainers = 2;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1.0e-8;

  TableParameter table_config;
  table_config.set_table_class("CommonSparseTable");
  FsClientParameter fs_config;
  Table *table = new CommonSparseTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  common_config->set_name("adam");
  common_config->set_table_name("adam_test_table");
  common_config->set_trainer_num(trainers);
  common_config->add_params("Param");
  common_config->add_dims(emb_dim);
  common_config->add_initializers("uniform_random&0&-1.0&1.0");
  common_config->add_params("LearningRate");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");
  common_config->add_params("Moment1");
  common_config->add_dims(emb_dim);
  common_config->add_initializers("fill_constant&0.0");
  common_config->add_params("Moment2");
  common_config->add_dims(emb_dim);
  common_config->add_initializers("fill_constant&0.0");
  common_config->add_params("Beta1Pow");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");
  common_config->add_params("Beta2Pow");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");
  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, 0);

  // pull parameters for create and check
  std::vector<uint64_t> init_keys = {0, 1, 2, 3, 4};
  std::vector<uint32_t> init_fres = {1, 1, 1, 1, 1};

  std::vector<float> init_values;
  init_values.resize(init_keys.size() * emb_dim);

  auto value = PullSparseValue(init_keys, init_fres, emb_dim);
  table->pull_sparse(init_values.data(), value);

  // push gradient
  std::vector<std::vector<uint64_t>> trainer_keys;
  std::vector<std::vector<float>> trainer_gradient_values;
  trainer_keys.resize(trainers);
  trainer_gradient_values.resize(trainers);
  float start = 0.0;
  for (int i = 0; i < trainers; i++) {
    trainer_keys[i] = init_keys;
    for (size_t j = 0; j < trainer_keys[i].size(); j++) {
      for (int k = 0; k < emb_dim; k++) {
        trainer_gradient_values[i].push_back(start);
        start += 0.1;
      }
    }
  }

  for (int i = 0; i < trainers; i++) {
    auto &push_keys = trainer_keys[i];
    auto &push_values = trainer_gradient_values[i];
    table->push_sparse(push_keys.data(), push_values.data(), push_keys.size());
  }

  std::vector<float> pull_values;
  pull_values.resize(init_keys.size() * emb_dim);
  table->pull_sparse(pull_values.data(), init_keys.data(), init_keys.size());

  for (size_t idx = 0; idx < init_keys.size(); idx += emb_dim) {
    std::vector<float> beta1_pow, beta2_pow, lr, mom1, mom2, param;
    beta1_pow.push_back(beta1);
    beta2_pow.push_back(beta2);
    lr.push_back(1.0);
    for (int i = 0; i < emb_dim; i++) {
      mom1.push_back(0.0);
      mom2.push_back(0.0);
      param.push_back(init_values[idx + i]);
    }
    for (int i = 0; i < trainers; i++) {
      auto lr_ = lr[0] * sqrt(1 - beta2_pow[0]) / (1 - beta1_pow[0]);
      for (int j = 0; j < emb_dim; j++) {
        mom1[j] =
            beta1 * mom1[j] + (1 - beta1) * trainer_gradient_values[i][idx + j];
        mom2[j] = beta2 * mom2[j] +
                  (1 - beta2) * trainer_gradient_values[i][idx + j] *
                      trainer_gradient_values[i][idx + j];
        param[j] = param[j] -
                   lr_ * (mom1[j] /
                          (sqrt(mom2[j]) + epsilon * sqrt(1 - beta2_pow[0])));
      }
      beta1_pow[0] *= beta1;
      beta2_pow[0] *= beta2;
    }
    for (int i = 0; i < emb_dim; i++) {
      ASSERT_TRUE(abs(param[i] - pull_values[idx + i]) < 1e-5);
    }
  }
}

}  // namespace distributed
}  // namespace paddle
