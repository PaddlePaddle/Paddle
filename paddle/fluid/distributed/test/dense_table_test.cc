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

// CommonDenseTable + Adam
TEST(CommonDenseTable, Adam) {
  int fea_dim = 10;
  int trainers = 2;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1.0e-8;

  TableParameter table_config;
  table_config.set_table_class("CommonDenseTable");
  FsClientParameter fs_config;
  Table *table = new CommonDenseTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  // set adam optimize config
  common_config->set_name("adam");
  common_config->set_table_name("adam_test_table");
  common_config->set_trainer_num(trainers);
  common_config->add_params("Param");
  common_config->add_dims(fea_dim);
  common_config->add_initializers("gaussian_random&0&0.0&1.0");
  common_config->add_params("LearningRate");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");
  common_config->add_params("Moment1");
  common_config->add_dims(fea_dim);
  common_config->add_initializers("fill_constant&0.0");
  common_config->add_params("Moment2");
  common_config->add_dims(fea_dim);
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
  std::vector<float> init_values;
  init_values.resize(fea_dim);
  table->pull_dense(init_values.data(), fea_dim);

  // push gradient
  std::vector<std::vector<float>> trainer_gradient_values;
  trainer_gradient_values.resize(trainers);
  float start = 10.0;
  for (int i = 0; i < trainers; i++) {
    for (int k = 0; k < fea_dim; k++) {
      trainer_gradient_values[i].push_back(start);
      start += 0.1;
    }
  }

  // for adam
  for (int i = 0; i < trainers; i++) {
    auto &push_values = trainer_gradient_values[i];
    table->push_dense(push_values.data(), push_values.size());
  }

  std::vector<float> pull_values;
  pull_values.resize(fea_dim);
  table->pull_dense(pull_values.data(), fea_dim);

  std::vector<float> beta1_pow, beta2_pow, lr, mom1, mom2, param;
  beta1_pow.push_back(beta1);
  beta2_pow.push_back(beta2);
  lr.push_back(1.0);
  for (int i = 0; i < fea_dim; i++) {
    mom1.push_back(0.0);
    mom2.push_back(0.0);
    param.push_back(init_values[i]);
  }

  for (int i = 0; i < trainers; i++) {
    auto lr_ = lr[0] * sqrt(1 - beta2_pow[0]) / (1 - beta1_pow[0]);
    for (int j = 0; j < fea_dim; j++) {
      mom1[j] = beta1 * mom1[j] + (1 - beta1) * trainer_gradient_values[i][j];
      mom2[j] = beta2 * mom2[j] +
                (1 - beta2) * trainer_gradient_values[i][j] *
                    trainer_gradient_values[i][j];
      param[j] =
          param[j] -
          lr_ * (mom1[j] / (sqrt(mom2[j]) + epsilon * sqrt(1 - beta2_pow[0])));
    }
    beta1_pow[0] *= beta1;
    beta2_pow[0] *= beta2;
  }
  for (int j = 0; j < fea_dim; j++) {
    ASSERT_TRUE(abs(param[j] - pull_values[j]) < 1e-6);
  }
}

// CommonDenseTable + Adam
TEST(CommonDenseTable, SGD) {
  int fea_dim = 10;
  int trainers = 2;

  TableParameter table_config;
  table_config.set_table_class("CommonDenseTable");
  FsClientParameter fs_config;
  Table *table = new CommonDenseTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  common_config->set_name("sgd");
  common_config->set_table_name("sgd_test_table");
  common_config->set_trainer_num(trainers);
  common_config->add_params("Param");
  common_config->add_dims(fea_dim);
  common_config->add_initializers("gaussian_random&0&0.0&1.0");
  common_config->add_params("LearningRate");
  common_config->add_dims(1);
  common_config->add_initializers("fill_constant&1.0");
  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, 0);

  // pull parameters for create and check
  std::vector<float> init_values;
  init_values.resize(fea_dim);
  table->pull_dense(init_values.data(), fea_dim);

  std::vector<float> total_gradients;
  total_gradients.resize(fea_dim);
  memset(total_gradients.data(), 0, sizeof(float) * total_gradients.size());
  // push gradient
  std::vector<std::vector<float>> trainer_gradient_values;
  trainer_gradient_values.resize(trainers);
  float start = 10.0;
  for (int i = 0; i < trainers; i++) {
    for (int k = 0; k < fea_dim; k++) {
      trainer_gradient_values[i].push_back(start);
      total_gradients[k] += start;
      start += 0.1;
    }
  }

  std::shared_ptr<::ThreadPool> pool_ =
      std::make_shared<::ThreadPool>(trainers);
  std::vector<std::future<void>> task_status;
  for (int i = 0; i < trainers; i++) {
    auto &push_values = trainer_gradient_values[i];
    auto task = [table, &push_values] {
      table->push_dense(push_values.data(), push_values.size());
    };
    task_status.push_back(pool_->enqueue(std::move(task)));
  }
  for (auto &status : task_status) {
    status.wait();
  }

  std::vector<float> pull_values;
  pull_values.resize(fea_dim);
  table->pull_dense(pull_values.data(), fea_dim);
  for (int j = 0; j < fea_dim; j++) {
    auto update_val = init_values[j] - 1.0 * total_gradients[j];
    ASSERT_TRUE(abs(update_val - pull_values[j]) < 1e-5);
  }
}

}  // namespace distributed
}  // namespace paddle
