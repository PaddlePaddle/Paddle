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
#include "paddle/fluid/distributed/ps/table/common_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/depends/large_scale_kv.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

TEST(BENCHMARK, LargeScaleKV) {
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
}

}  // namespace distributed
}  // namespace paddle
