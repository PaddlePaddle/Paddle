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
#include <unordered_map>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

TEST(BarrierTable, Barrier) {
  int emb_dim = 10;
  int trainers = 2;
  bool sync = true;

  TableParameter table_config;
  table_config.set_table_class("BarrierTable");
  FsClientParameter fs_config;
  Table *table = new BarrierTable();
  TableAccessorParameter *accessor_config = table_config.mutable_accessor();
  accessor_config->set_accessor_class("CommMergeAccessor");
  CommonAccessorParameter *common_config = table_config.mutable_common();
  common_config->set_table_name("barrier_table");
  common_config->set_trainer_num(trainers);
  common_config->set_sync(sync);

  auto ret = table->initialize(table_config, fs_config);

  std::unordered_map<uint32_t, std::shared_ptr<Table>> maps =
      std::unordered_map<uint32_t, std::shared_ptr<Table>>();

  table->set_table_map(&maps);

  std::shared_ptr<::ThreadPool> pool_ =
      std::make_shared<::ThreadPool>(trainers);
  std::vector<std::future<void>> task_status;

  for (auto x = 0; x < trainers; x++) {
    auto task = [table, x] { table->barrier(x, 0); };
    task_status.push_back(pool_->enqueue(std::move(task)));
  }

  for (auto &status : task_status) {
    status.wait();
  }

  ASSERT_EQ(ret, 0);
}

}  // namespace distributed
}  // namespace paddle
