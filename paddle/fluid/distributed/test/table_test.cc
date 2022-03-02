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

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/table/common_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/sparse_geo_table.h"

namespace paddle {
namespace distributed {

TEST(Table, Initialize) {
  TableParameter table_config;
  table_config.set_table_class("SparseGeoTable");
  FsClientParameter fs_config;
  // case 1. no accessor
  Table *table = new SparseGeoTable();
  auto ret = table->initialize(table_config, fs_config);
  ASSERT_EQ(ret, -1);
}
}  // namespace distributed
}  // // namespace paddle
