// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/io/shell.h"

namespace framework = paddle::framework;

TEST(FS, mv) {
  std::ofstream out("src.txt");
  out.close();
  framework::fs_mv("src.txt", "dest.txt");
  framework::hdfs_mv("", "");
  try {
    framework::hdfs_mv("afs:/none", "afs:/none");
  } catch (...) {
    VLOG(3) << "test hdfs_mv, catch errors";
  }
}
