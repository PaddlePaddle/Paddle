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

#include <fstream>

#include "paddle/fluid/framework/io/fs.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

TEST(FS, mv) {
#ifdef _LINUX
  std::ofstream out("src.txt");
  out.close();
  paddle::framework::fs_mv("src.txt", "dest.txt");
  paddle::framework::hdfs_mv("", "");
  paddle::framework::localfs_mv("", "");
  try {
    paddle::framework::hdfs_mv("afs:/none", "afs:/none");
  } catch (...) {
    VLOG(3) << "test hdfs_mv, catch expected errors of unknown path";
  }
  try {
    paddle::framework::fs_mv("afs:/none", "afs:/none");
  } catch (...) {
    VLOG(3) << "test hdfs_mv, catch expected errors of unknown path";
  }
  try {
    paddle::framework::hdfs_mv("unknown:/none", "unknown:/none");
  } catch (...) {
    VLOG(3) << "test hdfs_mv, catch expected errors of unknown prefix";
  }

  try {
    paddle::framework::dataset_hdfs_set_command(
        "hadoop -D hadoop.job.ugi=anotherxxx fs -text");
    int err_no = 0;
    paddle::framework::hdfs_open_read("afs:/none.gz", &err_no, "", true);
    paddle::framework::hdfs_open_read("afs:/none.gz", &err_no, "", false);
    paddle::framework::hdfs_open_read("afs:/none", &err_no, "", true);
    paddle::framework::hdfs_open_read("afs:/none", &err_no, "", false);
  } catch (...) {
    VLOG(3) << "test hdfs_open_read, catch expected errors of unknown path";
  }

#endif
}
