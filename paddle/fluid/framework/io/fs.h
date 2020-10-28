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

#pragma once

#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

int fs_select_internal(const std::string& path);

// localfs
extern size_t localfs_buffer_size();

extern void localfs_set_buffer_size(size_t x);

extern std::shared_ptr<FILE> localfs_open_read(std::string path,
                                               const std::string& converter);

extern std::shared_ptr<FILE> localfs_open_write(std::string path,
                                                const std::string& converter);

extern int64_t localfs_file_size(const std::string& path);

extern void localfs_remove(const std::string& path);

extern std::vector<std::string> localfs_list(const std::string& path);

extern std::string localfs_tail(const std::string& path);

extern bool localfs_exists(const std::string& path);

extern void localfs_mkdir(const std::string& path);

extern void localfs_mv(const std::string& src, const std::string& dest);

// hdfs
extern size_t hdfs_buffer_size();

extern void hdfs_set_buffer_size(size_t x);

extern const std::string& hdfs_command();

extern void hdfs_set_command(const std::string& x);

extern const std::string& download_cmd();

extern void set_download_command(const std::string& x);

extern std::shared_ptr<FILE> hdfs_open_read(std::string path, int* err_no,
                                            const std::string& converter);

extern std::shared_ptr<FILE> hdfs_open_write(std::string path, int* err_no,
                                             const std::string& converter);

extern void hdfs_remove(const std::string& path);

extern std::vector<std::string> hdfs_list(const std::string& path);

extern std::string hdfs_tail(const std::string& path);

extern bool hdfs_exists(const std::string& path);

extern void hdfs_mkdir(const std::string& path);

extern void hdfs_mv(const std::string& src, const std::string& dest);

// aut-detect fs
extern std::shared_ptr<FILE> fs_open_read(const std::string& path, int* err_no,
                                          const std::string& converter);

extern std::shared_ptr<FILE> fs_open_write(const std::string& path, int* err_no,
                                           const std::string& converter);

extern std::shared_ptr<FILE> fs_open(const std::string& path,
                                     const std::string& mode, int* err_no,
                                     const std::string& converter = "");

extern int64_t fs_file_size(const std::string& path);

extern void fs_remove(const std::string& path);

extern std::vector<std::string> fs_list(const std::string& path);

extern std::string fs_tail(const std::string& path);

extern bool fs_exists(const std::string& path);

extern void fs_mkdir(const std::string& path);

extern void fs_mv(const std::string& src, const std::string& dest);

}  // namespace framework
}  // namespace paddle
