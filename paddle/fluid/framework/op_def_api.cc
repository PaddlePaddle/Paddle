// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#include "paddle/fluid/framework/op_def_api.h"
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#ifdef _LINUX
#include <stdio_ext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "glog/logging.h"
#include "io/fs.h"
#include "paddle/fluid/framework/op_def.pb.h"

namespace paddle {
namespace framework {

const proto::OpDef& GetOpDef(const std::string& op_name) {
  static std::unordered_map<std::string, proto::OpDef> ops_definition;
  static std::mutex mtx;
  if (ops_definition.find(op_name) == ops_definition.end()) {
    std::lock_guard<std::mutex> lk(mtx);
    if (ops_definition.find(op_name) == ops_definition.end()) {
      proto::OpDef op_def;
      std::string op_path = OP_DEF_FOLDER + op_name + ".pbtxt";
      int fd = open(op_path.c_str(), O_RDONLY);
      if (fd == -1) {
        LOG(WARNING) << op_path << " open failed!";
      } else {
        ::google::protobuf::io::FileInputStream* input =
            new ::google::protobuf::io::FileInputStream(fd);
        if (!::google::protobuf::TextFormat::Parse(input, &op_def)) {
          LOG(WARNING) << "Failed to parse " << op_path;
        }
        delete input;
        close(fd);
      }
      ops_definition.emplace(std::make_pair(op_name, std::move(op_def)));
    }
  }
  return ops_definition.at(op_name);
}
}  // namespace framework
}  // namespace paddle
