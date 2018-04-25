//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <stdlib.h>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pybind/pybind.h"

bool read_from_file(const std::string& file, char** buf, int64_t* buf_len) {
  FILE* f = fopen(file.c_str(), "rb");
  if (NULL == f) {
    printf("open %s error\n", file.c_str());
    return false;
  }

  fseek(f, 0, SEEK_END);
  int64_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  // same as rewind(f);

  *buf = static_cast<char*>(malloc(fsize + 1));
  if (fread(*buf, fsize, 1, f) != 1) {
    fclose(f);
    return false;
  }

  *buf_len = fsize;

  fclose(f);
  (*buf)[fsize] = 0;
  return true;
}

using namespace paddle;  // NOLINT

framework::ProgramDesc* load_desc(const std::string& file) {
  char* buf = NULL;
  std::unique_ptr<char[]> tmp(buf);
  int64_t buf_len = 0;

  if (!read_from_file(file, &buf, &buf_len)) {
    return NULL;
  }

  framework::proto::ProgramDesc proto;
  if (!proto.ParseFromArray(buf, buf_len)) {
    printf("parse from %s error!\n", file.c_str());
    return NULL;
  }

  return (new framework::ProgramDesc(proto));
}

DEFINE_string(start_up_proto, "", "start up proto file");

DEFINE_string(loop_proto, "", "loop proto file");

int main(int argc, char** argv) {
  // init.
  google::ParseCommandLineFlags(&argc, &argv, true);
  framework::InitGLOG(argv[0]);
  framework::InitDevices(true);

  // check arguments.
  if (FLAGS_start_up_proto.empty()) {
    printf("please set start_up_proto path\n");
    return -1;
  }

  if (FLAGS_loop_proto.empty()) {
    printf("please set loop_proto path\n");
    return -1;
  }

  framework::ProgramDesc program;
  framework::Scope scope;

  platform::CPUPlace place;
  framework::Executor exe(place);

  // platform::CPUDeviceContext* ctx = platform::DeviceContextPool()Get(place);

  framework::ProgramDesc* start_up = load_desc(FLAGS_start_up_proto);
  framework::ProgramDesc* loop = load_desc(FLAGS_loop_proto);

  exe.Run(*start_up, &scope, 0, false, true);
  exe.Run(*loop, &scope, 0, false, true);

  delete start_up;
  delete loop;
  return 0;
}
