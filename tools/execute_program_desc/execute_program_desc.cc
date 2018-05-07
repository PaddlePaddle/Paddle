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

#include <algorithm>
#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/details/op_registry.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pybind/pybind.h"

DEFINE_string(start_up_proto, "", "start up proto file");
DEFINE_string(loop_proto, "", "loop proto file");
DEFINE_string(executor_device, "CPU", "executor's place:GPU or CPU");
DEFINE_int32(executor_device_id, 0, "GPU device id");

bool read_from_file(const std::string& file, char** buf, int64_t* buf_len) {
  FILE* f = fopen(file.c_str(), "rb");
  if (NULL == f) {
    fprintf(stderr, "open %s error\n", file.c_str());
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

  (*buf)[fsize] = 0;
  fclose(f);
  return true;
}

using namespace paddle;  // NOLINT

std::unique_ptr<framework::ProgramDesc> load_desc(const std::string& file) {
  char* buf = NULL;
  std::unique_ptr<char[]> tmp(buf);
  int64_t buf_len = 0;

  if (!read_from_file(file, &buf, &buf_len)) {
    return NULL;
  }

  framework::proto::ProgramDesc proto;
  if (!proto.ParseFromArray(buf, buf_len)) {
    fprintf(stderr, "parse from %s error!\n", file.c_str());
    return NULL;
  }

  return std::unique_ptr<framework::ProgramDesc>(
      new framework::ProgramDesc(proto));
}

int main(int argc, char** argv) {
  // init.
  google::ParseCommandLineFlags(&argc, &argv, true);
  framework::InitGLOG(argv[0]);
  framework::InitDevices(true);

  // check arguments.
  if (FLAGS_start_up_proto.empty()) {
    fprintf(stderr, "please set start_up_proto's path\n");
    return -1;
  }

  if (FLAGS_loop_proto.empty()) {
    fprintf(stderr, "please set loop_proto's path\n");
    return -1;
  }

  framework::ProgramDesc program;
  framework::Scope scope;

  std::unique_ptr<framework::ProgramDesc> start_up =
      load_desc(FLAGS_start_up_proto);
  std::unique_ptr<framework::ProgramDesc> loop = load_desc(FLAGS_loop_proto);

  std::string place_str = FLAGS_executor_device;
  std::transform(place_str.begin(),
                 place_str.end(),
                 place_str.begin(),
                 [](unsigned char ch) { return toupper(ch); });

  std::unique_ptr<framework::Executor> exe;
  if (place_str == "CPU") {
    platform::CPUPlace place;
    exe.reset(new framework::Executor(place));
  } else if (place_str == "GPU") {
    platform::CUDAPlace place(FLAGS_executor_device_id);
    exe.reset(new framework::Executor(place));
  } else {
    fprintf(stderr, "unkown device:%s\n", FLAGS_executor_device.c_str());
    return -1;
  }

  exe->Run(*start_up, &scope, 0, false, true);
  exe->Run(*loop, &scope, 0, false, true);

  return 0;
}
