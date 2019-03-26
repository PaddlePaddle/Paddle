/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstring>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/init.h"

int main(int argc, char** argv) {
  paddle::memory::allocation::UseAllocatorStrategyGFlag();
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> new_argv;
  std::string gflags_env;
  for (int i = 0; i < argc; ++i) {
    new_argv.push_back(argv[i]);
  }

  std::vector<std::string> envs;
  std::vector<std::string> undefok;
#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_GRPC)
  std::string str_max_body_size;
  if (google::GetCommandLineOption("max_body_size", &str_max_body_size)) {
    setenv("FLAGS_max_body_size", "2147483647", 1);
    envs.push_back("max_body_size");
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  envs.push_back("fraction_of_gpu_memory_to_use");
  envs.push_back("initial_gpu_memory_in_mb");
  envs.push_back("reallocate_gpu_memory_in_mb");
  envs.push_back("allocator_strategy");
#elif __clang__
  envs.push_back("use_mkldnn");
  envs.push_back("initial_cpu_memory_in_mb");
  envs.push_back("allocator_strategy");

  undefok.push_back("use_mkldnn");
  undefok.push_back("initial_cpu_memory_in_mb");
#else
  envs.push_back("use_pinned_memory");
  envs.push_back("use_mkldnn");
  envs.push_back("initial_cpu_memory_in_mb");
  envs.push_back("allocator_strategy");

  undefok.push_back("use_mkldnn");
  undefok.push_back("initial_cpu_memory_in_mb");
#endif

  if (envs.size() > 0) {
    std::string env_string = "--tryfromenv=";
    for (auto t : envs) {
      env_string += t + ",";
    }
    env_string = env_string.substr(0, env_string.length() - 1);
    new_argv.push_back(strdup(env_string.c_str()));
    VLOG(1) << "gtest env_string:" << env_string;
  }

  if (undefok.size() > 0) {
    std::string undefok_string = "--undefok=";
    for (auto t : undefok) {
      undefok_string += t + ",";
    }
    undefok_string = undefok_string.substr(0, undefok_string.length() - 1);
    new_argv.push_back(strdup(undefok_string.c_str()));
    VLOG(1) << "gtest undefok_string:" << undefok_string;
  }

  int new_argc = static_cast<int>(new_argv.size());
  char** new_argv_address = new_argv.data();
  google::ParseCommandLineFlags(&new_argc, &new_argv_address, false);
  paddle::framework::InitDevices(true);
  return RUN_ALL_TESTS();
}
