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

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/platform/init.h"

int main(int argc, char** argv) {
  paddle::memory::allocation::UseAllocatorStrategyGFlag();
  testing::InitGoogleTest(&argc, argv);
  // Because the dynamic library libpaddle_fluid.so clips the symbol table, the
  // external program cannot recognize the flag inside the so, and the flag
  // defined by the external program cannot be accessed inside the so.
  // Therefore, the ParseCommandLine function needs to be called separately
  // inside and outside.
  std::vector<char*> external_argv;
  std::vector<char*> internal_argv;

  // ParseNewCommandLineFlags in gflags.cc starts processing
  // commandline strings from idx 1.
  // The reason is, it assumes that the first one (idx 0) is
  // the filename of executable file.
  external_argv.push_back(argv[0]);
  internal_argv.push_back(argv[0]);

  std::vector<google::CommandLineFlagInfo> all_flags;
  std::vector<std::string> external_flags_name;
  google::GetAllFlags(&all_flags);
  for (size_t i = 0; i < all_flags.size(); ++i) {
    external_flags_name.push_back(all_flags[i].name);
  }

  for (int i = 0; i < argc; ++i) {
    bool flag = true;
    std::string tmp(argv[i]);
    for (size_t j = 0; j < external_flags_name.size(); ++j) {
      if (tmp.find(external_flags_name[j]) != std::string::npos) {
        external_argv.push_back(argv[i]);
        flag = false;
        break;
      }
    }
    if (flag) {
      internal_argv.push_back(argv[i]);
    }
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

  undefok.push_back("use_pinned_memory");
  undefok.push_back("use_mkldnn");
  undefok.push_back("initial_cpu_memory_in_mb");
#endif

  char* env_str = nullptr;
  if (envs.size() > 0) {
    std::string env_string = "--tryfromenv=";
    for (auto t : envs) {
      env_string += t + ",";
    }
    env_string = env_string.substr(0, env_string.length() - 1);
    env_str = strdup(env_string.c_str());
    internal_argv.push_back(env_str);
    VLOG(1) << "gtest env_string:" << env_string;
  }

  char* undefok_str = nullptr;
  if (undefok.size() > 0) {
    std::string undefok_string = "--undefok=";
    for (auto t : undefok) {
      undefok_string += t + ",";
    }
    undefok_string = undefok_string.substr(0, undefok_string.length() - 1);
    undefok_str = strdup(undefok_string.c_str());
    internal_argv.push_back(undefok_str);
    VLOG(1) << "gtest undefok_string:" << undefok_string;
  }

  int new_argc = static_cast<int>(external_argv.size());
  char** external_argv_address = external_argv.data();
  google::ParseCommandLineFlags(&new_argc, &external_argv_address, false);

  int internal_argc = internal_argv.size();
  char** arr = internal_argv.data();
  paddle::platform::ParseCommandLineFlags(internal_argc, arr, true);
  paddle::framework::InitDevices();

  int ret = RUN_ALL_TESTS();

  if (env_str) free(env_str);
  if (undefok_str) free(undefok_str);

  return ret;
}
