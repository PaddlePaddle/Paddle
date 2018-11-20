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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  new_argv.push_back(
      strdup("--tryfromenv=fraction_of_gpu_memory_to_use,allocator_strategy"));
#else
  new_argv.push_back(
      strdup("--tryfromenv=use_pinned_memory,use_mkldnn,initial_cpu_memory_in_"
             "mb,allocator_strategy"));
  new_argv.push_back(strdup("--undefok=use_mkldnn,initial_cpu_memory_in_mb"));
#endif
  int new_argc = static_cast<int>(new_argv.size());
  char** new_argv_address = new_argv.data();
  google::ParseCommandLineFlags(&new_argc, &new_argv_address, false);
  paddle::framework::InitDevices(true);
  return RUN_ALL_TESTS();
}
