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
#include "paddle/memory/memory.h"

int main(int argc, char** argv) {
  std::vector<const char*> new_argv;
  std::string gflags_env;

  new_argv.push_back(argv[0]);
  paddle::memory::Used(paddle::platform::CPUPlace());
#ifdef PADDLE_WITH_CUDA
  paddle::memory::Used(paddle::platform::GPUPlace(0));
  new_argv.push_back(
      "--tryfromenv=fraction_of_gpu_memory_to_use,use_pinned_memory");
#else
  new_argv.push_back("--tryfromenv=use_pinned_memory");
#endif
  int new_argc = 2;
  char** new_argv_address = const_cast<char**>(new_argv.data());
  google::ParseCommandLineFlags(&new_argc, &new_argv_address, true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
