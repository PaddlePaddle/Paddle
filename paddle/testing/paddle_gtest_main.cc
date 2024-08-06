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

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/init_default_kernel_signature_map.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/memory/allocation/allocator_strategy.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
COMMON_DECLARE_bool(enable_gpu_memory_usage_log);
#endif

int main(int argc, char** argv) {  // NOLINT
  paddle::memory::allocation::UseAllocatorStrategyGFlag();
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> new_argv;
  new_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    new_argv.push_back(argv[i]);
  }

  std::vector<std::string> envs;
#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_PSLIB)
  if (paddle::flags::FindFlag("max_body_size")) {
    setenv("FLAGS_max_body_size", "2147483647", 1);
    envs.emplace_back("max_body_size");
  }
#endif

  const auto& flag_map = phi::GetExportedFlagInfoMap();
  for (const auto& pair : flag_map) {
    const std::string& name = pair.second.name;
    if (pair.second.is_writable) {  // means public
      envs.push_back(name);
    }
  }

  char* env_str = nullptr;
  if (!envs.empty()) {
    std::string env_string = "--tryfromenv=";
    for (auto const& t : envs) {
      env_string += t + ",";
    }
    env_string = env_string.substr(0, env_string.length() - 1);
    env_str = strdup(env_string.c_str());
    new_argv.push_back(env_str);
    VLOG(1) << "gtest env_string:" << env_string;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (strstr(env_str, "enable_gpu_memory_usage_log")) {  // NOLINT
    VLOG(1) << "Set FLAGS_enable_gpu_memory_usage_log to true";
    FLAGS_enable_gpu_memory_usage_log = true;
  }
#endif

  int new_argc = static_cast<int>(new_argv.size());
  char** new_argv_address = new_argv.data();
  paddle::flags::AllowUndefinedFlags();
  paddle::flags::ParseCommandLineFlags(&new_argc, &new_argv_address);
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  paddle::framework::InitDefaultKernelSignatureMap();

  int ret = RUN_ALL_TESTS();

  if (env_str) free(env_str);  // NOLINT
  return ret;
}
