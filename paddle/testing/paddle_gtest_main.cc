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
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/flags.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
DECLARE_bool(enable_gpu_memory_usage_log);
#endif

int main(int argc, char** argv) {
  paddle::memory::allocation::UseAllocatorStrategyGFlag();
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> new_argv;
  for (int i = 0; i < argc; ++i) {
    new_argv.push_back(argv[i]);
  }

  std::vector<std::string> envs;
  std::vector<std::string> undefok;
#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_PSLIB)
  std::string str_max_body_size;
  if (::GFLAGS_NAMESPACE::GetCommandLineOption("max_body_size",
                                               &str_max_body_size)) {
    setenv("FLAGS_max_body_size", "2147483647", 1);
    envs.push_back("max_body_size");
  }
#endif

  const auto& flag_map = phi::GetExportedFlagInfoMap();
  for (const auto& pair : flag_map) {
    const std::string& name = pair.second.name;
    // NOTE(zhiqiu): some names may not linked in some tests, so add to
    // `undefok`.
    // One way to handle that is to check each flag item by item, and put it in
    // `envs` or `undefok`;
    // another way is to add all flags to `envs` and `undeok`, basically it is
    // not a good design,
    // but it can simplify the procedure of creating new flag and seems no side
    // effects.
    // see details: https://gflags.github.io/gflags/#special
    if (pair.second.is_writable) {  // means public
      envs.push_back(name);
      undefok.push_back(name);
    }
  }

  char* env_str = nullptr;
  if (envs.size() > 0) {
    std::string env_string = "--tryfromenv=";
    for (auto t : envs) {
      env_string += t + ",";
    }
    env_string = env_string.substr(0, env_string.length() - 1);
    env_str = strdup(env_string.c_str());
    new_argv.push_back(env_str);
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
    new_argv.push_back(undefok_str);
    VLOG(1) << "gtest undefok_string:" << undefok_string;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (strstr(undefok_str, "enable_gpu_memory_usage_log")) {
    VLOG(1) << "Set FLAGS_enable_gpu_memory_usage_log to true";
    FLAGS_enable_gpu_memory_usage_log = true;
  }
#endif

  int new_argc = static_cast<int>(new_argv.size());
  char** new_argv_address = new_argv.data();
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(
      &new_argc, &new_argv_address, false);
  paddle::framework::InitDevices();
  paddle::framework::InitDefaultKernelSignatureMap();

  int ret = RUN_ALL_TESTS();

#ifdef PADDLE_WITH_ASCEND_CL
  paddle::platform::AclInstance::Instance().Finalize();
#endif
  if (env_str) free(env_str);
  if (undefok_str) free(undefok_str);
  return ret;
}
