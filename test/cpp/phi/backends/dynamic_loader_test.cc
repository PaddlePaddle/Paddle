// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The linked phi library owns backend-specific flag registration.
#undef PADDLE_WITH_FLAGCX
#undef PADDLE_WITH_HIP
#undef PADDLE_WITH_XPU
#include "paddle/phi/backends/dynload/dynamic_loader.cc"  // NOLINT(build/include)

#include <dlfcn.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#ifndef DYNAMIC_LOADER_TEST_DSO_PATH
#error "DYNAMIC_LOADER_TEST_DSO_PATH must point to a loadable shared library"
#endif

namespace phi::dynload {
namespace {

TEST(DynamicLoaderTest, StopsAfterFirstSuccessfulExtraPath) {
  const auto temp_dir =
      std::filesystem::temp_directory_path() /
      ("paddle_dynamic_loader_test_" + std::to_string(getpid()));
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  const std::string dso_name = "paddle_dynamic_loader_test_dso";
  std::filesystem::create_symlink(DYNAMIC_LOADER_TEST_DSO_PATH,
                                  temp_dir / dso_name);

  const auto missing_dir = temp_dir / "missing";
  void* handle = GetDsoHandleFromSearchPath(
      "",
      dso_name,
      false,
      {missing_dir.string(), temp_dir.string(), missing_dir.string()},
      "");

  EXPECT_NE(handle, nullptr);
  if (handle != nullptr) {
    dlclose(handle);
  }
  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace phi::dynload
