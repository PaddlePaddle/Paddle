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

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/backends/device_manager.h"

namespace phi {
namespace {

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string path_template = "/tmp/paddle_list_all_libraries_XXXXXX";
    std::vector<char> writable_path(path_template.begin(), path_template.end());
    writable_path.push_back('\0');
    const char* created_path = mkdtemp(writable_path.data());
    if (created_path == nullptr) {
      throw std::runtime_error("Failed to create temporary directory");
    }
    path_ = created_path;
  }

  ~ScopedTempDir() {
    for (const auto& file : files_) {
      std::remove(file.c_str());
    }
    rmdir(path_.c_str());
  }

  const std::string& path() const { return path_; }

  std::string CreateFile(const std::string& filename) {
    const auto file_path = path_ + "/" + filename;
    std::ofstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create temporary file");
    }
    files_.push_back(file_path);
    return file_path;
  }

 private:
  std::string path_;
  std::vector<std::string> files_;
};

TEST(ListAllLibraries, FiltersByPlatformLibrarySuffix) {
  ScopedTempDir temp_dir;
#if defined(__APPLE__)
  const std::string suffix = ".dylib";
  const std::string other_suffix = ".so";
#else
  const std::string suffix = ".so";
  const std::string other_suffix = ".dylib";
#endif

  const auto first_library = temp_dir.CreateFile("libfirst" + suffix);
  const auto second_library = temp_dir.CreateFile("libsecond" + suffix);
  temp_dir.CreateFile("libversioned" + suffix + ".1");
  temp_dir.CreateFile("libbackup" + suffix + ".bak");
  temp_dir.CreateFile("libother" + other_suffix);
  temp_dir.CreateFile("README.txt");

  auto libraries = ListAllLibraries(temp_dir.path());
  std::sort(libraries.begin(), libraries.end());
  std::vector<std::string> expected = {first_library, second_library};
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(libraries, expected);
}

TEST(ListAllLibraries, ReturnsEmptyForMissingDirectory) {
  ScopedTempDir temp_dir;
  EXPECT_TRUE(ListAllLibraries(temp_dir.path() + "/missing").empty());
}

}  // namespace
}  // namespace phi
