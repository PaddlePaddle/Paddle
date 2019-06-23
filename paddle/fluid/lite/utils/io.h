// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <sys/stat.h>
#include <fstream>
#include <string>
#include "paddle/fluid/lite/utils/cp_logging.h"
#include "paddle/fluid/lite/utils/string.h"

namespace paddle {
namespace lite {

static bool IsFileExists(const std::string& path) {
  std::ifstream file(path);
  bool res = file.is_open();
  if (res) {
    file.close();
  }
  return res;
}

// ARM mobile not support mkdir in C++
static void MkDirRecur(const std::string& path) {
#ifndef LITE_WITH_ARM
  CHECK_EQ(system(string_format("mkdir -p %s", path.c_str()).c_str()), 0)
      << "Cann't mkdir " << path;
#else  // On ARM
  CHECK_NE(mkdir(path.c_str(), S_IRWXU), -1) << "Cann't mkdir " << path;
#endif
}

}  // namespace lite
}  // namespace paddle
