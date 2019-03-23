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

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#include <memory>
#include <string>
#include <utility>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

inline bool& shell_verbose_internal() {
  static bool x = false;
  return x;
}

inline bool shell_verbose() { return shell_verbose_internal(); }

inline void shell_set_verbose(bool x) { shell_verbose_internal() = x; }

extern std::shared_ptr<FILE> shell_fopen(const std::string& path,
                                         const std::string& mode);

extern std::shared_ptr<FILE> shell_popen(const std::string& cmd,
                                         const std::string& mode, int* err_no);

extern std::pair<std::shared_ptr<FILE>, std::shared_ptr<FILE>> shell_p2open(
    const std::string& cmd);

inline void shell_execute(const std::string& cmd) {
  int err_no = 0;
  do {
    err_no = 0;
    shell_popen(cmd, "w", &err_no);
  } while (err_no == -1);
}

extern std::string shell_get_command_output(const std::string& cmd);

}  // namespace framework
}  // namespace paddle
