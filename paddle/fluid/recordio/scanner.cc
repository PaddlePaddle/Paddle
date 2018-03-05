//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/recordio/chunk.h"

#include <glob.h>  // glob

namespace paddle {
namespace recordio {

Scanner::Scanner(const char* paths)
    : cur_file_(nullptr), path_idx_(0), end_(false) {
  glob_t glob_result;
  glob(paths, GLOB_TILDE, NULL, &glob_result);

  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    paths_.emplace_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
}

bool Scanner::Scan() {
  if (err_ == -1 || end_ == true) {
    return false;
  }
  if (cur_scanner_ == nullptr) {
    if (!NextFile()) {
      end_ = true;
      return false;
    }
    if (err_ == -1) {
      return false;
    }
  }
  if (!cur_scanner_->Scan()) {
    if (err_ == -1) {
      return false;
    }
  }

  return true;
}

bool Scanner::NextFile() {}

}  // namespace recordio
}  // namespace paddle
