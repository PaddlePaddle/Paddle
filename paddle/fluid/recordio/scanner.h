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

#pragma once

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Scanner

class Scanner {
public:
  Scanner(const char* paths);
  const std::string Record();
  bool Scan();
  void Close();

private:
  bool NextFile();
  int Err() { return err_; }

private:
  std::vector<std::string> paths_;
  FILE* cur_file_;
  RangeScanner* cur_scanner_;
  int path_idx_;
  bool end_;
  int err_;
};
