// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <streambuf>
#include <string>

#include "paddle/ap/include/adt/adt.h"

namespace ap::fs {

inline bool FileExists(const std::string& filepath) {
  std::fstream fp;
  fp.open(filepath, std::fstream::in);
  if (fp.is_open()) {
    fp.close();
    return true;
  } else {
    return false;
  }
}

// reference:
// https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
inline adt::Result<adt::Ok> ReadFileContent(const std::string& file_path,
                                            std::string* content) {
  std::ifstream ifs(file_path);

  ADT_CHECK(ifs.is_open()) << adt::errors::RuntimeError{
      std::string() + "file open failed. file_path: " + file_path};

  ifs.seekg(0, std::ios::end);
  content->reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);

  content->assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  return adt::Ok{};
}

inline adt::Result<adt::Ok> WriteFileContent(const std::string& file_path,
                                             const std::string& content) {
  std::ofstream ofs{file_path};
  ADT_CHECK(ofs.is_open()) << adt::errors::RuntimeError{
      std::string() + "file open failed. file_path: " + file_path};
  ofs << content;
  ofs.close();
  return adt::Ok{};
}

}  // namespace ap::fs
