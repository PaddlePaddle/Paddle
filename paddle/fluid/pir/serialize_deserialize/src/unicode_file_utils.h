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
#include <string>

#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

namespace pir {

#ifdef _WIN32
inline std::wstring Utf8ToWidePath(const std::string& file_path) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(file_path);
}
#endif

inline std::ifstream OpenInputFile(const std::string& file_path,
                                   std::ios::openmode mode = std::ios::in) {
#ifdef _WIN32
  return std::ifstream(Utf8ToWidePath(file_path), mode);
#else
  return std::ifstream(file_path, mode);
#endif
}

inline std::ofstream OpenOutputFile(const std::string& file_path,
                                    std::ios::openmode mode = std::ios::out) {
#ifdef _WIN32
  return std::ofstream(Utf8ToWidePath(file_path), mode);
#else
  return std::ofstream(file_path, mode);
#endif
}

}  // namespace pir
