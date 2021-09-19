/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

using STRING = std::string;
using STRINGS = std::vector<STRING>;
using WSTRING_MAP = std::unordered_map<std::wstring, std::int32_t>;

std::wstring ConvertStrToWstr(const std::string& src);
void ConvertStrToWstr(const std::string& src, std::wstring* res);
std::string ConvertWstrToStr(const std::wstring& src);
void ConvertWstrToStr(const std::wstring& src, std::string* res);
std::string NormalizeNfd(const std::string& s);
void NormalizeNfd(const std::string& s, std::string* ret);
class SerializableStringMap : public std::unordered_map<std::string, int32_t> {
 private:
  void write(std::ostream& os, int32_t t);
  void write(std::ostream& os, const std::string& str);
  void read(std::istream& is, int32_t* token_id);
  std::string read(std::istream& is);

 public:
  void MapTensorToStream(std::ostream& ss);
  void MapTensorFromStream(std::istream& is);
};
}  // namespace framework
}  // namespace paddle
