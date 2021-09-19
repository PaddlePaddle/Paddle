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

#include <utf8proc.h>

#include "glog/logging.h"
#include "paddle/fluid/framework/string_array.h"

namespace paddle {
namespace framework {

using STRING = std::string;
using STRINGS = std::vector<STRING>;
using WSTRING_MAP = std::unordered_map<std::wstring, std::int32_t>;

std::wstring_convert<std::codecvt_utf8<wchar_t>> kConverter;

std::wstring ConvertStrToWstr(const std::string& src) {
  return kConverter.from_bytes(src);
}

void ConvertStrToWstr(const std::string& src, std::wstring* res) {
  *res = kConverter.from_bytes(src);
}

// void ConvertStrToWstr(const std::string& src, std::wstring* tgt) {
//   *tgt = kConverter.from_bytes(src);
// }

std::string ConvertWstrToStr(const std::wstring& src) {
  return kConverter.to_bytes(src);
}

void ConvertWstrToStr(const std::wstring& src, std::string* res) {
  *res = kConverter.to_bytes(src);
}

std::string NormalizeNfd(const std::string& s) {
  std::string ret;
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    ret = std::string(result);
    free(result);
  }
  return ret;
}

void NormalizeNfd(const std::string& s, std::string* ret) {
  *ret = "";
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    *ret = std::string(result);
    free(result);
  }
}

void SerializableStringMap::write(std::ostream& os, int32_t t) {
  os.write(reinterpret_cast<const char*>(&t), sizeof(t));
}

void SerializableStringMap::write(std::ostream& os, const std::string& str) {
  size_t length = str.size();
  os.write(reinterpret_cast<const char*>(&length), sizeof(length));
  os.write(str.c_str(), length);
}

void SerializableStringMap::read(std::istream& is, int32_t* token_id) {
  is.read(reinterpret_cast<char*>(token_id), sizeof(*token_id));
}

std::string SerializableStringMap::read(std::istream& is) {
  size_t length;
  is.read(reinterpret_cast<char*>(&length), sizeof(length));
  char* tmp = new char[length];
  is.read(tmp, length);
  std::string s(tmp, tmp + length);
  return s;
}

void SerializableStringMap::MapTensorToStream(std::ostream& ss) {
  size_t t = this->size();
  ss.write(reinterpret_cast<const char*>(&t), sizeof(t));
  for (auto it = this->begin(); it != this->end(); ++it) {
    std::string str = it->first;
    int32_t value = it->second;
    write(ss, str);
    write(ss, value);
  }
}

void SerializableStringMap::MapTensorFromStream(std::istream& is) {
  size_t map_size;
  is.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
  for (size_t i = 0; i < map_size; ++i) {
    std::string key = read(is);
    int32_t value;
    read(is, &value);
    (*this)[key] = value;
  }
}

}  // namespace framework
}  // namespace paddle
