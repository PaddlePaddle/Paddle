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

std::string ConvertWstrToStr(const std::wstring& src) {
  return kConverter.to_bytes(src);
}

std::string NormalizeNfd(const std::string& s) {
  std::string ret;
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    ret = std::string(result);
    free(result);
    result = nullptr;
  }
  return ret;
}

void WstringMapToStream(std::ostream& os,
                        const std::unordered_map<std::string, int32_t>& data) {
  {
    // firstly write the data size.
    size_t t = data.size();
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  }
  {
    // then write the data
    for (auto it = data.begin(); it != data.end(); ++it) {
      std::string token = it->first;
      int32_t token_id = it->second;
      // write the token
      size_t length = token.size();
      os.write(reinterpret_cast<const char*>(&length), sizeof(length));
      os.write(token.c_str(), length);
      // write the token_id
      os.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
    }
  }
}

void WstringMapFromStream(std::istream& is,
                          std::unordered_map<std::string, int32_t>* data) {
  // first read the map size
  size_t map_size;
  is.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
  data->reserve(map_size);
  // then read the data
  for (size_t i = 0; i < map_size; ++i) {
    // read the token
    size_t token_length;
    is.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));
    char* tmp = new char[token_length];
    is.read(tmp, token_length);
    std::string token(tmp, tmp + token_length);
    // read the token_id
    int32_t token_id;
    is.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));

    data->emplace(token, token_id);
  }
}

}  // namespace framework
}  // namespace paddle
