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

#include "paddle/phi/core/vocab/string_array.h"
#include <utf8proc.h>
#include <exception>
#include "glog/logging.h"

namespace phi {

std::wstring_convert<std::codecvt_utf8<wchar_t>> kConverter;

// Convert the std::string type to the std::wstring type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res) {
  try {
    *res = kConverter.from_bytes(src);
  } catch (std::range_error& e) {
    VLOG(3) << "The string " << src << " was converted to unicode failedly! ";
    return false;
  }
  return true;
}

// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res) {
  *res = kConverter.to_bytes(src);
}

// Normalization Form Canonical Decomposition.
void NFD(const std::string& s, std::string* ret) {
  *ret = "";
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    *ret = std::string(result);
    free(result);  // NOLINT
  }
}

// Write the data which is type of
// std::unordered_map<std::string, int32_t> to ostream.
void StringMapToStream(std::ostream& os,
                       const std::unordered_map<std::string, int32_t>& data) {
  {
    // firstly write the data size.
    size_t t = data.size();
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  }
  {
    // then write the data
    for (const auto& item : data) {
      std::string token = item.first;
      int32_t token_id = item.second;
      // write the token
      size_t length = token.size();
      os.write(reinterpret_cast<const char*>(&length), sizeof(length));
      os.write(token.c_str(), length);  // NOLINT
      // write the token_id
      os.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
    }
  }
}

// Read the data which is type of
// std::unordered_map<td::string, int32_t> from istream.
void StringMapFromStream(std::istream& is,
                         std::unordered_map<std::string, int32_t>* data) {
  // first read the map size
  size_t map_size = 0;
  is.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
  data->reserve(map_size);
  // then read the data
  for (size_t i = 0; i < map_size; ++i) {
    // read the token
    size_t token_length = 0;
    is.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));
    char* tmp = new char[token_length];
    is.read(tmp, token_length);  // NOLINT
    std::string token(tmp, tmp + token_length);
    delete[] tmp;
    // read the token_id
    int32_t token_id = 0;
    is.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));

    data->emplace(token, token_id);
  }
}

}  // namespace phi
