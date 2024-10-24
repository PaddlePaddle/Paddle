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

#include "paddle/utils/string/string_helper.h"

#include <cctype>
#include <cstdio>

#include <cstring>
#include <string>

namespace paddle::string {

// remove leading and tailing spaces
std::string trim_spaces(const std::string& str) {
  const char* p = str.c_str();

  while (*p != 0 && isspace(*p)) {
    p++;
  }

  size_t len = strlen(p);

  while (len > 0 && isspace(p[len - 1])) {
    len--;
  }

  return std::string(p, len);
}

std::string erase_spaces(const std::string& str) {
  std::string result;
  result.reserve(str.size());
  const char* p = str.c_str();
  while (*p != 0) {
    if (!isspace(*p)) {
      result.append(p, 1);
    }
    ++p;
  }
  return result;
}

bool ends_with(std::string const& input, std::string const& test) {
  if (test.size() > input.size()) return false;
  return std::equal(test.rbegin(), test.rend(), input.rbegin());
}

// A helper class for reading lines from file.
// A line buffer is maintained. It
// doesn't need to know the maximum possible length of a line.
char* LineFileReader::getdelim(FILE* f, char delim) {
#ifndef _WIN32
  int32_t ret =
      static_cast<int32_t>(::getdelim(&_buffer, &_buf_size, delim, f));

  if (ret >= 0) {
    if (ret >= 1 && _buffer[ret - 1] == delim) {
      _buffer[--ret] = 0;
    }

    _length = static_cast<size_t>(ret);
    return _buffer;
  } else {
    _length = 0;
    int code = feof(f);
    (void)code;
    assert(code);
    return nullptr;
  }
#else
  return NULL;
#endif
}

}  // namespace paddle::string
