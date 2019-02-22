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

#include <ctype.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <vector>
#include "boost/lexical_cast.hpp"
#include "glog/logging.h"

namespace paddle {
namespace string {

inline size_t count_spaces(const char* s) {
  size_t count = 0;

  while (*s != 0 && isspace(*s++)) {
    count++;
  }

  return count;
}

inline size_t count_nonspaces(const char* s) {
  size_t count = 0;

  while (*s != 0 && !isspace(*s++)) {
    count++;
  }

  return count;
}

template <class... ARGS>
void format_string_append(std::string& str, const char* fmt,  // NOLINT
                          ARGS&&... args) {  // use VA_ARGS may be better ?
  int len = snprintf(NULL, 0, fmt, args...);
  CHECK_GE(len, 0);
  size_t oldlen = str.length();
  str.resize(oldlen + len + 1);
  CHECK(snprintf(&str[oldlen], (size_t)len + 1, fmt, args...) == len);
  str.resize(oldlen + len);
}

template <class... ARGS>
void format_string_append(std::string& str, const std::string& fmt,  // NOLINT
                          ARGS&&... args) {
  format_string_append(str, fmt.c_str(), args...);
}

template <class... ARGS>
std::string format_string(const char* fmt, ARGS&&... args) {
  std::string str;
  format_string_append(str, fmt, args...);
  return std::move(str);
}

template <class... ARGS>
std::string format_string(const std::string& fmt, ARGS&&... args) {
  return format_string(fmt.c_str(), args...);
}

// remove leading and tailing spaces
inline std::string trim_spaces(const std::string& str) {
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

inline int str_to_float(const char* str, float* v) {
  const char* head = str;
  char* cursor = NULL;
  int index = 0;
  while (*(head += count_spaces(head)) != 0) {
    v[index++] = std::strtof(head, &cursor);
    if (head == cursor) {
      break;
    }
    head = cursor;
  }
  return index;
}

// split string by delim
template <class T = std::string>
std::vector<T> split_string(const std::string& str, const std::string& delim) {
  size_t pre_pos = 0;
  size_t pos = 0;
  std::string tmp_str;
  std::vector<T> res_list;
  res_list.clear();
  if (str.empty()) {
    return res_list;
  }

  while ((pos = str.find(delim, pre_pos)) != std::string::npos) {
    tmp_str.assign(str, pre_pos, pos - pre_pos);
    res_list.push_back(tmp_str);
    pre_pos = pos + 1;
  }
  tmp_str.assign(str, pre_pos, str.length() - pre_pos);
  if (!tmp_str.empty()) {
    res_list.push_back(tmp_str);
  }
  return res_list;
  /*
  size_t num = 1;
  const char* p;

  for (p = str.c_str(); *p != 0; p++) {
      if (*p == delim) {
          num++;
      }
  }

  std::vector<T> list(num);
  const char* last = str.c_str();
  num = 0;

  for (p = str.c_str(); *p != 0; p++) {
      if (*p == delim) {
          list[num++] = boost::lexical_cast<T>(last, p - last);
          last = p + 1;
      }
  }

  list[num] = boost::lexical_cast<T>(last, p - last);
  return list;
  */
}

// split string by spaces. Leading and tailing spaces are ignored. Consecutive
// spaces are treated as one delim.
template <class T = std::string>
std::vector<T> split_string(const std::string& str) {
  std::vector<T> list;
  const char* p;
  int pre_pos = 0;
  int pos = 0;
  std::string tmp_str;
  if (str.empty()) {
    return list;
  }
  for (p = str.c_str(); *p != 0;) {
    if (!isspace(*p)) {
      pos = pre_pos;
      p++;

      while (*p != 0 && !isspace(*p)) {
        pos++;
        p++;
      }
      tmp_str.assign(str, pre_pos, pos - pre_pos + 1);
      list.push_back(tmp_str);
      pre_pos = pos + 1;
    } else {
      pre_pos++;
      p++;
    }
  }

  return list;
}

template <class T>
std::string join_strings(const std::vector<T>& strs, char delim) {
  std::string str;

  for (size_t i = 0; i < strs.size(); i++) {
    if (i > 0) {
      str += delim;
    }

    str += boost::lexical_cast<std::string>(strs[i]);
  }

  return str;
}

// A helper class for reading lines from file. A line buffer is maintained. It
// doesn't need to know the maximum possible length of a line.
class LineFileReader {
 public:
  LineFileReader() {}
  LineFileReader(LineFileReader&&) = delete;
  LineFileReader(const LineFileReader&) = delete;
  ~LineFileReader() { ::free(_buffer); }
  char* getline(FILE* f) { return this->getdelim(f, '\n'); }
  char* getdelim(FILE* f, char delim) {
    ssize_t ret = ::getdelim(&_buffer, &_buf_size, delim, f);

    if (ret >= 0) {
      if (ret >= 1 && _buffer[ret - 1] == delim) {
        _buffer[--ret] = 0;
      }

      _length = (size_t)ret;
      return _buffer;
    } else {
      _length = 0;
      CHECK(feof(f));
      return NULL;
    }
  }
  char* get() { return _buffer; }
  size_t length() { return _length; }

 private:
  char* _buffer = NULL;
  size_t _buf_size = 0;
  size_t _length = 0;
};
}  // end namespace string
}  // end namespace paddle
