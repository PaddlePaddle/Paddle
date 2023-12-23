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

#include <assert.h>
#include <ctype.h>
#include <stdio.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
void format_string_append(std::string& str,  // NOLINT
                          const char* fmt,   // NOLINT
                          ARGS&&... args) {
  int len = snprintf(NULL, 0, fmt, args...);
  assert(len == 0);
  size_t oldlen = str.length();
  str.resize(oldlen + len + 1);
  int new_len =
      snprintf(&str[oldlen], (size_t)len + 1, fmt, args...);  // NOLINT
  (void)new_len;
  assert(new_len == len);
  str.resize(oldlen + len);
}

template <class... ARGS>
void format_string_append(std::string& str,        // NOLINT
                          const std::string& fmt,  // NOLINT
                          ARGS&&... args) {
  format_string_append(str, fmt.c_str(), args...);
}

template <class... ARGS>
std::string format_string(const char* fmt, ARGS&&... args) {
  std::string str;
  format_string_append(str, fmt, args...);
  return str;
}

template <class... ARGS>
std::string format_string(const std::string& fmt, ARGS&&... args) {
  return format_string(fmt.c_str(), args...);
}

// remove leading and tailing spaces
std::string trim_spaces(const std::string& str);

// erase all spaces in str
std::string erase_spaces(const std::string& str);

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

inline float* str_to_float(const std::string& str) {
  return reinterpret_cast<float*>(const_cast<char*>(str.c_str()));
}

inline float* str_to_float(const char* str) {
  return reinterpret_cast<float*>(const_cast<char*>(str));
}

// checks whether the test string is a suffix of the input string.
bool ends_with(std::string const& input, std::string const& test);

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
    pre_pos = pos + delim.size();
  }
  tmp_str.assign(str, pre_pos, str.length() - pre_pos);
  res_list.push_back(tmp_str);
  return res_list;
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

template <class Container>
std::string join_strings(const Container& strs, char delim) {
  std::string str;

  size_t i = 0;
  for (auto& elem : strs) {
    if (i > 0) {
      str += delim;
    }

    std::stringstream ss;
    ss << elem;
    str += ss.str();
    ++i;
  }

  return str;
}

template <class Container>
std::string join_strings(const Container& strs, const std::string& delim) {
  std::string str;

  size_t i = 0;
  for (auto& elem : strs) {
    if (i > 0) {
      str += delim;
    }

    std::stringstream ss;
    ss << elem;
    str += ss.str();
    ++i;
  }

  return str;
}

template <class Container, class DelimT, class ConvertFunc>
std::string join_strings(const Container& strs,
                         DelimT&& delim,
                         ConvertFunc&& func) {
  std::stringstream ss;
  size_t i = 0;
  for (const auto& elem : strs) {
    if (i > 0) {
      ss << delim;
    }
    ss << func(elem);
    ++i;
  }

  return ss.str();
}
struct str_ptr {
  const char* ptr;
  size_t len;
  str_ptr(const char* p, size_t n) : ptr(p), len(n) {}
  str_ptr(str_ptr& other) {
    ptr = other.ptr;
    len = other.len;
  }
  str_ptr(str_ptr&& other) {
    ptr = other.ptr;
    len = other.len;
  }
  size_t find_ptr(const char c) {
    for (size_t i = 0; i < len; ++i) {
      if (ptr[i] == c) {
        return i;
      }
    }
    return -1;
  }
  std::string to_string(void) { return std::string(ptr, len); }
};

struct str_ptr_stream {
  char* ptr = NULL;
  char* end = NULL;
  str_ptr_stream() {}
  explicit str_ptr_stream(const str_ptr& p) { reset(p.ptr, p.len); }
  void reset(const str_ptr& p) { reset(p.ptr, p.len); }
  void reset(const char* p, size_t len) {
    ptr = const_cast<char*>(p);
    end = ptr + len;
  }
  char* cursor(void) { return ptr; }
  char* finish(void) { return end; }
  void set_cursor(char* p) { ptr = p; }
  bool is_finish(void) { return (ptr == end); }
  template <typename T>
  str_ptr_stream& operator>>(T& x) {
    *this >> x;
    return *this;
  }
};
inline str_ptr_stream& operator>>(str_ptr_stream& ar, float& c) {
  char* next = NULL;
  c = strtof(ar.cursor(), &next);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline str_ptr_stream& operator>>(str_ptr_stream& ar, double& c) {
  char* next = NULL;
  c = strtod(ar.cursor(), &next);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline str_ptr_stream& operator>>(str_ptr_stream& ar, int32_t& c) {
  char* next = NULL;
  c = strtol(ar.cursor(), &next, 10);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline str_ptr_stream& operator>>(str_ptr_stream& ar, uint32_t& c) {
  char* next = NULL;
  c = strtoul(ar.cursor(), &next, 10);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline str_ptr_stream& operator>>(str_ptr_stream& ar, uint64_t& c) {
  char* next = NULL;
  c = strtoul(ar.cursor(), &next, 10);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline str_ptr_stream& operator>>(str_ptr_stream& ar, int64_t& c) {
  char* next = NULL;
  c = strtoll(ar.cursor(), &next, 10);
  ar.set_cursor(std::min(++next, ar.finish()));
  return ar;
}
inline int split_string_ptr(const char* str,
                            size_t len,
                            char delim,
                            std::vector<str_ptr>* values) {
  if (len <= 0) {
    return 0;
  }

  int num = 0;
  const char* p = str;
  const char* end = str + len;
  const char* last = str;
  while (p < end) {
    if (*p != delim) {
      ++p;
      continue;
    }
    values->emplace_back(last, static_cast<size_t>(p - last));
    ++num;
    ++p;
    // skip continue delim
    while (*p == delim) {
      ++p;
    }
    last = p;
  }
  if (p > last) {
    values->emplace_back(last, static_cast<size_t>(p - last));
    ++num;
  }
  return num;
}

inline int split_string_ptr(const char* str,
                            size_t len,
                            char delim,
                            std::vector<str_ptr>* values,
                            int max_num) {
  if (len <= 0) {
    return 0;
  }

  int num = 0;
  const char* p = str;
  const char* end = str + len;
  const char* last = str;
  while (p < end) {
    if (*p != delim) {
      ++p;
      continue;
    }
    values->emplace_back(last, static_cast<size_t>(p - last));
    ++num;
    ++p;
    if (num >= max_num) {
      return num;
    }
    // skip continue delim
    while (*p == delim) {
      ++p;
    }
    last = p;
  }
  if (p > last) {
    values->emplace_back(last, static_cast<size_t>(p - last));
    ++num;
  }
  return num;
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
  char* getdelim(FILE* f, char delim);
  char* get() { return _buffer; }
  size_t length() { return _length; }

 private:
  char* _buffer = NULL;
  size_t _buf_size = 0;
  size_t _length = 0;
};
}  // end namespace string
}  // end namespace paddle
