// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace utils {

//! Get the content of a stream.
template <typename T>
std::string GetStreamCnt(const T& x);

/**
 * Construct a formatted string with arguments.
 * @param fmt_str The format.
 * @param ... The parameters of the format.
 * @return The formatted string.
 */
std::string StringFormat(const std::string& fmt_str, ...);

/**
 * Delete the '_outer' and '_inner' suffix of a var's name.
 * @param name The input string.
 * @return The edited string.
 */
std::string RemoveSuffix(const std::string& name);

/**
 * Join multiple fields to a single string. Similar to Python's str.join method.
 */
template <typename T = std::string>
std::string Join(const std::vector<T>& fields, const std::string& splitter) {
  if (fields.empty()) return "";
  std::stringstream ss;
  for (int i = 0; i < fields.size() - 1; i++) ss << fields[i] << splitter;
  ss << fields.back();
  return ss.str();
}

std::vector<std::string> Split(const std::string& str,
                               const std::string& splitter);

std::string Trim(const std::string& s, const char* empty = " \n\r\t");

//! Convert a string to its uppercase.
std::string Uppercase(const std::string& x);

//! Replace a substr 'from' to 'to' in string s.
void Replace(std::string* s, const std::string& from, const std::string& to);

//! Count how many times substr 'sub' appears in string s.
size_t Count(std::string* s, const std::string& sub);

//! Tell if a char is prefix of a tensor's name.
bool IsPrefix(const char& c);

//! Tell if a char is suffix of a tensor's name.
bool IsSuffix(const char& c);

//! Tell if a string \p x start with \p str.
bool StartsWith(const std::string& x, const std::string& str);

//! Tell if a string \p x ends with \p str.
bool EndsWith(const std::string& x, const std::string& str);

template <typename T>
std::string GetStreamCnt(const T& x) {
  std::stringstream os;
  os << x;
  return os.str();
}

std::string TransValidVarName(std::string name);

std::string Attribute2String(const utils::Attribute& attr);

}  // namespace utils
}  // namespace cinn
