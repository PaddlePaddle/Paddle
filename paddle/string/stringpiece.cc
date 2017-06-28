/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/strings/stringpiece.h"

#include <string.h>

#include <algorithm>
#include <iosfwd>
#include <stdexcept>

namespace paddle {

StringPiece::StringPiece() : data_(NULL), size_(0) {}

StringPiece::StringPiece(const char* d, size_t n) : data_(d), size_(n) {
  if (d == NULL && n != 0)
    throw std::invalid_argument(
        "StringPiece requires len to be 0 for NULL data");
}

StringPiece::StringPiece(const char* s) : data_(s) {
  size_ = (s == NULL) ? 0 : strlen(s);
}

StringPiece::StringPiece(const std::string& s)
    : data_(s.data()), size_(s.size()) {}

char StringPiece::operator[](size_t n) const {
  if (n >= len())
    throw std::invalid_argument("index out of StringPiece length");
  return data_[n];
}

int Compare(StringPiece a, StringPiece b) {
  const size_t min_len = (a.len() < b.len()) ? a.len() : b.len();
  int r = memcmp(a.data(), b.data(), min_len);
  if (r == 0) {
    if (a.len() < b.len())
      return -1;
    else if (a.len() > b.len())
      return 1;
  }
  return r;
}

bool operator==(StringPiece x, StringPiece y) {
  return ((x.len() == y.len()) &&
          (x.data() == y.data() || memcmp(x.data(), y.data(), x.len()) == 0));
}

bool operator!=(StringPiece x, StringPiece y) { return !(x == y); }

bool operator<(StringPiece x, StringPiece y) { return Compare(x, y) < 0; }
bool operator>(StringPiece x, StringPiece y) { return Compare(x, y) > 0; }

bool operator<=(StringPiece x, StringPiece y) { return Compare(x, y) <= 0; }
bool operator>=(StringPiece x, StringPiece y) { return Compare(x, y) >= 0; }

bool HasPrefix(StringPiece s, StringPiece x) {
  return ((s.len() >= x.len()) && (memcmp(s.data(), x.data(), x.len()) == 0));
}

bool HasSuffix(StringPiece s, StringPiece x) {
  return ((s.len() >= x.len()) &&
          (memcmp(s.data() + (s.len() - x.len()), x.data(), x.len()) == 0));
}

StringPiece SkipPrefix(StringPiece s, size_t n) {
  if (n > s.len())
    throw std::invalid_argument("Skip distance larger than StringPiece length");
  return StringPiece(s.data() + n, s.len() - n);
}

StringPiece SkipSuffix(StringPiece s, size_t n) {
  if (n > s.len())
    throw std::invalid_argument("Skip distance larger than StringPiece length");
  return StringPiece(s.data(), s.len() - n);
}

StringPiece TrimPrefix(StringPiece s, StringPiece x) {
  return HasPrefix(s, x) ? SkipPrefix(s, x.len()) : s;
}

StringPiece TrimSuffix(StringPiece s, StringPiece x) {
  return HasSuffix(s, x) ? SkipSuffix(s, x.len()) : s;
}

bool Contains(StringPiece s, StringPiece sub) {
  return std::search(s.begin(), s.end(), sub.begin(), sub.end()) != s.end();
}

size_t Index(StringPiece s, StringPiece sub) {
  auto e = std::search(s.begin(), s.end(), sub.begin(), sub.end());
  return e != s.end() ? e - s.data() : StringPiece::npos;
}

size_t Find(StringPiece s, char c, size_t pos) {
  if (pos >= s.len()) {
    return StringPiece::npos;
  }
  const char* result =
      reinterpret_cast<const char*>(memchr(s.data() + pos, c, s.len() - pos));
  return result != nullptr ? result - s.data() : StringPiece::npos;
}

size_t RFind(StringPiece s, char c, size_t pos) {
  if (s.len() == 0) return StringPiece::npos;
  for (const char* p = s.data() + std::min(pos, s.len() - 1); p >= s.data();
       p--) {
    if (*p == c) {
      return p - s.data();
    }
  }
  return StringPiece::npos;
}

StringPiece SubStr(StringPiece s, size_t pos, size_t n) {
  if (pos > s.len()) pos = s.len();
  if (n > s.len() - pos) n = s.len() - pos;
  return StringPiece(s.data() + pos, n);
}

std::ostream& operator<<(std::ostream& o, StringPiece piece) {
  return o << piece.ToString();
}

}  // namespace paddle
