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

#pragma once

#include <string>
#include <ostream>

namespace paddle {

// StringPiece points into a std::string object but doesn't own the
// string.  It is for efficient access to strings.  Like Go's string
// type.  Not that StringPiece doesn't mutate the underlying string,
// so it is thread-safe given that the underlying string doesn't
// change.  Because StringPiece contains a little data members, and
// its syntax is simple as it doesn't own/manage the string, it is
// cheap to construct StringPieces and pass them around.
class StringPiece {
public:
  static const size_t npos = static_cast<size_t>(-1);

  // We provide non-explicit singleton constructors so users can
  // pass in a "const char*" or a "string" wherever a "StringPiece"
  // is expected.  These contructors ensure that if data_ is NULL,
  // size_ is 0.
  StringPiece();
  StringPiece(const char* d, size_t n);
  StringPiece(const char* d);
  StringPiece(const std::string& s);

  const char* data() const { return data_; }
  size_t len() const { return size_; }

  char operator[](size_t n) const;

  // StringPiece doesn't own the string, so both iterator and const
  // iterator are const char* indeed.
  typedef const char* const_iterator;
  typedef const char* iterator;
  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  // Return a string that contains the copy of the referenced data.
  std::string ToString() const { return std::string(data_, size_); }

private:
  const char* data_;
  size_t size_;

  // Intentionally copyable
};

int Compare(StringPiece a, StringPiece b);

bool operator==(StringPiece x, StringPiece y);
bool operator!=(StringPiece x, StringPiece y);
bool operator<(StringPiece x, StringPiece y);
bool operator>(StringPiece x, StringPiece y);
bool operator<=(StringPiece x, StringPiece y);
bool operator>=(StringPiece x, StringPiece y);

bool HasPrefix(StringPiece s, StringPiece prefix);
bool HasSuffix(StringPiece s, StringPiece suffix);

StringPiece SkipPrefix(StringPiece s, size_t n);
StringPiece SkipSuffix(StringPiece s, size_t n);

// Skip the prefix (or suffix) if it matches with the string.
StringPiece TrimPrefix(StringPiece s, StringPiece prefix);
StringPiece TrimSuffix(StringPiece s, StringPiece suffix);

// Returns if s contains sub.  Any s except for empty s contains an
// empty sub.
bool Contains(StringPiece s, StringPiece sub);

// Return the first occurrence of sub in s, or npos.  If both s and
// sub is empty, it returns npos; otherwise, if only sub is empty, it
// returns 0.
size_t Index(StringPiece s, StringPiece sub);

// Return the first occurrence of c in s[pos:end], or npos.
size_t Find(StringPiece s, char c, size_t pos);

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
size_t RFind(StringPiece s, char c, size_t pos);

StringPiece SubStr(StringPiece s, size_t pos, size_t n);

// allow StringPiece to be logged
std::ostream& operator<<(std::ostream& o, StringPiece piece);

}  // namespace paddle
