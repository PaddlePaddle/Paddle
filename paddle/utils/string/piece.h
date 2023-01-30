//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ostream>
#include <string>

namespace paddle {
namespace string {

// Piece points into a std::string object but doesn't own the
// string.  It is for efficient access to strings.  Like Go's string
// type.  Not that Piece doesn't mutate the underlying string,
// so it is thread-safe given that the underlying string doesn't
// change.  Because Piece contains a little data members, and
// its syntax is simple as it doesn't own/manage the string, it is
// cheap to construct Pieces and pass them around.
class Piece {
 public:
  static const size_t npos = static_cast<size_t>(-1);

  // We provide non-explicit singleton constructors so users can
  // pass in a "const char*" or a "string" wherever a "Piece"
  // is expected.  These constructors ensure that if data_ is NULL,
  // size_ is 0.
  Piece();
  Piece(const char* d, size_t n);
  Piece(const char* d);         // NOLINT: accept C string into Piece.
  Piece(const std::string& s);  // NOLINT: accept C++ string into Piece.

  const char* data() const { return data_; }
  size_t len() const { return size_; }

  char operator[](size_t n) const;

  // Piece doesn't own the string, so both iterator and const
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

int Compare(Piece a, Piece b);

bool operator==(Piece x, Piece y);
bool operator!=(Piece x, Piece y);
bool operator<(Piece x, Piece y);
bool operator>(Piece x, Piece y);
bool operator<=(Piece x, Piece y);
bool operator>=(Piece x, Piece y);

bool HasPrefix(Piece s, Piece prefix);
bool HasSuffix(Piece s, Piece suffix);

Piece SkipPrefix(Piece s, size_t n);
Piece SkipSuffix(Piece s, size_t n);

// Skip the prefix (or suffix) if it matches with the string.
Piece TrimPrefix(Piece s, Piece prefix);
Piece TrimSuffix(Piece s, Piece suffix);

// Returns if s contains sub.  Any s except for empty s contains an
// empty sub.
bool Contains(Piece s, Piece sub);

// Return the first occurrence of sub in s, or npos.  If both s and
// sub is empty, it returns npos; otherwise, if only sub is empty, it
// returns 0.
size_t Index(Piece s, Piece sub);

// Return the first occurrence of c in s[pos:end], or npos.
size_t Find(Piece s, char c, size_t pos);

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
size_t RFind(Piece s, char c, size_t pos);

Piece SubStr(Piece s, size_t pos, size_t n);

// allow Piece to be logged
std::ostream& operator<<(std::ostream& o, Piece piece);

}  // namespace string
}  // namespace paddle
