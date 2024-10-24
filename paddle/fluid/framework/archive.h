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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include <glog/logging.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/expect.h"

namespace paddle {
namespace framework {

// not a virtual class
class ArchiveBase {
 protected:
  ArchiveBase() {}

  // Archive is not copyable. But to allow move capture by function objects,
  // check it at runtime rather than at compile time.
  ArchiveBase(const ArchiveBase&) {
    PADDLE_THROW(common::errors::Unavailable(
        "ArchiveBase class does not support copy construction."));
  }

  ArchiveBase(ArchiveBase&& other)
      : buffer_(other.buffer_),
        cursor_(other.cursor_),
        finish_(other.finish_),
        limit_(other.limit_),
        deleter_(std::move(other.deleter_)) {
    other.buffer_ = NULL;
    other.cursor_ = NULL;
    other.finish_ = NULL;
    other.limit_ = NULL;
    other.deleter_ = nullptr;
  }

  ~ArchiveBase() { FreeBuffer(); }

 public:
  ArchiveBase& operator=(const ArchiveBase&) {
    PADDLE_THROW(common::errors::Unavailable(
        "ArchiveBase class does not support assignment construction."));
    return *this;
  }

  ArchiveBase& operator=(ArchiveBase&& other) {
    if (this != &other) {
      FreeBuffer();
      buffer_ = other.buffer_;
      cursor_ = other.cursor_;
      finish_ = other.finish_;
      limit_ = other.limit_;
      deleter_ = std::move(other.deleter_);
      other.buffer_ = NULL;
      other.cursor_ = NULL;
      other.finish_ = NULL;
      other.limit_ = NULL;
      other.deleter_ = nullptr;
    }
    return *this;
  }

  char* Buffer() { return buffer_; }

  void SetReadBuffer(char* buffer,
                     size_t length,
                     std::function<void(char*)>&& deleter) {
    SetBuffer(buffer, length, length, std::move(deleter));
  }

  void SetWriteBuffer(char* buffer,
                      size_t capacity,
                      std::function<void(char*)>&& deleter) {
    SetBuffer(buffer, 0, capacity, std::move(deleter));
  }

  void SetBuffer(char* buffer,
                 size_t length,
                 size_t capacity,
                 std::function<void(char*)>&& deleter) {
    PADDLE_ENFORCE_LE(
        length,
        capacity,
        common::errors::InvalidArgument(
            "Param length should be less than or equal to param capacity, but "
            "the length got %d, the capacity got %d.",
            length,
            capacity));
    FreeBuffer();
    buffer_ = buffer;
    cursor_ = buffer_;
    finish_ = buffer + length;
    limit_ = buffer + capacity;
    deleter_ = std::move(deleter);
  }

  char* Cursor() { return cursor_; }

  void SetCursor(char* cursor) {
    PADDLE_ENFORCE_EQ(
        cursor >= buffer_ && cursor <= finish_,
        true,
        common::errors::InvalidArgument(
            "Param cursor should be greater than or equal to buffer, and "
            "should be less than or equal to finish, but the cursor got %d, "
            "the buffer got %d, the finish got %d.",
            cursor,
            buffer_,
            finish_));
    cursor_ = cursor;
  }

  void AdvanceCursor(size_t offset) {
    PADDLE_ENFORCE_LE(
        offset,
        size_t(finish_ - cursor_),
        common::errors::InvalidArgument(
            "Param offset should be less than or equal to %d, but got %d.",
            size_t(finish_ - cursor_),
            offset));
    cursor_ += offset;
  }

  char* Finish() { return finish_; }

  void SetFinish(char* finish) {
    PADDLE_ENFORCE_EQ(
        finish >= cursor_ && finish <= limit_,
        true,
        common::errors::InvalidArgument(
            "Param finish should be greater than or equal to cursor, and "
            "should be less than or equal to limit, but the finish got %d, "
            "the cursor got %d, the limit got %d.",
            finish,
            cursor_,
            limit_));
    finish_ = finish;
  }

  void AdvanceFinish(size_t offset) {
    PADDLE_ENFORCE_LE(
        offset,
        size_t(limit_ - finish_),
        common::errors::InvalidArgument(
            "Param offset should be less than or equal to %d, but got %d.",
            size_t(limit_ - finish_),
            offset));
    finish_ += offset;
  }

  char* Limit() { return limit_; }

  size_t Position() { return cursor_ - buffer_; }

  size_t Length() { return finish_ - buffer_; }

  size_t Capacity() { return limit_ - buffer_; }

  bool Empty() { return finish_ == buffer_; }

  void Reset() {
    FreeBuffer();
    buffer_ = NULL;
    cursor_ = NULL;
    finish_ = NULL;
    limit_ = NULL;
  }

  void Clear() {
    cursor_ = buffer_;
    finish_ = buffer_;
  }

  char* Release() {
    char* buf = buffer_;
    buffer_ = NULL;
    cursor_ = NULL;
    finish_ = NULL;
    deleter_ = nullptr;
    return buf;
  }

  void Resize(size_t newsize) {
#ifdef _LINUX
    if (unlikely(newsize > Capacity())) {
#else
    if (newsize > Capacity()) {
#endif
      Reserve((std::max)(Capacity() * 2, newsize));
    }
    finish_ = buffer_ + newsize;
    cursor_ = (std::min)(cursor_, finish_);
  }

  void Reserve(size_t newcap) {
    if (newcap > Capacity()) {
      char* newbuf = NULL;
      newbuf = new char[newcap];
      PADDLE_ENFORCE_NE(
          newbuf,
          nullptr,
          common::errors::InvalidArgument("Reserve failed, out of memory."));
      if (Length() > 0) {
        memcpy(newbuf, buffer_, Length());
      }
      cursor_ = newbuf + (cursor_ - buffer_);
      finish_ = newbuf + (finish_ - buffer_);
      limit_ = newbuf + newcap;
      FreeBuffer();
      buffer_ = newbuf;
      deleter_ = std::default_delete<char[]>();
    }
  }

  void PrepareRead(size_t size) {
#ifdef _LINUX
    if (unlikely(!(size <= size_t(finish_ - cursor_)))) {
#else
    if (!(size <= size_t(finish_ - cursor_))) {
#endif
      PADDLE_ENFORCE_LE(
          size,
          size_t(finish_ - cursor_),
          common::errors::InvalidArgument(
              "Param size should be less than or equal to %d, but got %d.",
              size_t(finish_ - cursor_),
              size));
    }
  }

  void PrepareWrite(size_t size) {
#ifdef _LINUX
    if (unlikely(size > size_t(limit_ - finish_))) {
#else
    if (size > size_t(limit_ - finish_)) {
#endif
      Reserve((std::max)(Capacity() * 2, Length() + size));
    }
  }

  void Read(void* data, size_t size) {
    if (size > 0) {
      PrepareRead(size);
      memcpy(data, cursor_, size);
      AdvanceCursor(size);
    }
  }

  void ReadBack(void* data, size_t size) {
    if (size > 0) {
      PADDLE_ENFORCE_LE(
          size,
          size_t(finish_ - cursor_),
          common::errors::InvalidArgument(
              "Param size should be less than or equal to %d, but got %d.",
              size_t(finish_ - cursor_),
              size));
      memcpy(data, finish_ - size, size);
      finish_ -= size;
    }
  }

  void Write(const void* data, size_t size) {
    if (size > 0) {
      PrepareWrite(size);
      memcpy(finish_, data, size);
      AdvanceFinish(size);
    }
  }

  template <class T>
  void GetRaw(T& x) {  // NOLINT
    PrepareRead(sizeof(T));
    memcpy(&x, cursor_, sizeof(T));
    AdvanceCursor(sizeof(T));
  }

  template <class T>
  T GetRaw() {
    T x;
    GetRaw<T>(x);
    return x;
  }

  template <class T>
  void PutRaw(const T& x) {
    PrepareWrite(sizeof(T));
    memcpy(finish_, &x, sizeof(T));
    AdvanceFinish(sizeof(T));
  }

 protected:
  char* buffer_ = NULL;
  char* cursor_ = NULL;
  char* finish_ = NULL;
  char* limit_ = NULL;
  std::function<void(char*)> deleter_ = nullptr;

  void FreeBuffer() {
    if (deleter_) {
      deleter_(buffer_);
    }
    deleter_ = nullptr;
  }
};  // NOLINT

template <class Type>
class Archive {};

class BinaryArchiveType {};

typedef Archive<BinaryArchiveType> BinaryArchive;

template <>
class Archive<BinaryArchiveType> : public ArchiveBase {
 public:
#define ARCHIVE_REPEAT(T)                 \
  BinaryArchive& operator>>(T& x) {       \
    GetRaw(x);                            \
    return *this;                         \
  }                                       \
  BinaryArchive& operator<<(const T& x) { \
    PutRaw(x);                            \
    return *this;                         \
  }

  ARCHIVE_REPEAT(int16_t)
  ARCHIVE_REPEAT(uint16_t)
  ARCHIVE_REPEAT(int32_t)
  ARCHIVE_REPEAT(uint32_t)
  ARCHIVE_REPEAT(int64_t)
  ARCHIVE_REPEAT(uint64_t)
  ARCHIVE_REPEAT(float)
  ARCHIVE_REPEAT(double)
  ARCHIVE_REPEAT(signed char)
  ARCHIVE_REPEAT(unsigned char)
  ARCHIVE_REPEAT(bool)

#undef ARCHIVE_REPEAT

  template <class T>
  T Get() {
    T x;
    *this >> x;
    return x;
  }

  template <class... ARGS>
  void Printf(const char* fmt, ARGS&&... args) {
    size_t temp = Limit() - Finish();
    int len = snprintf(Finish(), temp, fmt, args...);
    PADDLE_ENFORCE_GE(
        len,
        0,
        common::errors::InvalidArgument(
            "Param len should be greater than or equal to 0, but got %d.",
            len));  // NOLINT
    if (static_cast<size_t>(len) >= temp) {
      PrepareWrite(len + 1);
      PADDLE_ENFORCE_EQ(
          snprintf(Finish(), static_cast<size_t>(len) + 1, fmt, args...),
          len,
          common::errors::InvalidArgument(
              "The snprintf(Finish(), static_cast<size_t>(len) + 1, fmt, "
              "args...) should be equal to %d, but got %d.",
              len,
              snprintf(Finish(), static_cast<size_t>(len) + 1, fmt, args...)));
    }
    AdvanceFinish(len);
  }
};

template <class AR, class T, size_t N>
Archive<AR>& operator<<(Archive<AR>& ar, const T (&p)[N]) {
  for (size_t i = 0; i < N; i++) {
    ar << p[i];
  }
  return ar;
}

template <class AR, class T, size_t N>
Archive<AR>& operator>>(Archive<AR>& ar, T (&p)[N]) {
  for (size_t i = 0; i < N; i++) {
    ar >> p[i];
  }
  return ar;
}

template <class AR, class T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::vector<T>& p) {
#ifdef _LINUX
  ar << static_cast<size_t>(p.size());
#else
  ar << (uint64_t)p.size();
#endif
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}

template <class AR, class T>
Archive<AR>& operator>>(Archive<AR>& ar, std::vector<T>& p) {
#ifdef _LINUX
  p.resize(ar.template Get<size_t>());
#else
  p.resize(ar.template Get<uint64_t>());
#endif
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

template <class AR, class T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::valarray<T>& p) {
#ifdef _LINUX
  ar << static_cast<size_t>(p.size());
#else
  ar << (uint64_t)p.size();
#endif
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}

template <class AR, class T>
Archive<AR>& operator>>(Archive<AR>& ar, std::valarray<T>& p) {
#ifdef _LINUX
  p.resize(ar.template Get<size_t>());
#else
  p.resize(ar.template Get<uint64_t>());
#endif
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

inline BinaryArchive& operator<<(BinaryArchive& ar, const std::string& s) {
#ifdef _LINUX
  ar << static_cast<size_t>(s.length());
#else
  ar << (uint64_t)s.length();
#endif
  ar.Write(&s[0], s.length());
  return ar;
}

inline BinaryArchive& operator>>(BinaryArchive& ar, std::string& s) {
#ifdef _LINUX
  size_t len = ar.template Get<size_t>();
#else
  size_t len = ar.template Get<uint64_t>();
#endif
  ar.PrepareRead(len);
  s.assign(ar.Cursor(), len);
  ar.AdvanceCursor(len);
  return ar;
}

template <class AR, class T1, class T2>
Archive<AR>& operator<<(Archive<AR>& ar, const std::pair<T1, T2>& x) {
  return ar << x.first << x.second;
}

template <class AR, class T1, class T2>
Archive<AR>& operator>>(Archive<AR>& ar, std::pair<T1, T2>& x) {  // NOLINT
  return ar >> x.first >> x.second;
}

#ifdef _LINUX
template <class AR, class... T>
Archive<AR>& SerializeTuple(Archive<AR>& ar,                        // NOLINT
                            const std::tuple<T...>& x,              // NOLINT
                            std::integral_constant<size_t, 0> n) {  // NOLINT
  return ar;
}
#else
template <class AR, class... T>
Archive<AR>& SerializeTuple(Archive<AR>& ar,                          // NOLINT
                            const std::tuple<T...>& x,                // NOLINT
                            std::integral_constant<uint64_t, 0> n) {  // NOLINT
  return ar;
}
#endif

#ifdef _LINUX
template <class AR, class... T, size_t N>
Archive<AR>& serialize_tuple(Archive<AR>& ar,                        // NOLINT
                             const std::tuple<T...>& x,              // NOLINT
                             std::integral_constant<size_t, N> n) {  // NOLINT
  return SerializeTuple(ar, x, std::integral_constant<size_t, N - 1>())
         << std::get<N - 1>(x);
}
#else
template <class AR, class... T, uint64_t N>
Archive<AR>& serialize_tuple(Archive<AR>& ar,                          // NOLINT
                             const std::tuple<T...>& x,                // NOLINT
                             std::integral_constant<uint64_t, N> n) {  // NOLINT
  return SerializeTuple(ar, x, std::integral_constant<uint64_t, N - 1>())
         << std::get<N - 1>(x);
}
#endif

#ifdef _LINUX
template <class AR, class... T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::tuple<T...>& x) {
  const size_t size = std::tuple_size<std::tuple<T...>>::value;
  return SerializeTuple(ar, x, std::integral_constant<size_t, size>());
}
#else
template <class AR, class... T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::tuple<T...>& x) {
  const uint64_t size = std::tuple_size<std::tuple<T...>>::value;
  return SerializeTuple(ar, x, std::integral_constant<uint64_t, size>());
}
#endif

#ifdef _LINUX
template <class AR, class... T>
Archive<AR>& DeserializeTuple(const Archive<AR>& ar,
                              std::tuple<T...>& x,  // NOLINT
                              std::integral_constant<size_t, 0> n) {
  return ar;
}
#else
template <class AR, class... T>
Archive<AR>& DeserializeTuple(const Archive<AR>& ar,
                              std::tuple<T...>& x,  // NOLINT
                              std::integral_constant<uint64_t, 0> n) {
  return ar;
}
#endif

#ifdef _LINUX
template <class AR, class... T, size_t N>
Archive<AR>& DeserializeTuple(const Archive<AR>& ar,
                              std::tuple<T...>& x,  // NOLINT
                              std::integral_constant<size_t, N> n) {
  return DeserializeTuple(ar, x, std::integral_constant<size_t, N - 1>()) >>
         std::get<N - 1>(x);
}
#else
template <class AR, class... T, uint64_t N>
Archive<AR>& DeserializeTuple(const Archive<AR>& ar,
                              std::tuple<T...>& x,  // NOLINT
                              std::integral_constant<uint64_t, N> n) {
  return DeserializeTuple(ar, x, std::integral_constant<uint64_t, N - 1>()) >>
         std::get<N - 1>(x);
}
#endif

#ifdef _LINUX
template <class AR, class... T>
Archive<AR>& operator>>(Archive<AR>& ar, std::tuple<T...>& x) {
  const size_t size = std::tuple_size<std::tuple<T...>>::value;
  return DeserializeTuple(ar, x, std::integral_constant<size_t, size>());
}
#else
template <class AR, class... T>
Archive<AR>& operator>>(Archive<AR>& ar, std::tuple<T...>& x) {
  const uint64_t size = std::tuple_size<std::tuple<T...>>::value;
  return DeserializeTuple(ar, x, std::integral_constant<uint64_t, size>());
}
#endif

#ifdef _LINUX
#define ARCHIVE_REPEAT(MAP_TYPE, RESERVE_STATEMENT)                            \
  template <class AR, class KEY, class VALUE, class... ARGS>                   \
  Archive<AR>& operator<<(Archive<AR>& ar,                                     \
                          const MAP_TYPE<KEY, VALUE, ARGS...>& p) {            \
    ar << static_cast<size_t>(p.size());                                       \
    for (auto it = p.begin(); it != p.end(); ++it) {                           \
      ar << *it;                                                               \
    }                                                                          \
    return ar;                                                                 \
  }                                                                            \
  template <class AR, class KEY, class VALUE, class... ARGS>                   \
  Archive<AR>& operator>>(Archive<AR>& ar, MAP_TYPE<KEY, VALUE, ARGS...>& p) { \
    size_t size = ar.template get<size_t>();                                   \
    p.clear();                                                                 \
    RESERVE_STATEMENT;                                                         \
    for (size_t i = 0; i < size; i++) {                                        \
      p.insert(ar.template get<std::pair<KEY, VALUE>>());                      \
    }                                                                          \
    return ar;                                                                 \
  }
#else
#define ARCHIVE_REPEAT(MAP_TYPE, RESERVE_STATEMENT)                            \
  template <class AR, class KEY, class VALUE, class... ARGS>                   \
  Archive<AR>& operator<<(Archive<AR>& ar,                                     \
                          const MAP_TYPE<KEY, VALUE, ARGS...>& p) {            \
    ar << (uint64_t)p.size();                                                  \
    for (auto it = p.begin(); it != p.end(); ++it) {                           \
      ar << *it;                                                               \
    }                                                                          \
    return ar;                                                                 \
  }                                                                            \
  template <class AR, class KEY, class VALUE, class... ARGS>                   \
  Archive<AR>& operator>>(Archive<AR>& ar, MAP_TYPE<KEY, VALUE, ARGS...>& p) { \
    size_t size = ar.template get<uint64_t>();                                 \
    p.clear();                                                                 \
    RESERVE_STATEMENT;                                                         \
    for (size_t i = 0; i < size; i++) {                                        \
      p.insert(ar.template get<std::pair<KEY, VALUE>>());                      \
    }                                                                          \
    return ar;                                                                 \
  }
#endif

ARCHIVE_REPEAT(std::map, )
ARCHIVE_REPEAT(std::multimap, )
ARCHIVE_REPEAT(std::unordered_map, p.reserve(size))
ARCHIVE_REPEAT(std::unordered_multimap, p.reserve(size))

#undef ARCHIVE_REPEAT

#ifdef _LINUX
#define ARCHIVE_REPEAT(SET_TYPE, RESERVE_STATEMENT)                           \
  template <class AR, class KEY, class... ARGS>                               \
  Archive<AR>& operator<<(Archive<AR>& ar, const SET_TYPE<KEY, ARGS...>& p) { \
    ar << static_cast<size_t>(p.size());                                      \
    for (auto it = p.begin(); it != p.end(); ++it) {                          \
      ar << *it;                                                              \
    }                                                                         \
    return ar;                                                                \
  }                                                                           \
  template <class AR, class KEY, class... ARGS>                               \
  Archive<AR>& operator>>(Archive<AR>& ar, SET_TYPE<KEY, ARGS...>& p) {       \
    size_t size = ar.template get<size_t>();                                  \
    p.clear();                                                                \
    RESERVE_STATEMENT;                                                        \
    for (size_t i = 0; i < size; i++) {                                       \
      p.insert(ar.template get<KEY>());                                       \
    }                                                                         \
    return ar;                                                                \
  }
#else
#define ARCHIVE_REPEAT(SET_TYPE, RESERVE_STATEMENT)                           \
  template <class AR, class KEY, class... ARGS>                               \
  Archive<AR>& operator<<(Archive<AR>& ar, const SET_TYPE<KEY, ARGS...>& p) { \
    ar << (uint64_t)p.size();                                                 \
    for (auto it = p.begin(); it != p.end(); ++it) {                          \
      ar << *it;                                                              \
    }                                                                         \
    return ar;                                                                \
  }                                                                           \
  template <class AR, class KEY, class... ARGS>                               \
  Archive<AR>& operator>>(Archive<AR>& ar, SET_TYPE<KEY, ARGS...>& p) {       \
    size_t size = ar.template get<uint64_t>();                                \
    p.clear();                                                                \
    RESERVE_STATEMENT;                                                        \
    for (size_t i = 0; i < size; i++) {                                       \
      p.insert(ar.template get<KEY>());                                       \
    }                                                                         \
    return ar;                                                                \
  }
#endif

ARCHIVE_REPEAT(std::set, )
ARCHIVE_REPEAT(std::multiset, )
ARCHIVE_REPEAT(std::unordered_set, p.reserve(size))
ARCHIVE_REPEAT(std::unordered_multiset, p.reserve(size))

#undef ARCHIVE_REPEAT

}  // namespace framework
}  // namespace paddle
