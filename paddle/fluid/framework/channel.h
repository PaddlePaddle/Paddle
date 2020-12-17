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
#include <condition_variable>  // NOLINT
#include <deque>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>
#include "paddle/fluid/framework/expect.h"

namespace paddle {
namespace framework {

template <class T>
class ChannelObject {
 public:
  ChannelObject() {}

  // capacity can be zero
  explicit ChannelObject(size_t capacity) {
    capacity_ = (std::min)(MaxCapacity(), capacity);
  }

  const std::deque<T>& GetData() const { return data_; }
  void Clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.clear();
    data_.shrink_to_fit();
  }

  size_t Capacity() {
    return capacity_;  // atomic
  }

  void SetCapacity(size_t x) {  // capacity can be zero
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = std::min(MaxCapacity(), x);
    Notify();
  }

  size_t BlockSize() {
    return block_size_;  // atomic
  }

  void SetBlockSize(size_t x) {
    CHECK(x >= 1) << "block size must be >= 1";
    std::lock_guard<std::mutex> lock(mutex_);
    block_size_ = x;
  }

  template <class U>
  void InheritFrom(const std::shared_ptr<ChannelObject<U>>& other) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = other->Capacity();
    block_size_ = other->BlockSize();
  }

  bool Closed() {
    return closed_;  // atomic
  }

  // open channel, then data can be write() to channel
  void Open() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = false;
    Notify();
  }

  // close channel, then no more data can be write() to channel
  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    Notify();
  }

  size_t Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.size();
  }

  bool Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return EmptyUnlocked();
  }

  // blocking operation
  bool Get(T& val) { return Read(1, &val) != 0; }  // NOLINT

  // blocking operation
  // returns 0 if the channel is closed and empty
  size_t Read(size_t n, T* p) {
    if (n == 0) {
      return 0;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    size_t finished = Read(n, p, lock);
    Notify();
    return finished;
  }

  // blocking operation
  bool Put(T&& val) { return WriteMove(1, &val) != 0; }

  // blocking operation
  bool Put(const T& val) { return Write(1, &val) != 0; }

  // blocking operation
  // returns value less than n if the channel is closed
  size_t Write(size_t n, const T* p) {
    if (n == 0) {
      return 0;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    size_t finished = Write(n, p, lock);
    Notify();
    return finished;
  }

  // WriteMove() will clear original contents of input array
  size_t WriteMove(size_t n, T* p) {
    if (n == 0) {
      return 0;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    size_t finished = WriteMove(n, p, lock);
    Notify();
    return finished;
  }

  // read data of block size from channel to vector
  size_t Read(std::vector<T>& p) {  // NOLINT
    p.resize(block_size_);
    size_t finished = Read(p.size(), &p[0]);
    p.resize(finished);
    return finished;
  }

  size_t ReadAll(std::vector<T>& p) {  // NOLINT
    p.clear();
    size_t finished = 0;
    size_t n = 0;
    do {
      // _block_size may change anytime
      n = block_size_;
      p.resize(finished + n);
      n = Read(n, &p[finished]);
      finished += n;
    } while (n != 0);
    p.resize(finished);
    return finished;
  }

  // write data from vector to channel
  size_t Write(const std::vector<T>& p) { return Write(p.size(), &p[0]); }

  // write data from vector to channel
  size_t Write(std::vector<T>&& p) { return WriteMove(p.size(), &p[0]); }

 private:
  size_t capacity_ = MaxCapacity();
  size_t block_size_ = 1024;
  bool closed_ = false;
  std::mutex mutex_;
  // use deque to store data
  std::deque<T> data_;
  size_t reading_count_ = 0;
  int empty_waiters_ = 0;
  int full_waiters_ = 0;
  std::condition_variable empty_cond_;
  std::condition_variable full_cond_;

  static constexpr size_t MaxCapacity() {
    return (std::numeric_limits<size_t>::max)() / 2;
  }

  void Notify() {
    if (empty_waiters_ != 0 && (!EmptyUnlocked() || closed_)) {
      empty_cond_.notify_one();
    }
    if (full_waiters_ != 0 && (!FullUnlocked() || closed_)) {
      full_cond_.notify_one();
    }
  }

  bool EmptyUnlocked() { return data_.empty(); }

  bool FullUnlocked() { return data_.size() >= capacity_ + reading_count_; }

  bool WaitForRead(std::unique_lock<std::mutex>& lock) {  // NOLINT
#ifdef _LINUX
    while (unlikely(EmptyUnlocked() && !closed_)) {
#else
    while (EmptyUnlocked() && !closed_) {
#endif
      if (full_waiters_ != 0) {
        full_cond_.notify_one();
      }
      empty_waiters_++;
      empty_cond_.wait(lock);
      empty_waiters_--;
    }
    return !EmptyUnlocked();
  }

  bool WaitForWrite(std::unique_lock<std::mutex>& lock) {  // NOLINT
#ifdef _LINUX
    while (unlikely(FullUnlocked() && !closed_)) {
#else
    while (FullUnlocked() && !closed_) {
#endif
      if (empty_waiters_ != 0) {
        empty_cond_.notify_one();
      }
      full_waiters_++;
      full_cond_.wait(lock);
      full_waiters_--;
    }
    return !closed_;
  }

  size_t Read(size_t n, T* p, std::unique_lock<std::mutex>& lock) {  // NOLINT
    size_t finished = 0;
    CHECK(n <= MaxCapacity() - reading_count_);
    reading_count_ += n;
    while (finished < n && WaitForRead(lock)) {
      size_t m = std::min(n - finished, data_.size());
      for (size_t i = 0; i < m; i++) {
        p[finished++] = std::move(data_.front());
        data_.pop_front();
      }
      reading_count_ -= m;
    }
    reading_count_ -= n - finished;
    return finished;
  }

  size_t Write(size_t n,
               const T* p,                            // NOLINT
               std::unique_lock<std::mutex>& lock) {  // NOLINT
    size_t finished = 0;
    while (finished < n && WaitForWrite(lock)) {
      size_t m =
          std::min(n - finished, capacity_ + reading_count_ - data_.size());
      for (size_t i = 0; i < m; i++) {
        data_.push_back(p[finished++]);
      }
    }
    return finished;
  }

  size_t WriteMove(size_t n,
                   T* p,                                  // NOLINT
                   std::unique_lock<std::mutex>& lock) {  // NOLINT
    size_t finished = 0;
    while (finished < n && WaitForWrite(lock)) {
      size_t m =
          (std::min)(n - finished, capacity_ + reading_count_ - data_.size());
      for (size_t i = 0; i < m; i++) {
        data_.push_back(std::move(p[finished++]));
      }
    }
    return finished;
  }
};  // NOLINT

template <class T>
using Channel = std::shared_ptr<ChannelObject<T>>;

template <class T>
Channel<T> MakeChannel(size_t capacity = (std::numeric_limits<size_t>::max)()) {
  return std::make_shared<ChannelObject<T>>(capacity);
}

template <class T, class U>
Channel<T> MakeChannel(const Channel<U>& other) {
  CHECK(other != nullptr) << "channel can not be NULL";
  Channel<T> chan = std::make_shared<ChannelObject<T>>();
  chan->InheritFrom(other);
  return chan;
}

// NOTE: ChannelReader is a wrapper for quick read channel with a buffer. It
// will read a block data from channel, but user can get data one by one. So it
// is important to notice that user must call operator>> until false, or call
// get_buffer_remain until false to make sure the buffered data all readed.
template <class T>
class ChannelReader {
 public:
  explicit ChannelReader(ChannelObject<T>* channel = nullptr) {
    Reset(channel);
  }

  ~ChannelReader() { CHECK(cursor_ == 0) << "Forgot to read buffer data"; }

  ChannelObject<T>* channel() { return channel_; }

  void Reset(ChannelObject<T>* channel) {
    CHECK(channel != nullptr) << "Channel can not be nullptr";
    channel_ = channel;
    cursor_ = 0;
    failed_ = !channel;
  }

  // whether there were read failed
  operator bool() { return !failed_; }

  ChannelReader<T>& operator>>(T& val) {
    if (failed_) {
      return *this;
    }
    if (cursor_ >= buffer_.size()) {
      cursor_ = 0;
      if (channel_->read(buffer_) == 0) {
        failed_ = true;
        return *this;
      }
    }
    val = std::move(buffer_[cursor_++]);
    return *this;
  }

  bool GetBufferRemain(T& val) {  // NOLINT
    if (cursor_ >= buffer_.size()) {
      cursor_ = 0;
      return false;
    }
    val = std::move(buffer_[cursor_++]);
    return true;
  }

 private:
  ChannelObject<T>* channel_ = nullptr;
  std::vector<T> buffer_;
  size_t cursor_ = 0;
  bool failed_ = true;
};  // NOLINT

template <class T>
class ChannelWriter {
 public:
  explicit ChannelWriter(ChannelObject<T>* channel = nullptr) {
    Reset(channel);
  }

  ~ChannelWriter() { CHECK(buffer_.empty()) << "Forgot to flush"; }

  ChannelObject<T>* channel() { return channel_; }

  void Reset(ChannelObject<T>* channel) {
    CHECK(buffer_.empty()) << "Forgot to flush";
    //    CHECK(channel != nullptr) << "Channel can not be nullptr";
    channel_ = channel;
    buffer_.clear();
    failed_ = !channel;
  }

  // whether there were write failed
  operator bool() { return !failed_; }

  ChannelWriter<T>& operator<<(T&& val) {
    if (failed_) {
      return *this;
    }
    buffer_.push_back(std::move(val));
    if (buffer_.size() >= channel_->BlockSize()) {
      Flush();
    }
    return *this;
  }

  ChannelWriter<T>& operator<<(const T& val) {
    if (failed_) {
      return *this;
    }
    buffer_.push_back(val);
    if (buffer_.size() >= channel_->BlockSize()) {
      Flush();
    }
    return *this;
  }

  void Flush() {
    if (failed_ || buffer_.empty()) {
      buffer_.clear();
      return;
    }
    failed_ |=
        channel_->WriteMove(buffer_.size(), &buffer_[0]) != buffer_.size();
    buffer_.clear();
  }

 private:
  ChannelObject<T>* channel_ = nullptr;
  std::vector<T> buffer_;
  bool failed_ = true;
};  // NOLINT

// only used for range-for loop
// for (auto& x : chan) {...}
template <class T>
struct ChannelIterator {
  std::shared_ptr<ChannelReader<T>> reader_;
  T data_;

  void operator++() {
    CHECK(reader_ != nullptr) << "reader can not be NULL";
    if (!(*reader_ >> data_)) {
      reader_ = nullptr;
    }
  }

  T& operator*() { return data_; }

  friend bool operator==(const ChannelIterator<T>& a,
                         const ChannelIterator<T>& b) {
    return a.reader_ == b.reader_;
  }

  friend bool operator!=(const ChannelIterator<T>& a,
                         const ChannelIterator<T>& b) {
    return a.reader_ != b.reader_;
  }
};  // NOLINT

template <class T>
ChannelIterator<T> begin(ChannelObject<T>* chan) {
  ChannelIterator<T> it{std::make_shared<ChannelReader<T>>(chan), T()};
  ++it;
  return it;
}

template <class T>
ChannelIterator<T> end(ChannelObject<T>* chan) {
  return {nullptr, T()};
}

}  // namespace framework
}  // namespace paddle
