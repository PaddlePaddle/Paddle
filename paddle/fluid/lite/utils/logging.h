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

/*
 * This file implements an lightweight alternative for glog, which is more
 * friendly for mobile.
 */
#pragma once
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <sstream>

// LOG()
#define LOG(status) LOG_##status.stream()
#define LOG_ERROR LOG_INFO
#define LOG_INFO paddle::lite::LogMessage(__FILE__, __LINE__)
#define LOG_WARNING paddle::lite::LogMessage(__FILE__, __LINE__)
#define LOG_FATAL paddle::lite::LogMessageFatal(__FILE__, __LINE__)
// Not supported yet.
#define VLOG(level) LOG_INFO.stream()

// CHECK()
#define CHECK(x)                                             \
  if (!(x))                                                  \
  paddle::lite::LogMessageFatal(__FILE__, __LINE__).stream() \
      << "Check failed: " #x << ": "
#define CHECK_EQ(x, y) _CHECK_BINARY(x, ==, y)
#define CHECK_LT(x, y) _CHECK_BINARY(x, <, y)
#define CHECK_LE(x, y) _CHECK_BINARY(x, <=, y)
#define CHECK_GT(x, y) _CHECK_BINARY(x, >, y)
#define CHECK_GE(x, y) _CHECK_BINARY(x, >=, y)
#define _CHECK_BINARY(x, cmp, y) CHECK(x cmp y) << x << "!" #cmp << y << " "

namespace paddle {
namespace lite {

class LogMessage {
 public:
  LogMessage(const char* file, int lineno) {
    const int kMaxLen = 20;
    const int len = strlen(file);
    if (len > kMaxLen) {
      log_stream_ << '[' << "..." << file + len - kMaxLen << ":" << lineno
                  << "] ";
    } else {
      log_stream_ << '[' << file << ":" << lineno << "] ";
    }
  }

  ~LogMessage() {
    log_stream_ << '\n';
    std::cerr << log_stream_.str();
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int lineno) : LogMessage(file, lineno) {}

  ~LogMessageFatal() {
    log_stream_ << '\n';
    std::cerr << log_stream_.str();
    abort();
  }
};

}  // namespace lite
}  // namespace paddle
