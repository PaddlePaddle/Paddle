/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * Basically from tensorflow/core/platform/default/logging.h
 * Used in embedded system where there is no glogs.
 */

#pragma once
#include <sstream>
#include <memory>
#include <string>

#ifndef PADDLE_USE_GLOG

//! TODO(yuyang18): Move this utility macro into some global header.
#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a##b)
#define PP_CAT_II(p, res) res

/**
 * Generate Unique Variable Name, Usefully in macro.
 * @SEE http://stackoverflow.com/questions/1082192/how-to-generate-random-variable-names-in-c-using-macros
 */
#define UNIQUE_NAME(base) PP_CAT(base, __LINE__)


namespace paddle {

//! Log levels.
const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

protected:
  /**
   * @brief Print log message to stderr, files, etc.
   */
  void generateLogMessage();

private:
  const char* fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
public:
  LogMessageFatal(const char* file, int line) __attribute__((cold));
  ~LogMessageFatal() __attribute__((noreturn));
};

#define _P_LOG_INFO \
  ::paddle::internal::LogMessage(__FILE__, __LINE__, paddle::INFO)
#define _P_LOG_WARNING \
  ::paddle::internal::LogMessage(__FILE__, __LINE__, paddle::WARNING)
#define _P_LOG_ERROR \
  ::paddle::internal::LogMessage(__FILE__, __LINE__, paddle::ERROR)
#define _P_LOG_FATAL ::paddle::internal::LogMessageFatal(__FILE__, __LINE__)

#define P_LOG(severity) _P_LOG_##severity

#define P_LOG_FIRST_N(severity, n)                                       \
  static int UNIQUE_NAME(LOG_OCCURRENCES) = 0;                           \
  if (UNIQUE_NAME(LOG_OCCURRENCES) <= n) ++UNIQUE_NAME(LOG_OCCURRENCES); \
  if (UNIQUE_NAME(LOG_OCCURRENCES) <= n) P_LOG(severity)

#define P_LOG_IF_EVERY_N(severity, condition, n)                              \
  static int UNIQUE_NAME(LOG_OCCURRENCES) = 0;                                \
  if (condition && ((UNIQUE_NAME(LOG_OCCURRENCES) =                           \
                         (UNIQUE_NAME(LOG_OCCURRENCES) + 1) % n) == (1 % n))) \
  P_LOG(severity)

#define P_LOG_EVERY_N(severity, n) P_LOG_IF_EVERY_N(severity, true, n)

// TODO(jeff): Define a proper implementation of VLOG_IS_ON
#define P_VLOG_IS_ON(lvl) ((lvl) <= 0)

#define P_LOG_IF(severity, condition) \
  if (condition) P_LOG(severity)

#define P_VLOG(lvl) P_LOG_IF(INFO, P_VLOG_IS_ON(lvl))

#define P_VLOG_IF(lvl, cond) P_LOG_IF(INFO, P_VLOG_IS_ON(lvl) && cond)

#define P_VLOG_EVERY_N(lvl, n) P_LOG_IF_EVERY_N(INFO, P_VLOG_IS_ON(lvl), n)

#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

// CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define P_CHECK(condition)         \
  if (PREDICT_FALSE(!(condition))) \
  P_LOG(FATAL) << "Check failed: " #condition " "

#define P_CHECK_EQ(val1, val2) P_CHECK((val1) == (val2))
#define P_CHECK_NE(val1, val2) P_CHECK((val1) != (val2))
#define P_CHECK_LE(val1, val2) P_CHECK((val1) <= (val2))
#define P_CHECK_LT(val1, val2) P_CHECK((val1) < (val2))
#define P_CHECK_GE(val1, val2) P_CHECK((val1) >= (val2))
#define P_CHECK_GT(val1, val2) P_CHECK((val1) > (val2))
#define P_CHECK_NOTNULL(val) P_CHECK((val) != NULL)

//! GLOG compatible APIs
//! NOTE: only implement Paddle actually used APIs.
#define LOG(x) P_LOG(x)
#define VLOG(x) P_VLOG(x)
#define DLOG(x) P_VLOG(5)
#define CHECK(x) P_CHECK(x)
#define PCHECK(x) P_CHECK(x)
#define CHECK_EQ(val1, val2) P_CHECK((val1) == (val2))
#define CHECK_NE(val1, val2) P_CHECK((val1) != (val2))
#define CHECK_LE(val1, val2) P_CHECK((val1) <= (val2))
#define CHECK_LT(val1, val2) P_CHECK((val1) < (val2))
#define CHECK_GE(val1, val2) P_CHECK((val1) >= (val2))
#define CHECK_GT(val1, val2) P_CHECK((val1) > (val2))
#define CHECK_NOTNULL(val) P_CHECK((val) != NULL)
#define VLOG_IS_ON(x) P_VLOG_IS_ON(x)
#define LOG_FIRST_N(severity, n) P_LOG_FIRST_N(severity, n)
#define LOG_IF(severity, condition) P_LOG_IF(severity, condition)
#define VLOG_EVERY_N(lvl, n) P_VLOG_EVERY_N(lvl, n)
#define VLOG_IF(lvl, cond) P_VLOG_IF(lvl, cond)
#define LOG_EVERY_N(severity, n) P_LOG_EVERY_N(severity, n)
}  //  namespace internal

/**
 * @brief initialize logging
 * @note: Current implement of logging is lack of:
 *          PrintCallStack when fatal.
 *          VLOG_IS_ON
 *        But it is portable to multi-platform, and simple enough to modify.
 */
void initializeLogging(int argc, char** argv);
namespace logging {
/**
 * @brief Set Min Log Level. if Log.level < minLogLevel, then will not print log
 *        to stream
 * @param level. Any integer is OK, but only 0 <= x <= NUM_SEVERITIES is useful.
 */
void setMinLogLevel(int level);

/**
 * @brief Install Log(Fatal) failure function. Default is abort();
 * @param callback: The failure function.
 */
void installFailureFunction(void (*callback)());

/**
 * @brief installFailureWriter
 * @note: not implemented currently.
 */
inline void installFailureWriter(void(*callback)(const char*, int)) {
  (void)(callback);  // unused callback.
}
}  //  namespace logging
}  //  namespace paddle
#else
#include <glog/logging.h>
namespace paddle {
void initializeLogging(int argc, char** argv);
namespace logging {
void setMinLogLevel(int level);
void installFailureFunction(void (*callback)());
void installFailureWriter(void(*callback)(const char*, int));
}  //  namespace logging
}
#endif  // PADDLE_USE_GLOG

#ifndef NDEBUG
#define DEBUG_LEVEL 5
#define DBG VLOG(DEBUG_LEVEL)
#else
#define DBG DLOG(INFO)
#endif
