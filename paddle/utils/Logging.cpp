/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
 * Basically from tensorflow/core/platform/default/logging.cc
 * Used in embedded system where there is no glogs.
 */

#include "Logging.h"
#ifndef PADDLE_USE_GLOG
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace paddle {

namespace internal {

std::string join(const std::string& part1, const std::string& part2) {
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline bool env2bool(const char* envName, bool defaultValue = false) {
  char* envValue = getenv(envName);
  if (envValue == nullptr) {
    return defaultValue;
  } else {
    return memchr("tTyY1\0", envValue[0], 6) != nullptr;
  }
}

static inline int env2int(const char* envName, int defaultValue = 0) {
  char* envValue = getenv(envName);
  if (envValue == nullptr) {
    return defaultValue;
  } else {
    int retValue = defaultValue;
    try {
      retValue = std::stoi(envValue);
    } catch (...) {
      // pass
    }
    return retValue;
  }
}

static inline int env2index(const char* envName,
                            const std::vector<std::string>& options,
                            int defaultValue) {
  char* envValue = getenv(envName);
  if (envValue == nullptr) {
    return defaultValue;
  } else {
    for (size_t i = 0; i < options.size(); ++i) {
      if (options[i] == envValue) {
        return static_cast<int>(i);
      }
    }
    return defaultValue;
  }
}

static bool gLogToStderr = env2bool("PLOG_LOGTOSTDERR", true);
static const std::vector<std::string> gLevelName = {
    "INFO", "WARNING", "ERROR", "FATAL"};
static int gMinLogLevel =
    env2int("PLOG_MINLOGLEVEL", env2index("PLOG_MINLOGLEVEL", gLevelName, 0));

static std::vector<std::vector<int>> gLogFds;
static std::vector<int> gLogFileFds;
static bool gLogInited = false;
static void freeLogFileFds() {
  for (auto fd : gLogFileFds) {
    close(fd);
  }
}

static void initializeLogFds(char* argv0) {
  gLogFds.resize(NUM_SEVERITIES);

  for (int i = gMinLogLevel; i < NUM_SEVERITIES && gLogToStderr;
       ++i) {  // Add stderr
    std::vector<int>& fds = gLogFds[i];
    fds.push_back(STDERR_FILENO);
  }

  char* logDir = getenv("PLOG_LOGDIR");

  for (int i = gMinLogLevel; i < NUM_SEVERITIES && logDir != nullptr; ++i) {
    std::string filename =
        join(logDir, std::string(argv0) + "." + gLevelName[i]);
    int fd = open(filename.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd == -1) {
      fprintf(stderr, "Open log file error!");
      exit(1);
    }
    gLogFileFds.push_back(fd);

    std::vector<int>& curFds = gLogFds[i];
    curFds.insert(curFds.end(), gLogFileFds.begin(), gLogFileFds.end());
  }

  atexit(freeLogFileFds);
  gLogInited = true;
}

static void (*gFailureFunctionPtr)() ATTR_NORETURN = abort;

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage::~LogMessage() { this->generateLogMessage(); }

void LogMessage::generateLogMessage() {
  if (!gLogInited) {
    fprintf(stderr,
            "%c %s:%d] %s\n",
            "IWEF"[severity_],
            fname_,
            line_,
            str().c_str());
  } else {
    for (auto& fd : gLogFds[this->severity_]) {
      dprintf(fd,
              "%c %s:%d] %s\n",
              "IWEF"[severity_],
              fname_,
              line_,
              str().c_str());
    }
  }
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  generateLogMessage();
  gFailureFunctionPtr();
}
}  // namespace internal

void initializeLogging(int argc, char** argv) {
  internal::initializeLogFds(argv[0]);
}

namespace logging {
void setMinLogLevel(int level) { paddle::internal::gMinLogLevel = level; }

void installFailureFunction(void (*callback)() ATTR_NORETURN) {
  paddle::internal::gFailureFunctionPtr = callback;
}

}  // namespace logging

}  // namespace paddle

#else
namespace paddle {
void initializeLogging(int argc, char** argv) {
  (void)(argc);
  if (!getenv("GLOG_logtostderr")) {
    google::LogToStderr();
  }
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
}

namespace logging {
void setMinLogLevel(int level) { FLAGS_minloglevel = level; }
void installFailureFunction(void (*callback)()) {
  google::InstallFailureFunction(callback);
}
void installFailureWriter(void (*callback)(const char*, int)) {
  google::InstallFailureWriter(callback);
}
}  // namespace logging
}  // namespace paddle
#endif
