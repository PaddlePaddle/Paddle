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
#include <cstdlib>

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
