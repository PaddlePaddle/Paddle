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
 * Basically from tensorflow/core/platform/default/logging.h
 * Used in embedded system where there is no glogs.
 */

#pragma once
#include <memory>
#include <sstream>
#include <string>

#include <glog/logging.h>
namespace paddle {

void initializeLogging(int argc, char** argv);

namespace logging {

void setMinLogLevel(int level);

void installFailureFunction(void (*callback)());

void installFailureWriter(void (*callback)(const char*, int));

}  // namespace logging
}  // namespace paddle

#ifndef NDEBUG
#define DEBUG_LEVEL 5
#define DBG VLOG(DEBUG_LEVEL)
#else
#define DBG DLOG(INFO)
#endif
