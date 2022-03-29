/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <ctime>
#include <string>
#include "paddle/fluid/platform/dynload/cupti.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/os_info.h"

namespace paddle {
namespace platform {

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  PADDLE_ENFORCE_GE(size_s, 0, platform::errors::Fatal(
                                   "Error during profiler data formatting."));
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), size - 1);  // exclude the '\0'
}

static std::string GetStringFormatLocalTime() {
  std::time_t rawtime;
  std::tm* timeinfo;
  char buf[100];
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::strftime(buf, 100, "%F-%X", timeinfo);
  return std::string(buf);
}

static int64_t nsToUs(int64_t ns) { return ns / 1000; }

#ifdef PADDLE_WITH_CUPTI
float CalculateEstOccupancy(uint32_t deviceId, uint16_t registersPerThread,
                            int32_t staticSharedMemory,
                            int32_t dynamicSharedMemory, int32_t blockX,
                            int32_t blockY, int32_t blockZ, float blocksPerSm);
#endif
}  // namespace platform
}  // namespace paddle
