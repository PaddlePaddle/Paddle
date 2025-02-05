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
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/phi/backends/dynload/cupti.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/os_info.h"

namespace paddle {
namespace platform {

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  PADDLE_ENFORCE_GE(
      size_s,
      0,
      common::errors::Fatal("Error during profiler data formatting."));
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), size - 1);  // exclude the '\0'
}

template <typename basic_type>
std::string json_vector(const std::vector<basic_type> type_vector) {
  std::ostringstream res_stream;
  auto count = type_vector.size();
  res_stream << "[";
  for (auto it = type_vector.begin(); it != type_vector.end(); it++) {
    if (count > 1) {
      res_stream << (*it) << ",";
    } else {
      res_stream << (*it);
    }
    count--;
  }
  res_stream << "]";
  return res_stream.str();
}

template <typename basic_type>
std::string json_vector(
    const std::vector<std::vector<basic_type>> shape_vector) {
  std::ostringstream res_stream;
  auto count = shape_vector.size();
  res_stream << "[";
  for (auto it = shape_vector.begin(); it != shape_vector.end(); it++) {
    if (count > 1) {
      res_stream << json_vector(*it) << ",";
    } else {
      res_stream << json_vector(*it);
    }
    count--;
  }
  res_stream << "]";
  return res_stream.str();
}

template <>
std::string json_vector<std::string>(
    const std::vector<std::string> type_vector);

template <typename type>
std::string json_dict(const std::map<std::string, std::vector<type>> data_map) {
  std::ostringstream res_stream;
  auto count = data_map.size();
  res_stream << "{";
  for (auto it = data_map.begin(); it != data_map.end(); it++) {
    if (count > 1) {
      res_stream << "\"" << it->first << "\""
                 << ":" << json_vector(it->second) << ",";
    } else {
      res_stream << "\"" << it->first << "\""
                 << ":" << json_vector(it->second);
    }
    count--;
  }
  res_stream << "}";
  return res_stream.str();
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

static int64_t nsToUs(uint64_t end_ns, uint64_t start_ns = 0) {
  return (end_ns - start_ns) / 1000;
}

const char* StringTracerMemEventType(phi::TracerMemEventType type);

const char* StringTracerEventType(phi::TracerEventType type);

static float nsToUsFloat(uint64_t end_ns, uint64_t start_ns = 0) {
  return static_cast<float>(end_ns - start_ns) / 1000;
}
static float nsToMsFloat(uint64_t end_ns, uint64_t start_ns = 0) {
  return static_cast<float>(end_ns - start_ns) / 1000 / 1000;
}

#ifdef PADDLE_WITH_CUPTI
#ifdef PADDLE_WITH_HIP
float CalculateEstOccupancy(uint32_t DeviceId,
                            int32_t DynamicSharedMemory,
                            int32_t BlockX,
                            int32_t BlockY,
                            int32_t BlockZ,
                            void* kernelFunc,
                            uint8_t launchType);
#else
float CalculateEstOccupancy(uint32_t deviceId,
                            uint16_t registersPerThread,
                            int32_t staticSharedMemory,
                            int32_t dynamicSharedMemory,
                            int32_t blockX,
                            int32_t blockY,
                            int32_t blockZ,
                            float blocksPerSm);
#endif  // PADDLE_WITH_HIP
#endif  // PADDLE_WITH_CUPTI

}  // namespace platform
}  // namespace paddle
