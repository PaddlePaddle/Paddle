// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cinn {
namespace utils {

/*
TODO(Aurelius84): For now, we don't implement EventLevel to
control or shield the event greater than specified level.
So EventType is not strictly a single-layer structure.
*/
enum class EventType {
  // kOrdinary is default type
  kOrdinary,
  // kGraph is frontend Graph process
  kGraph,
  // kProgram is fronted Program process
  kProgram,
  // kFusePass is Graph and Program pass process
  kFusePass,
  // kCompute is NetBuilder OpLower process in OpLowering
  kCompute,
  // kSchedule is applying Schedule process in OpLowering
  kSchedule,
  // kOptimize is applying Optimize process in OpLowering
  kOptimize,
  // kCodeGen is AstCodegen process
  kCodeGen,
  // kCompile is LLVM or CUDA NVTX compile process
  kCompile,
  // kInstruction is running instruction process
  kInstruction
};

inline std::string EventTypeToString(const EventType& type);

std::ostream& operator<<(std::ostream& os, const EventType& type);

struct HostEvent {
  std::string annotation_;
  double duration_;  // ms
  EventType type_;

  HostEvent(const std::string& annotation, double duration, EventType type)
      : annotation_(annotation), duration_(duration), type_(type) {}
};

class Summary {
 public:
  struct Ratio {
    double value;
    Ratio(double val) : value(val) {}  // NOLINT
    std::string ToStr() const { return std::to_string(value); }
  };

  struct Item {
    HostEvent info;
    Ratio sub_ratio{0.0};    // percentage of EventType
    Ratio total_ratio{0.0};  // percentage of total process

    explicit Item(const HostEvent& e) : info(e) {}
    bool operator<(const Item& other) const {
      return total_ratio.value > other.total_ratio.value;
    }
  };

  static std::string Format(const std::vector<HostEvent>& events);

  static std::string AsStr(const std::vector<Item>& itemsm, int data_width);
};

class HostEventRecorder {
 public:
  // singleton
  static HostEventRecorder& GetInstance() {
    static HostEventRecorder instance;
    return instance;
  }

  static std::string Table() { return Summary::Format(GetInstance().Events()); }

  void Clear() { events_.clear(); }

  std::vector<HostEvent>& Events() { return events_; }

  void RecordEvent(const std::string& annotation,
                   double duration,
                   EventType type) {
    GetInstance().Events().emplace_back(annotation, duration, type);
  }

 private:
  std::vector<HostEvent> events_;
};

}  // namespace utils
}  // namespace cinn
