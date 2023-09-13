// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/adt.h"

namespace cinn::adt {

class AutoSize final {};

// ScheduleSize = Int64 | AutoSize
DEFINE_ADT_UNION(ScheduleSize, std::int64_t, AutoSize);

// S(Spatial): S0 = BlockIdx; S1 = ThreadIdx
// ScheduleType = S0x | S0y | S0z | S1x | S1y | S1z | Temporal | Vectorize |
// Unroll
class S0x final {
 public:
  bool IsSpatial() const { return true; }
};

class S0y final {
 public:
  bool IsSpatial() const { return true; }
};

class S0z final {
 public:
  bool IsSpatial() const { return true; }
};

class S1x final {
 public:
  bool IsSpatial() const { return true; }
};

class S1y final {
 public:
  bool IsSpatial() const { return true; }
};

class S1z final {
 public:
  bool IsSpatial() const { return true; }
};

class Temporal final {
 public:
  bool IsSpatial() const { return false; }

 private:
  std::string iter_var_name_;
};

class Vectorize final {
 public:
  bool IsSpatial() const { return false; }

 private:
  std::string iter_var_name_;
};

class Unroll final {
 public:
  bool IsSpatial() const { return false; }

 private:
  std::string iter_var_name_;
};

DEFINE_ADT_UNION(
    ScheduleType, S0x, S0y, S0z, S1x, S1y, S1z, Temporal, Vectorize, Unroll);

// LoopDescriptor = (ScheduleType, ScheduleSize)
class LoopDescriptor final : public Tuple<ScheduleType, ScheduleSize> {
 public:
  using Tuple<ScheduleType, ScheduleSize>::Tuple;

  const ScheduleType& GetScheduleType() const {
    return std::get<0>(this->tuple());
  }

  const ScheduleSize& GetScheduleSize() const {
    return std::get<1>(this->tuple());
  }

  bool operator==(const LoopDescriptor& other) const {
    return &this->tuple() == &other.tuple();
  }
};

// ScheduleDescriptor = [LoopDescriptor]
using ScheduleDescriptor = List<LoopDescriptor>;

inline bool IsSpatial(const ScheduleType& schedule_type) {
  return std::visit([](const auto& impl) { return impl.IsSpatial(); },
                    schedule_type.variant());
}

}  // namespace cinn::adt
