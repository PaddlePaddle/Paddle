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

// LoopSize = Int64 | AutoSize
DEFINE_ADT_UNION(LoopSize, std::int64_t, AutoSize);

// S(Spatial): S0 = BlockIdx; S1 = ThreadIdx
// LoopType = S0x | S0y | S0z | S1x | S1y | S1z | Temporal | Vectorize |
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

  const std::string& iter_var_name() const { return iter_var_name_; }

 private:
  std::string iter_var_name_;
};

class Vectorize final {
 public:
  bool IsSpatial() const { return false; }

  const std::string& iter_var_name() const { return iter_var_name_; }

 private:
  std::string iter_var_name_;
};

class Unroll final {
 public:
  bool IsSpatial() const { return false; }

  const std::string& iter_var_name() const { return iter_var_name_; }

 private:
  std::string iter_var_name_;
};

DEFINE_ADT_UNION(
    LoopType, S0x, S0y, S0z, S1x, S1y, S1z, Temporal, Vectorize, Unroll);

// LoopDescriptor = (LoopType, LoopSize)
class LoopDescriptor final : public Tuple<LoopType, LoopSize> {
 public:
  using Tuple<LoopType, LoopSize>::Tuple;

  const LoopType& GetLoopType() const { return std::get<0>(this->tuple()); }

  const LoopSize& GetLoopSize() const { return std::get<1>(this->tuple()); }

  bool operator==(const LoopDescriptor& other) const {
    return &this->tuple() == &other.tuple();
  }
};

// ScheduleDescriptor = [LoopDescriptor]
using ScheduleDescriptor = List<LoopDescriptor>;

inline bool IsSpatial(const LoopType& loop_type) {
  return std::visit([](const auto& impl) { return impl.IsSpatial(); },
                    loop_type.variant());
}

std::string DebugStringImpl(const LoopDescriptor& loop_descriptor);

inline std::string DebugString(const LoopDescriptor& loop_descriptor) {
  return DebugStringImpl(loop_descriptor);
}

}  // namespace cinn::adt
