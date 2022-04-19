// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <ostream>
#include <vector>

namespace infrt {
namespace common {

struct Target {
  /**
   * The operating system used by the target. Determines which system calls to
   * generate.
   */
  enum class OS : int {
    Unk = -1,
    Linux,
    Windows,
  };

  /**
   * The architecture used by the target. Determines the instruction set to use.
   */
  enum class Arch : int {
    Unk = -1,
    X86,
    ARM,
    NVGPU,
  };

  enum class Bit : int {
    Unk = -1,
    k32,
    k64,
  };

  OS os{OS::Unk};
  Arch arch{Arch::Unk};
  Bit bits{Bit::Unk};

  enum class Feature : int {
    JIT = 0,
    Debug,
  };

  /**
   * The library used by the target.
   */
  enum class Lib : int {
    Unk = -1,
    MKL,
  };
  std::vector<Feature> features;
  std::vector<Lib> libs;

  explicit Target(OS o = OS::Linux,
                  Arch a = Arch::Unk,
                  Bit b = Bit::Unk,
                  const std::vector<Feature>& features = {},
                  const std::vector<Lib>& libs = {})
      : os(o), arch(a), bits(b), features(features), libs(libs) {}

  bool defined() const {
    return os != OS::Unk && arch != Arch::Unk && bits != Bit::Unk;
  }

  int max_num_threads() const;

  int get_target_bits() const;

  std::vector<Lib> get_target_libs() const;

  bool operator==(const Target& other) const;
  bool operator!=(const Target& other) const { return !(*this == other); }
  friend std::ostream& operator<<(std::ostream& os, const Target& target);
};

static const Target& UnkTarget() {
  static Target target(
      Target::OS::Unk, Target::Arch::Unk, Target::Bit::Unk, {}, {});
  return target;
}

static const Target& DefaultHostTarget() {
  static Target target(
      Target::OS::Linux, Target::Arch::X86, Target::Bit::k64, {}, {});
  return target;
}

static const Target& DefaultNVGPUTarget() {
  static Target target(
      Target::OS::Linux, Target::Arch::NVGPU, Target::Bit::k64, {}, {});
  return target;
}

std::ostream& operator<<(std::ostream& os, Target::Arch arch);

}  // namespace common
}  // namespace infrt
