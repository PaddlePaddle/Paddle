// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <string>
#include <vector>

namespace cinn {
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

  //! Get the Runtime architecture, it is casted to integer to avoid header file
  //! depending.
  int runtime_arch() const;

  int max_num_threads() const;

  int get_multi_processor_count() const;

  int get_max_threads_per_sm() const;

  int get_max_blocks_per_sm() const;

  int get_target_bits() const;

  std::vector<Lib> get_target_libs() const;

  std::string arch_str() const;

  bool operator==(const Target& other) const;
  bool operator!=(const Target& other) const { return !(*this == other); }
  friend std::ostream& operator<<(std::ostream& os, const Target& target);
};

const Target& UnkTarget();

const Target& DefaultHostTarget();

const Target& DefaultNVGPUTarget();

const Target& DefaultTarget();

int GetMaxThreads();

int GetMaxBlocks();

std::ostream& operator<<(std::ostream& os, Target::Arch arch);

}  // namespace common
}  // namespace cinn
