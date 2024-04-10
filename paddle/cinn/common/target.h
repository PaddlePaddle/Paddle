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

#include <array>
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
    AMDGPU,
    IntelGPU,
    CambrianMLU,
  };

  enum class Language : int {
    Unk = -1,
    llvm,
    cuda,
    hip,
    sycl,
    bangc,
  };

  enum class Bit : int {
    Unk = -1,
    k32,
    k64,
  };

  OS os{OS::Unk};
  Arch arch{Arch::Unk};
  Language language{Language::Unk};
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
                  Language l = Language::Unk,
                  Bit b = Bit::Unk,
                  const std::vector<Feature>& features = {},
                  const std::vector<Lib>& libs = {});

  bool defined() const {
    return os != OS::Unk && arch != Arch::Unk && bits != Bit::Unk;
  }
  // gpu use SIMT
  bool arch_is_gpu() const;
  // xpu use vector/metric intrinsics
  bool arch_is_xpu() const;
  //! Get the Runtime architecture, it is casted to integer to avoid header file
  //! depending.
  int runtime_arch() const;

  int max_num_threads() const;

  int get_warp_size() const;

  int get_multi_processor_count() const;

  int get_max_threads_per_sm() const;

  int get_max_blocks_per_sm() const;

  int get_target_bits() const;

  std::array<int, 3> get_max_grid_dims() const;
  std::array<int, 3> get_max_block_dims() const;

  std::vector<Lib> get_target_libs() const;

  std::string arch_str() const;
  // only support for sycl backend
  void SetActiveDevices(std::vector<int> deviceIds);
  bool operator==(const Target& other) const;
  bool operator!=(const Target& other) const { return !(*this == other); }
  friend std::ostream& operator<<(std::ostream& os, const Target& target);
};

const Target& UnkTarget();

const Target& DefaultHostTarget();

const Target& DefaultNVGPUTarget();

const Target& SYCLTarget(Target::Arch arch = Target::Arch::Unk);

const Target& DefaultROCMTarget();

const Target& DefaultTarget();

int GetMaxThreads();

int GetMaxBlocks();

std::ostream& operator<<(std::ostream& os, Target::Arch arch);
std::ostream& operator<<(std::ostream& os, Target::Language language);

}  // namespace common
}  // namespace cinn
