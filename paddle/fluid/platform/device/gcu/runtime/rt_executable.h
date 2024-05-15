/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"

namespace hlir {
class HlirDispatch;
}  // namespace hlir

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

struct Executable {
  topsExecutable_t tops_exec = nullptr;
  hlir::HlirDispatch* dispatch = nullptr;
  const std::vector<uint64_t> input_sizes;
  const std::vector<uint64_t> output_sizes;

  static std::shared_ptr<Executable> Create(
      const std::tuple<std::unique_ptr<char[]>, size_t>& exec);
  static std::shared_ptr<Executable> Create(
      hlir::HlirDispatch* dispatch,
      const std::vector<uint64_t> input_sizes,
      const std::vector<uint64_t> output_sizes);

  topsExecutable_t GetTopsExecutable() const { return tops_exec; }
  hlir::HlirDispatch* GetHlirDispatch() const { return dispatch; }

  uint64_t NumOfInputs() const;

  uint64_t NumOfOutputs() const;

  void OutputSizeList(uint64_t* output_size_list) const;
  void InputSizeList(uint64_t* output_size_list) const;

  explicit Executable(topsExecutable_t executable);
  explicit Executable(hlir::HlirDispatch* dispatch,
                      const std::vector<uint64_t> input_sizes,
                      const std::vector<uint64_t> output_sizes);

  ~Executable();

  Executable() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(Executable);
};
using ExecutablePtr = std::shared_ptr<Executable>;
}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
