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

#include <string>

#include "paddle/cinn/common/target.h"

namespace cinn {
namespace runtime {

bool CheckStringFlagTrue(const std::string &flag);
bool CheckStringFlagFalse(const std::string &flag);

void SetCinnCudnnDeterministic(bool state);
bool GetCinnCudnnDeterministic();

bool CanUseNvccCompiler();
bool UseHipccCompiler();

class RandomSeed {
 public:
  static uint64_t GetOrSet(uint64_t seed = 0);
  static uint64_t Clear();

 private:
  RandomSeed() = default;
  RandomSeed(const RandomSeed &) = delete;
  RandomSeed &operator=(const RandomSeed &) = delete;

  static uint64_t seed_;
};

bool IsCompiledWithCUDA();
bool IsCompiledWithCUDNN();

class CurrentTarget {
 public:
  static cinn::common::Target &GetCurrentTarget();
  static void SetCurrentTarget(const cinn::common::Target &target);

 private:
  CurrentTarget() = default;
  CurrentTarget(const CurrentTarget &) = delete;
  CurrentTarget &operator=(const CurrentTarget &) = delete;

  static cinn::common::Target target_;
};

}  // namespace runtime
}  // namespace cinn
