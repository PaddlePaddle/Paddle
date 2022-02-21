/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <random>

namespace pten {

class Generator {
 public:
  struct GeneratorState {
    int64_t device = -1;
    uint64_t current_seed = 34342423252;
    uint64_t thread_offset = 0;
    std::mt19937_64 cpu_engine;
  };

  virtual ~Generator() = default;

  // get random state
  virtual GeneratorState GetState() = 0;
  // set random state
  virtual void SetState(const GeneratorState&) = 0;
  // get current seed
  virtual uint64_t GetCurrentSeed() = 0;
  // random a seed and get
  virtual uint64_t Seed() = 0;
  // set seed
  virtual void SetCurrentSeed(uint64_t seed) = 0;
  // get cpu engine
  virtual std::shared_ptr<std::mt19937_64> GetCPUEngine() = 0;
  // set cpu engine
  virtual void SetCPUEngine(std::shared_ptr<std::mt19937_64>) = 0;
  virtual uint64_t Random64() = 0;
  virtual std::pair<uint64_t, uint64_t> IncrementOffset(
      uint64_t increament_offset) = 0;

  // NOTE(zhiqiu): is_init_py_ is used to make generator be compatible with
  // old seed, and it should be removed after all random-related operators
  // and unittests upgrades to use generator.
  virtual void SetIsInitPy(bool) = 0;
  virtual bool GetIsInitPy() const = 0;

  virtual uint64_t get_device_id() = 0;
};

}  // namespace pten
