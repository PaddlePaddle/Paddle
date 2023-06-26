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

#include <stdint.h>

#include <atomic>
#include <deque>
#include <iostream>  // temp for debug
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <typeinfo>
#include <utility>

#include "paddle/phi/common/place.h"

namespace phi {

class Generator {
 public:
  struct GeneratorState {
    int64_t device = -1;
    uint64_t current_seed = 34342423252;
    uint64_t thread_offset = 0;
    std::mt19937_64 cpu_engine;
  };

  Generator();

  explicit Generator(uint64_t seed);

  Generator(uint64_t seed, uint64_t device_id);

  Generator(const Generator& other) = delete;

  ~Generator() = default;

  // get random state
  GeneratorState GetState();
  // set random state
  void SetState(const GeneratorState&);
  // get current seed
  uint64_t GetCurrentSeed();
  // random a seed and get
  uint64_t Seed();
  // set seed
  void SetCurrentSeed(uint64_t seed);
  // get cpu engine
  std::shared_ptr<std::mt19937_64> GetCPUEngine();
  // set cpu engine
  void SetCPUEngine(std::shared_ptr<std::mt19937_64>);

  uint64_t Random64();

  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t increment_offset);

  uint64_t get_device_id() { return this->state_.device; }

 private:
  GeneratorState state_;
  std::shared_ptr<std::mt19937_64> engine_;
  mutable std::mutex mu_;
};

// The DefaultCPUGenerator is used in manual_seed()
const std::shared_ptr<Generator>& DefaultCPUGenerator();

const std::shared_ptr<Generator>& DefaultCUDAGenerator(int64_t device_id = -1);

const std::shared_ptr<Generator>& DefaultXPUGenerator(int64_t device_id = -1);

const std::shared_ptr<Generator>& DefaultCustomDeviceGenerator(
    const phi::CustomPlace& place);

std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t);

const std::shared_ptr<Generator>& SetRandomSeedGenerator(
    const std::string& name, uint64_t seed);

const std::shared_ptr<Generator>& GetRandomSeedGenerator(
    const std::string& name);

}  // namespace phi
