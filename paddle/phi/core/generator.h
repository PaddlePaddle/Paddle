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

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <typeinfo>
#include <utility>
#include <vector>

#include "paddle/phi/common/place.h"

namespace phi {

#define MAGIC_RANDOM_SEED 34342423252
class Generator {
 public:
  struct GeneratorState {
    int64_t device;
    uint64_t seed;
    uint64_t offset;
    std::shared_ptr<std::mt19937_64> cpu_engine;

    GeneratorState(int64_t device_ = -1,
                   uint64_t seed_ = MAGIC_RANDOM_SEED,
                   uint64_t offset_ = 0)
        : device(device_), seed(seed_), offset(offset_) {
      std::seed_seq seq({seed});
      cpu_engine = std::make_shared<std::mt19937_64>(seq);
    }

    GeneratorState(const GeneratorState& state)
        : device(state.device), seed(state.seed), offset(state.offset) {
      if (state.cpu_engine) {
        std::seed_seq seq({state.seed});
        cpu_engine = std::make_shared<std::mt19937_64>(seq);
        // Clone the engine state
        *(cpu_engine) = *(state.cpu_engine);
      }
    }

    GeneratorState& operator=(const GeneratorState& state) {
      if (this != &state) {
        device = state.device;
        seed = state.seed;
        offset = state.offset;

        if (state.cpu_engine) {
          std::seed_seq seq({state.seed});
          cpu_engine = std::make_shared<std::mt19937_64>(seq);
          *cpu_engine = *(state.cpu_engine);
        } else {
          cpu_engine = nullptr;
        }
      }
      return *this;
    }

    void reset(uint64_t new_seed = MAGIC_RANDOM_SEED) {
      std::seed_seq seq({new_seed});
      cpu_engine->seed(seq);
      offset = 0;
      seed = new_seed;
    }
  };

  Generator();

  explicit Generator(uint64_t seed);

  Generator(uint64_t seed, int64_t device_id);

  Generator(const Generator& other) = delete;

  ~Generator() = default;

  // Retrieves the cloned current state of the generator.
  GeneratorState GetState();
  // Directly sets the generator's current state to a specified state.
  void SetState(const GeneratorState&);

  // Retrieves the seed of the current generator state.
  uint64_t GetCurrentSeed();
  // Retrieves the offset of the current generator state.
  uint64_t GetCurrentOffset();

  // Retrieves the index of the current generator state.
  uint64_t GetStateIndex();
  // Sets the index for the current generator state, switching the active state.
  void SetStateIndex(uint64_t StateIndex);

  // Registers a new state with the generator and switch to new state.
  // Returns the index of this new state.
  uint64_t RegisterStateIndex(const GeneratorState&);

  // Generates and sets a new random seed.
  uint64_t Seed();
  // Sets the seed of the current generator state.
  void SetCurrentSeed(uint64_t seed);

  // Retrieves cpu cpu_engine in current state.
  std::shared_ptr<std::mt19937_64> GetCPUEngine();
  // Set CPU random number generation cpu_engine to current state
  void SetCPUEngine(std::shared_ptr<std::mt19937_64> cpu_engine);

  uint64_t Random64();

  // Increments the offset of the current generator state by a specified amount
  // and returns the new seed and offset.
  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t increment_offset);

 private:
  // Accesses the current generator state by index.
  inline GeneratorState& state();
  // Accesses the current cpu cpu_engine by index.
  inline std::shared_ptr<std::mt19937_64> cpu_engine();
  // Outputs detailed information about the current generator state to the log.
  inline void print_state_info();

  size_t current_index = 0;
  std::vector<GeneratorState> states_;
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
