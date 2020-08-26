/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace framework {

static uint64_t GetRandomSeed() {
  std::random_device rd;
  // double has 53 bit significant, so limit uint64 to 53 bits
  return ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
}

struct GeneratorState {
  int64_t device = -1;
  uint64_t current_seed = 34342423252;
  std::shared_ptr<std::mt19937_64> cpu_engine;
};

struct Generator {
  Generator() {
    GeneratorState default_gen_state_cpu;
    default_gen_state_cpu.device = -1;
    default_gen_state_cpu.current_seed = GetRandomSeed();
    std::seed_seq seq({default_gen_state_cpu.current_seed});
    default_gen_state_cpu.cpu_engine = std::make_shared<std::mt19937_64>(seq);
    this->state_ = std::make_shared<GeneratorState>(default_gen_state_cpu);
  }
  explicit Generator(uint64_t seed) {
    GeneratorState default_gen_state_cpu;
    default_gen_state_cpu.device = -1;
    default_gen_state_cpu.current_seed = seed;
    std::seed_seq seq({seed});
    default_gen_state_cpu.cpu_engine = std::make_shared<std::mt19937_64>(seq);
    this->state_ = std::make_shared<GeneratorState>(default_gen_state_cpu);
    this->is_init_py_ = true;  // TODO(zhiqiu): remove it in future
  }
  explicit Generator(GeneratorState state_in)
      : state_{std::make_shared<GeneratorState>(state_in)} {}
  Generator(const Generator& other)
      : Generator(other, std::lock_guard<std::mutex>(other.mutex)) {}

  // get random state
  GeneratorState* GetState();
  // set random state
  void SetState(GeneratorState* state_in);
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

  void SetIsInitPy(bool);
  bool GetIsInitPy() const;

 private:
  std::shared_ptr<GeneratorState> state_;
  mutable std::mutex mutex;

  Generator(const Generator& other, const std::lock_guard<std::mutex>&)
      : state_(std::make_shared<GeneratorState>(*(other.state_))) {}
  // NOTE(zhiqiu): is_init_py_ is used to make generator be compatible with old
  // seed, and it should be removed after all random-related operators and
  // unittests upgrades to use generator.
  bool is_init_py_ = false;
};

// The DefaultCPUGenerator is used in manual_seed()
const std::shared_ptr<Generator>& DefaultCPUGenerator();

// If op seed is set or global is not set, the OpDefaultCPUEngine is used.
std::shared_ptr<std::mt19937_64> OpDefaultCPUEngine();

std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t);

}  // namespace framework
}  // namespace paddle
