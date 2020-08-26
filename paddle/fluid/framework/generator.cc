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

#include "paddle/fluid/framework/generator.h"

#include <glog/logging.h>

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace paddle {
namespace framework {

const std::shared_ptr<Generator>& DefaultCPUGenerator() {
  static auto default_cpu_generator =
      std::make_shared<Generator>(GetRandomSeed());
  return default_cpu_generator;
}

std::shared_ptr<std::mt19937_64> OpDefaultCPUEngine() {
  static auto op_default_cpu_engine = std::make_shared<std::mt19937_64>();
  return op_default_cpu_engine;
}

// NOTE(zhiqiu): there are 3 conditions:
// (1) op seed is not set and DefaultCPUGenerator is inited, use
// DefaultCPUGenerator
// (2) op seed is not set and DefaultCPUGenerator is not inited, use se
// OpDefaultCPUEngine() and set a radnom seed
// (3) op seed is set, use OpDefaultCPUEngine() and set the seed
std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t seed) {
  if (DefaultCPUGenerator()->GetIsInitPy() && seed == 0) {
    return DefaultCPUGenerator()->GetCPUEngine();
  } else {
    if (seed == 0) {
      seed = GetRandomSeed();
    }
    static std::mutex mu_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      OpDefaultCPUEngine()->seed(seed);
    }
    return OpDefaultCPUEngine();
  }
}

GeneratorState* Generator::GetState() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_.get();
}

void Generator::SetState(GeneratorState* state_in) {
  std::lock_guard<std::mutex> lock(this->mutex);
  *this->state_ = *state_in;
}

uint64_t Generator::GetCurrentSeed() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->current_seed;
}

uint64_t Generator::Seed() {
  std::lock_guard<std::mutex> lock(this->mutex);
  uint64_t seed;
  std::random_device de;
  seed = ((((uint64_t)de()) << 32) + de()) & 0x1FFFFFFFFFFFFF;
  this->state_->current_seed = seed;
  std::seed_seq seq({seed});
  this->state_->cpu_engine->seed(seq);

  return this->state_->current_seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->current_seed = uint64_t(seed);
  std::seed_seq seq({seed});
  this->state_->cpu_engine->seed(seq);
}

std::shared_ptr<std::mt19937_64> Generator::GetCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->cpu_engine;
}

void Generator::SetCPUEngine(std::shared_ptr<std::mt19937_64> engine) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->cpu_engine = engine;
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(this->mutex);
  auto engine = this->state_->cpu_engine;
  return (*engine)();
}

void Generator::SetIsInitPy(bool is_init_py) {
  this->is_init_py_ = is_init_py;
  VLOG(4) << "SetIsInitPy:" << this->is_init_py_;
}
bool Generator::GetIsInitPy() const { return this->is_init_py_; }

}  // namespace framework
}  // namespace paddle
