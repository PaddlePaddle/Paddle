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

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/generator.h"

namespace paddle {
namespace framework {

std::shared_ptr<Generator> Generator::gen_instance_ = NULL;

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
  this->state_->cpu_engine.seed(seq);

  return this->state_->current_seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->current_seed = uint64_t(seed);
  std::seed_seq seq({seed});
  this->state_->cpu_engine.seed(seq);
}

std::mt19937_64& Generator::GetCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->cpu_engine;
}

void Generator::SetCPUEngine(std::mt19937_64 engine) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->cpu_engine = std::mt19937_64(engine);
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->cpu_engine();
}

}  // namespace framework
}  // namespace paddle
