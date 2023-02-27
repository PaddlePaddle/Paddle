/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/generator.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

const std::shared_ptr<Generator>& DefaultXPUGenerator(int64_t device_id) {
#if defined(PADDLE_WITH_XPU)

  static int64_t num_xpu_devices = -1;
  static std::once_flag num_devices_init_flag;
  static std::deque<std::once_flag> xpu_device_flags;
  static std::vector<std::shared_ptr<Generator>> default_xpu_generators;

  std::call_once(num_devices_init_flag, []() {
    num_xpu_devices = phi::backends::xpu::GetXPUDeviceCount();
    xpu_device_flags.resize(num_xpu_devices);
    default_xpu_generators.resize(num_xpu_devices);
  });
  if (device_id < 0) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("xpu device id shoule be greater than 0"));
  }

  std::call_once(xpu_device_flags[device_id], [device_id]() {
    default_xpu_generators[device_id] =
        std::make_shared<Generator>(GetRandomSeed(), device_id);
    VLOG(4) << "initial seed: "
            << default_xpu_generators[device_id]->GetCurrentSeed();
  });
  return default_xpu_generators[device_id];
#else
  PADDLE_THROW(phi::errors::PermissionDenied(
      "getDefaultXPUGenerator only support in XPU place"));
#endif
}

const std::shared_ptr<Generator>& DefaultCUDAGenerator(int64_t device_id) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  static int64_t num_cuda_devices = -1;
  static std::once_flag num_devices_init_flag;
  static std::deque<std::once_flag> cuda_device_flags;
  static std::vector<std::shared_ptr<Generator>> default_cuda_generators;

  std::call_once(num_devices_init_flag, []() {
    num_cuda_devices = phi::backends::gpu::GetGPUDeviceCount();
    cuda_device_flags.resize(num_cuda_devices);
    default_cuda_generators.resize(num_cuda_devices);
  });
  if (device_id < 0) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "cuda device id shoule be greater than 0"));
  }

  std::call_once(cuda_device_flags[device_id], [device_id]() {
    default_cuda_generators[device_id] =
        std::make_shared<Generator>(GetRandomSeed(), device_id);
    VLOG(4) << "initial seed: "
            << default_cuda_generators[device_id]->GetCurrentSeed();
  });
  return default_cuda_generators[device_id];
#else
  PADDLE_THROW(phi::errors::PermissionDenied(
      "getDefaultCUDAGenerator only support in CUDA place"));
#endif
}

const std::shared_ptr<Generator>& DefaultCPUGenerator() {
  static auto default_cpu_generator =
      std::make_shared<Generator>(GetRandomSeed());
  return default_cpu_generator;
}

using RNGMap = std::unordered_map<std::string, std::shared_ptr<Generator>>;

static RNGMap& GetRandomSeedGeneratorMap() {
  static auto random_seed_generator_map = RNGMap();
  return random_seed_generator_map;
}

const std::shared_ptr<Generator>& SetRandomSeedGenerator(
    const std::string& name, uint64_t seed) {
  auto& rng_map = GetRandomSeedGeneratorMap();
  auto iter = rng_map.find(name);
  PADDLE_ENFORCE_EQ(iter == rng_map.end(),
                    true,
                    phi::errors::AlreadyExists(
                        "%s RandomSeedGenerator is already exist", name));

  auto generator = std::make_shared<Generator>(seed);
  bool emplace_success = rng_map.emplace(name, generator).second;
  PADDLE_ENFORCE_EQ(
      emplace_success,
      true,
      phi::errors::PermissionDenied(
          "SetRandomSeedGenerator cannot emplace %s RandomSeedGenerator",
          name));
  return rng_map[name];
}

const std::shared_ptr<Generator>& GetRandomSeedGenerator(
    const std::string& name) {
  auto& rng_map = GetRandomSeedGeneratorMap();
  auto iter = rng_map.find(name);
  PADDLE_ENFORCE_EQ(
      iter != rng_map.end(),
      true,
      phi::errors::NotFound("%s RandomSeedGenerator is not found, please "
                            "use `set_random_seed_generator` to set rng first",
                            name));
  return iter->second;
}

// There are 3 conditions:
// (1) op seed is set, use op seed.
// (2) op seed is not set, global seed is set, use global seed.
// (3) op seed is not set, global seed is not set too, use random seed from
// RandomGenerator.
std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t seed) {
  if (seed == 0) {
    VLOG(4) << "Use random engine from generator";
    return DefaultCPUGenerator()->GetCPUEngine();
  } else {
    // NOTE(zhiqiu): creating an engine instance everytime instead of using
    // OpDefaultCPUEngine(), this is the legacy behavior of random operators.
    // The benefit is that when runing PE with fixed-seed in multiple thrads,
    // each thread has their own engine, and doesn't affect each other.
    //
    // And we need to measure the determinacy of Generator in PE.
    auto engine = std::make_shared<std::mt19937_64>();
    static std::mutex mu_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      engine->seed(seed);
    }
    return engine;
  }
}

Generator::Generator() {
  auto seed = GetRandomSeed();
  std::seed_seq seq({seed});
  auto engine = std::make_shared<std::mt19937_64>(seq);
  this->state_.cpu_engine = *engine;
  this->state_.device = -1;
  this->state_.current_seed = seed;
  this->state_.thread_offset = 0;
  this->engine_ = engine;
  VLOG(4) << "initial seed: " << this->state_.current_seed
          << ", cpu engine: " << &this->state_.cpu_engine;
}

Generator::Generator(uint64_t seed) {
  std::seed_seq seq({seed});
  auto engine = std::make_shared<std::mt19937_64>(seq);
  this->state_.cpu_engine = *engine;
  this->state_.device = -1;
  this->state_.current_seed = seed;
  this->state_.thread_offset = 0;
  this->engine_ = engine;
  VLOG(4) << "initial seed: " << this->state_.current_seed
          << ", cpu engine: " << &this->state_.cpu_engine;
}

Generator::Generator(uint64_t seed, uint64_t device_id) {
  std::seed_seq seq({seed});
  auto engine = std::make_shared<std::mt19937_64>(seq);
  this->state_.cpu_engine = *engine;
  this->state_.device = device_id;
  this->state_.current_seed = seed;
  this->state_.thread_offset = 0;
  this->engine_ = engine;
  VLOG(4) << "initial seed: " << this->state_.current_seed
          << ", cpu engine: " << &this->state_.cpu_engine;
}

phi::Generator::GeneratorState Generator::GetState() {
  std::lock_guard<std::mutex> lock(this->mu_);
  state_.cpu_engine = *engine_;
  VLOG(4) << "Get Random state: "
          << "device id: " << (uint64_t)(this->state_.device)
          << ", current_seed: " << this->state_.current_seed
          << ", thread_offset: " << this->state_.thread_offset
          << ", cpu engine: " << *(this->engine_);
  return this->state_;
}

void Generator::SetState(const phi::Generator::GeneratorState& state) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->state_ = state;
  this->engine_ = std::make_shared<std::mt19937_64>(state.cpu_engine);
  VLOG(4) << "Set Random state: "
          << "device id: " << (uint64_t)(this->state_.device)
          << ", current_seed: " << this->state_.current_seed
          << ", thread_offset: " << this->state_.thread_offset
          << ", cpu engine: " << *(this->engine_);
}

uint64_t Generator::GetCurrentSeed() {
  std::lock_guard<std::mutex> lock(this->mu_);
  return this->state_.current_seed;
}

uint64_t Generator::Seed() {
  std::lock_guard<std::mutex> lock(this->mu_);
  uint64_t seed;
  std::random_device de;
  seed = ((((uint64_t)de()) << 32) + de()) & 0x1FFFFFFFFFFFFF;
  this->state_.current_seed = seed;
  std::seed_seq seq({seed});
  this->engine_->seed(seq);

  return this->state_.current_seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->state_.current_seed = seed;
  this->state_.thread_offset = 0;
  std::seed_seq seq({seed});
  this->engine_->seed(seq);
}

std::shared_ptr<std::mt19937_64> Generator::GetCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mu_);
  return this->engine_;
}

void Generator::SetCPUEngine(std::shared_ptr<std::mt19937_64> engine) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->engine_ = engine;
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(this->mu_);
  auto engine = this->engine_;
  return (*engine)();
}

std::pair<uint64_t, uint64_t> Generator::IncrementOffset(
    uint64_t increament_offset) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::lock_guard<std::mutex> lock(this->mu_);
  uint64_t cur_offset = this->state_.thread_offset;
  this->state_.thread_offset += increament_offset;
  return std::make_pair(this->state_.current_seed, cur_offset);
#else
  PADDLE_THROW(phi::errors::PermissionDenied(
      "Increment Offset only support in CUDA place"));
#endif
}

}  // namespace phi
