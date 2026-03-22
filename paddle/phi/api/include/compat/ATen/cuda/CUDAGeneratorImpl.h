// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/core/Generator.h>
#include <c10/cuda/PhiloxCudaState.h>

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/generator.h"

namespace at {

// Forward declaration
struct CUDAGeneratorImpl;

// ---- Helper functions ------------------------------------------------------

/// Resolve device_index: if < 0, use current GPU device.
inline DeviceIndex resolve_device_index(DeviceIndex idx) {
  if (idx < 0) {
    return static_cast<DeviceIndex>(phi::backends::gpu::GetCurrentDeviceId());
  }
  return idx;
}

/**
 * CUDAGeneratorImpl — CUDA-specific GeneratorImpl backed by Paddle.
 *
 * Wraps phi::DefaultCUDAGenerator for the given device and exposes the
 * PyTorch-compatible Philox-based random number generator interface.
 */
struct CUDAGeneratorImpl : public c10::GeneratorImpl {
  // ------- constructors -----------------------------------------------------

  /// Construct for a specific device (or current device if -1).
  /// If use_default_gen is true, uses phi::DefaultCUDAGenerator (shared state).
  /// If false, creates a new independent Generator instance.
  explicit CUDAGeneratorImpl(
      DeviceIndex device_index = -1,  // NOLINT(runtime/int)
      bool use_default_gen = true)
      : c10::GeneratorImpl(
            c10::Device(c10::kCUDA, resolve_device(device_index)),
            use_default_gen
                ? get_default_paddle_gen(resolve_device(device_index))
                : create_new_paddle_gen(resolve_device(device_index))),
        philox_offset_per_thread_(0) {}

  ~CUDAGeneratorImpl() override = default;

  // ------- seed / offset (override base) ------------------------------------

  void set_current_seed(uint64_t seed) override {
    gen_->SetCurrentSeed(seed);
    philox_offset_per_thread_ = 0;
  }

  uint64_t current_seed() const override { return gen_->GetCurrentSeed(); }

  uint64_t seed() override {
    auto s = gen_->Seed();
    philox_offset_per_thread_ = 0;
    return s;
  }

  void set_offset(uint64_t offset) override {
    philox_offset_per_thread_ = offset;
  }

  uint64_t get_offset() const override { return philox_offset_per_thread_; }

  // ------- Philox -----------------------------------------------------------

  void set_philox_offset_per_thread(uint64_t offset) {
    philox_offset_per_thread_ = offset;
  }

  uint64_t philox_offset_per_thread() const {
    return philox_offset_per_thread_;
  }

  /// Generate a PhiloxCudaState and advance the internal offset.
  PhiloxCudaState philox_cuda_state(uint64_t increment) {
    PhiloxCudaState state(gen_->GetCurrentSeed(), philox_offset_per_thread_);
    philox_offset_per_thread_ += increment;
    return state;
  }

  /// Legacy helper — returns (seed, offset) and advances offset.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) {
    uint64_t offset = philox_offset_per_thread_;
    philox_offset_per_thread_ += increment;
    return {gen_->GetCurrentSeed(), offset};
  }

  // ------- clone ------------------------------------------------------------

  c10::intrusive_ptr<c10::GeneratorImpl> clone() const override {
    auto new_gen = std::make_shared<phi::Generator>(gen_->GetCurrentSeed());
    auto state = gen_->GetState();
    new_gen->SetState(state);

    auto impl = c10::make_intrusive<CUDAGeneratorImpl>(
        static_cast<DeviceIndex>(device_.index()));
    impl->gen_ = new_gen;
    impl->philox_offset_per_thread_ = philox_offset_per_thread_;
    return impl;
  }

  // ------- device_type (static) ---------------------------------------------

  static c10::DeviceType device_type() { return c10::kCUDA; }

 private:
  uint64_t philox_offset_per_thread_;

  /// Resolve device_index: if < 0, use current GPU device.
  static DeviceIndex resolve_device(DeviceIndex idx) {
    return resolve_device_index(idx);
  }

  /// Get the default Paddle phi::Generator for the given device (shared state).
  static std::shared_ptr<phi::Generator> get_default_paddle_gen(
      DeviceIndex device_index) {
    return phi::DefaultCUDAGenerator(static_cast<int64_t>(device_index));
  }

  /// Create a new independent Paddle phi::Generator for the given device.
  static std::shared_ptr<phi::Generator> create_new_paddle_gen(
      DeviceIndex /*device_index*/) {
    // Create a new Generator with a random seed (similar to PyTorch behavior)
    // Use default constructor which generates a random seed
    return std::make_shared<phi::Generator>();
  }
};

// ---- Free functions --------------------------------------------------------

namespace cuda {
namespace detail {

/// Return a reference to the default CUDA Generator for `device_index`.
/// If device_index < 0, uses the current CUDA device.
inline const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1) {
  auto idx = resolve_device_index(device_index);
  // One Generator per device, lazily initialised.
  // We use a function-local static vector guarded by call_once.
  static std::vector<Generator> generators;
  static std::once_flag init_flag;
  static int64_t num_devices = 0;

  std::call_once(init_flag, []() {
    num_devices = phi::backends::gpu::GetGPUDeviceCount();
    generators.reserve(num_devices);
    for (int64_t i = 0; i < num_devices; ++i) {
      generators.emplace_back(c10::make_intrusive<CUDAGeneratorImpl>(
          static_cast<DeviceIndex>(i), /*use_default_gen=*/true));
    }
  });

  TORCH_CHECK(idx < static_cast<DeviceIndex>(num_devices),
              "CUDA device index out of range: ",
              idx);
  return generators[static_cast<size_t>(idx)];
}

/// Create a new (non-default) CUDA Generator for `device_index`.
/// The created generator has independent state from the default generator.
inline Generator createCUDAGenerator(DeviceIndex device_index = -1) {
  return Generator(c10::make_intrusive<CUDAGeneratorImpl>(
      device_index, /*use_default_gen=*/false));
}

}  // namespace detail
}  // namespace cuda
}  // namespace at
