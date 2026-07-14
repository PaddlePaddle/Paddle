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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <utility>

#include "paddle/phi/core/generator.h"

// Forward declare PyObject for the pyobj interface.
#ifndef PyObject
struct _object;
using PyObject = _object;
#endif

namespace c10 {

/**
 * GeneratorImpl — base implementation class for at::Generator.
 *
 * Wraps a Paddle phi::Generator and exposes the PyTorch-style API
 * (set_current_seed, seed, offset, device, key_set, etc.).
 *
 * Subclasses (e.g., CPUGeneratorImpl, CUDAGeneratorImpl) may extend
 * this to add backend-specific functionality.
 *
 * Note: Inherits from intrusive_ptr_target to support intrusive_ptr.
 */
class GeneratorImpl : public intrusive_ptr_target {
 public:
  // ------- constructors / destructor ----------------------------------------

  /// Construct from an existing Paddle generator and a device.
  explicit GeneratorImpl(Device device,
                         std::shared_ptr<phi::Generator> gen = nullptr)
      : device_(device), pyobj_(nullptr) {
    if (gen) {
      gen_ = std::move(gen);
    } else {
      gen_ = std::make_shared<phi::Generator>();
    }
  }

  // Virtual destructor required for polymorphic base class
  // Note: intrusive_ptr_target destructor is protected, but we need public
  // destructor for delete to work through base pointer
  ~GeneratorImpl() override = default;

  // Non-copyable / non-movable (mirroring PyTorch semantics).
  GeneratorImpl(const GeneratorImpl&) = delete;
  GeneratorImpl& operator=(const GeneratorImpl&) = delete;

  // ------- seed / offset API ------------------------------------------------

  virtual void set_current_seed(uint64_t seed) { gen_->SetCurrentSeed(seed); }

  virtual uint64_t current_seed() const { return gen_->GetCurrentSeed(); }

  /// Generate and set a new random seed; return it.
  virtual uint64_t seed() { return gen_->Seed(); }

  /// Set the Philox offset (supported in CUDA / MPS generators).
  virtual void set_offset(uint64_t offset) {
    // phi::Generator stores offset in its state; we reset it via
    // IncrementOffset after backing up the current offset.
    auto state = gen_->GetState();
    uint64_t cur = state.offset;
    if (offset > cur) {
      gen_->IncrementOffset(offset - cur);
    } else {
      // To move backwards, we need to reset the state.
      state.offset = offset;
      gen_->SetState(state);
    }
  }

  virtual uint64_t get_offset() const { return gen_->GetCurrentOffset(); }

  // ------- device / dispatch ------------------------------------------------

  Device device() const { return device_; }

  DispatchKeySet key_set() const {
    auto dt = device_.type();
    if (dt == kCPU) {
      return DispatchKeySet(DispatchKey::CPU);
    } else if (dt == kCUDA) {
      return DispatchKeySet(DispatchKey::CUDA);
    }
    return DispatchKeySet();
  }

  // ------- clone ------------------------------------------------------------

  virtual intrusive_ptr<GeneratorImpl> clone() const {
    auto state = gen_->GetState();
    auto new_gen = std::make_shared<phi::Generator>(state.seed);
    new_gen->SetState(state);
    auto impl = make_intrusive<GeneratorImpl>(device_, new_gen);
    return impl;
  }

  // ------- mutex (for thread-safe usage) ------------------------------------

  /// Public mutex for external locking (see PyTorch Note [Acquire lock ...]).
  std::mutex mutex_;

  // ------- PyObject binding -------------------------------------------------

  void set_pyobj(PyObject* pyobj) noexcept { pyobj_ = pyobj; }
  PyObject* pyobj() const noexcept { return pyobj_; }

  // ------- internal accessor ------------------------------------------------

  /// Access the underlying Paddle generator.
  std::shared_ptr<phi::Generator> paddle_generator() const { return gen_; }

 protected:
  std::shared_ptr<phi::Generator> gen_;
  Device device_;
  PyObject* pyobj_;
};

}  // namespace c10
