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
#include <c10/core/GeneratorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <optional>
#include <utility>

/**
 * Note [Generator]
 * ~~~~~~~~~~~~~~~~
 * A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm
 * to generate a seemingly random sequence of numbers, that may be later be
 * used in creating a random distribution. Such an engine almost always
 * maintains a state and requires a seed to start off the creation of random
 * numbers. Often times, users have found it beneficial to be able to
 * explicitly create, retain, and destroy PRNG states and also be able to
 * have control over the seed value.
 *
 * A Generator in ATen gives users the ability to read, write and modify a
 * PRNG engine. For instance, it does so by letting users seed a PRNG engine,
 * fork the state of the engine, etc.
 *
 * By default, there is one generator per device, and a device's generator is
 * lazily created. A user can use the torch.Generator() api to create their
 * own generator.
 *
 * This implementation wraps Paddle's phi::Generator via c10::GeneratorImpl.
 */

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please use the
 * public mutex_ when using any methods from these classes, except for the
 * read-only methods.
 */

// Forward declare PyObject if not already available.
#ifndef PyObject
struct _object;
using PyObject = _object;
#endif

namespace at {

using c10::Device;
using c10::DispatchKeySet;

class Tensor;

struct Generator {
  Generator() = default;

  explicit Generator(c10::intrusive_ptr<c10::GeneratorImpl> gen_impl)
      : impl_(std::move(gen_impl)) {
    TORCH_CHECK(impl_.get(), "GeneratorImpl with nullptr is not supported");
  }

  bool operator==(const Generator& rhs) const {
    return this->impl_ == rhs.impl_;
  }

  bool operator!=(const Generator& rhs) const { return !((*this) == rhs); }

  bool defined() const { return static_cast<bool>(impl_); }

  c10::GeneratorImpl* unsafeGetGeneratorImpl() const { return impl_.get(); }

  c10::GeneratorImpl* unsafeReleaseGeneratorImpl() { return impl_.release(); }

  const c10::intrusive_ptr<c10::GeneratorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  void set_current_seed(uint64_t seed) { impl_->set_current_seed(seed); }

  /// Sets the offset of Generator state to the desired offset.
  /// Supported for Philox based Generators (CUDA / MPS).
  void set_offset(uint64_t offset) { impl_->set_offset(offset); }

  /// Returns the offset of Generator state.
  /// Supported for Philox based Generators (CUDA / MPS).
  uint64_t get_offset() const { return impl_->get_offset(); }

  uint64_t current_seed() const { return impl_->current_seed(); }

  uint64_t seed() { return impl_->seed(); }

  // ----- state transfer (not inlined to break header cycles) ----------------
  // These methods mirror PyTorch's set_state / get_state which operate on
  // serialised byte tensors.  In the Paddle compat layer we provide a simpler
  // state-copy semantic through graphsafe_set_state / graphsafe_get_state.

  /// Copy the full PRNG state from another Generator.
  void graphsafe_set_state(const Generator& src) {
    TORCH_CHECK(src.defined(), "Source generator is not defined");
    TORCH_CHECK(defined(), "Target generator is not defined");
    auto src_state = src.impl_->paddle_generator()->GetState();
    impl_->paddle_generator()->SetState(src_state);
  }

  /// Obtain a Generator whose state is a snapshot (clone) of this one.
  Generator graphsafe_get_state() const {
    TORCH_CHECK(defined(), "Generator is not defined");
    return clone();
  }

  std::mutex& mutex() { return impl_->mutex_; }

  DispatchKeySet key_set() const { return impl_->key_set(); }

  Device device() const { return impl_->device(); }

  inline void set_pyobj(PyObject* pyobj) const noexcept {
    impl_->set_pyobj(pyobj);
  }

  inline PyObject* pyobj() const noexcept { return impl_->pyobj(); }

  template <typename T>
  T* get() const {
    return static_cast<T*>(impl_.get());
  }

  Generator clone() const { return Generator(impl_->clone()); }

  /// Access the underlying Paddle phi::Generator (convenience).
  std::shared_ptr<phi::Generator> paddle_generator() const {
    return impl_->paddle_generator();
  }

 private:
  c10::intrusive_ptr<c10::GeneratorImpl> impl_;
};

template <class Impl, class... Args>
Generator make_generator(Args&&... args) {
  return Generator(c10::make_intrusive<Impl>(std::forward<Args>(args)...));
}

/**
 * Utility function to static cast input Generator to
 * the backend generator type (CPUGeneratorImpl / CUDAGeneratorImpl etc.)
 */
template <typename T>
inline T* check_generator(std::optional<Generator> gen) {
  TORCH_CHECK(gen.has_value(), "Expected Generator but received nullopt");
  TORCH_CHECK(gen->defined(),
              "Generator with undefined implementation is not allowed");
  TORCH_CHECK(
      T::device_type() == gen->device().type(),
      "Expected a generator for ",
      phi::AllocationTypeStr(c10::DeviceTypeToPhi(T::device_type())),
      " but found one for ",
      phi::AllocationTypeStr(c10::DeviceTypeToPhi(gen->device().type())));
  return gen->get<T>();
}

/**
 * Utility function used in tensor implementations, which supplies the
 * default generator to tensors if an input generator is not supplied.
 * The input Generator is also static-cast to the backend generator type.
 */
template <typename T>
inline T* get_generator_or_default(const std::optional<Generator>& gen,
                                   const Generator& default_gen) {
  return gen.has_value() && gen->defined() ? check_generator<T>(gen)
                                           : check_generator<T>(default_gen);
}

}  // namespace at
