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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <optional>

#include "gtest/gtest.h"
#include "paddle/common/exception.h"

// ============================================================================
// Tests for at::get_generator_or_default<at::CUDAGeneratorImpl>
// ============================================================================

// Verify that getDefaultCUDAGenerator returns a valid, defined Generator whose
// underlying impl is a CUDAGeneratorImpl on device 0.
TEST(CUDAGeneratorTest, DefaultGeneratorIsDefined) {
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);
  ASSERT_TRUE(default_gen.defined());
  ASSERT_EQ(default_gen.device().type(), at::kCUDA);
  ASSERT_EQ(default_gen.device().index(), 0);
}

// get_generator_or_default should return the default generator when the
// optional is empty (nullopt).
TEST(CUDAGeneratorTest, GetGeneratorOrDefaultWithNullopt) {
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);

  std::optional<at::Generator> empty_gen = std::nullopt;
  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(empty_gen,
                                                          default_gen);

  ASSERT_NE(impl, nullptr);
  ASSERT_EQ(impl->device().type(), at::kCUDA);
}

// get_generator_or_default should return the default generator when the
// optional holds a default-constructed (undefined) Generator.
TEST(CUDAGeneratorTest, GetGeneratorOrDefaultWithUndefined) {
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);

  std::optional<at::Generator> undef_gen = at::Generator();  // undefined
  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(undef_gen,
                                                          default_gen);

  ASSERT_NE(impl, nullptr);
  ASSERT_EQ(impl->device().type(), at::kCUDA);
}

// get_generator_or_default should return the user-supplied generator when the
// optional contains a valid (defined) Generator.
TEST(CUDAGeneratorTest, GetGeneratorOrDefaultWithUserGenerator) {
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);

  // Create a distinct user generator.
  at::Generator user_gen = at::cuda::detail::createCUDAGenerator(0);
  user_gen.set_current_seed(42);

  std::optional<at::Generator> opt_gen = user_gen;
  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(opt_gen, default_gen);

  ASSERT_NE(impl, nullptr);
  ASSERT_EQ(impl->current_seed(), 42u);
}

// Verify that check_generator works for a valid optional<Generator>.
TEST(CUDAGeneratorTest, CheckGenerator) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(123);

  std::optional<at::Generator> opt = gen;
  at::CUDAGeneratorImpl* impl = at::check_generator<at::CUDAGeneratorImpl>(opt);

  ASSERT_NE(impl, nullptr);
  ASSERT_EQ(impl->current_seed(), 123u);
}

// check_generator should throw when given nullopt.
TEST(CUDAGeneratorTest, CheckGeneratorThrowsOnNullopt) {
  std::optional<at::Generator> empty;
  EXPECT_THROW(at::check_generator<at::CUDAGeneratorImpl>(empty),
               ::common::PD_Exception);
}

// check_generator should throw when the optional holds a default-constructed
// (undefined) Generator — exercises the gen->defined() TORCH_CHECK branch.
TEST(CUDAGeneratorTest, CheckGeneratorThrowsOnUndefined) {
  std::optional<at::Generator> undef_gen = at::Generator();  // undefined impl
  EXPECT_THROW(at::check_generator<at::CUDAGeneratorImpl>(undef_gen),
               ::common::PD_Exception);
}

// Verify Philox state management via the CUDAGeneratorImpl pointer returned
// from get_generator_or_default.
TEST(CUDAGeneratorTest, PhiloxStateThroughGetGeneratorOrDefault) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(999);

  std::optional<at::Generator> opt = gen;
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);

  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(opt, default_gen);

  // Initial Philox offset should be 0.
  ASSERT_EQ(impl->philox_offset_per_thread(), 0u);

  // Advance via philox_engine_inputs.
  auto [seed, offset] = impl->philox_engine_inputs(4);
  ASSERT_EQ(seed, 999u);
  ASSERT_EQ(offset, 0u);
  ASSERT_EQ(impl->philox_offset_per_thread(), 4u);

  // Further advance via philox_cuda_state.
  at::PhiloxCudaState state = impl->philox_cuda_state(8);
  (void)state;  // Silence unused variable warning - state is used for its side
                // effect
  ASSERT_EQ(impl->philox_offset_per_thread(), 12u);
}

// Seed / offset round-trip through get_generator_or_default.
TEST(CUDAGeneratorTest, SeedOffsetRoundTrip) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);

  std::optional<at::Generator> opt = gen;
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);

  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(opt, default_gen);

  impl->set_current_seed(12345);
  ASSERT_EQ(impl->current_seed(), 12345u);

  impl->set_offset(100);
  ASSERT_EQ(impl->get_offset(), 100u);

  // seed() should reset the offset.
  uint64_t new_seed = impl->seed();
  ASSERT_EQ(impl->get_offset(), 0u);
  ASSERT_EQ(impl->current_seed(), new_seed);
}

// Clone via the Generator wrapper preserves state.
TEST(CUDAGeneratorTest, ClonePreservesState) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(777);

  at::CUDAGeneratorImpl* impl = gen.get<at::CUDAGeneratorImpl>();
  impl->set_philox_offset_per_thread(50);

  at::Generator cloned = gen.clone();
  at::CUDAGeneratorImpl* cloned_impl = cloned.get<at::CUDAGeneratorImpl>();

  ASSERT_EQ(cloned_impl->current_seed(), 777u);
  ASSERT_EQ(cloned_impl->philox_offset_per_thread(), 50u);

  // Modifying clone should not affect original.
  cloned_impl->set_current_seed(888);
  ASSERT_EQ(impl->current_seed(), 777u);
  ASSERT_EQ(cloned_impl->current_seed(), 888u);
}

// Verify that CUDAGeneratorImpl::device_type() returns kCUDA.
TEST(CUDAGeneratorTest, DeviceTypeStaticMethod) {
  ASSERT_EQ(at::CUDAGeneratorImpl::device_type(), at::kCUDA);
}

// Verify that constructing CUDAGeneratorImpl with default device_index (-1)
// uses the current GPU device.
TEST(CUDAGeneratorTest, DefaultDeviceIndex) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(-1);
  ASSERT_TRUE(gen.defined());
  ASSERT_EQ(gen.device().type(), at::kCUDA);
  // device index should be the current device (>= 0).
  ASSERT_GE(gen.device().index(), 0);
}

// Verify that getDefaultCUDAGenerator with default device (-1) resolves to
// the current GPU device.
TEST(CUDAGeneratorTest, GetDefaultCUDAGeneratorWithDefaultDevice) {
  const at::Generator& gen = at::cuda::detail::getDefaultCUDAGenerator(-1);
  ASSERT_TRUE(gen.defined());
  ASSERT_EQ(gen.device().type(), at::kCUDA);
  ASSERT_GE(gen.device().index(), 0);
}

// graphsafe_set_state / graphsafe_get_state round-trip.
TEST(CUDAGeneratorTest, GraphsafeStateTransfer) {
  at::Generator gen_a = at::cuda::detail::createCUDAGenerator(0);
  gen_a.set_current_seed(111);
  // Clone to get a generator with independent state.
  at::Generator gen_b = gen_a.clone();
  gen_b.set_current_seed(222);

  ASSERT_NE(gen_a.current_seed(), gen_b.current_seed());

  // Copy state from gen_a to gen_b.
  gen_b.graphsafe_set_state(gen_a);
  ASSERT_EQ(gen_b.current_seed(), 111u);

  // graphsafe_get_state returns a snapshot.
  at::Generator snapshot = gen_a.graphsafe_get_state();
  gen_a.set_current_seed(333);
  ASSERT_EQ(snapshot.current_seed(), 111u);
  ASSERT_EQ(gen_a.current_seed(), 333u);
}

// ============================================================================
// Test for createCUDAGenerator independence (AC-2 verification)
// ============================================================================

// Verify that createCUDAGenerator creates a generator with independent state
// that does not share RNG state with the default generator.
TEST(CUDAGeneratorTest, CreateGeneratorDoesNotShareDefaultState) {
  // Get the default generator and set its seed.
  at::Generator default_gen = at::cuda::detail::getDefaultCUDAGenerator(0);
  default_gen.set_current_seed(1000);
  ASSERT_EQ(default_gen.current_seed(), 1000u);

  // Create a user generator and set a different seed.
  at::Generator user_gen = at::cuda::detail::createCUDAGenerator(0);
  user_gen.set_current_seed(2000);

  // Verify the user generator has the new seed.
  ASSERT_EQ(user_gen.current_seed(), 2000u);

  // Verify the default generator's seed is unchanged (independence).
  ASSERT_EQ(default_gen.current_seed(), 1000u);

  // Now change the default generator's seed and verify user is unaffected.
  default_gen.set_current_seed(3000);
  ASSERT_EQ(default_gen.current_seed(), 3000u);
  ASSERT_EQ(user_gen.current_seed(), 2000u);  // Still 2000, not affected
}

// ============================================================================
// Tests for unsafeReleaseGeneratorImpl (AC-3 verification)
// ============================================================================

// Verify that unsafeReleaseGeneratorImpl transfers ownership and makes
// the generator undefined.
TEST(CUDAGeneratorTest, UnsafeReleaseMakesGeneratorUndefined) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(42);
  ASSERT_TRUE(gen.defined());

  // Release the implementation - this transfers ownership to us.
  c10::GeneratorImpl* raw_impl = gen.unsafeReleaseGeneratorImpl();
  ASSERT_NE(raw_impl, nullptr);

  // After release, generator should be undefined.
  ASSERT_FALSE(gen.defined());

  // We can still access the released impl via the raw pointer.
  ASSERT_EQ(raw_impl->current_seed(), 42u);

  // Clean up: properly delete the released implementation.
  delete raw_impl;
}

// Verify that the released pointer can be reclaimed into a new intrusive_ptr
// without double-free.
TEST(CUDAGeneratorTest, UnsafeReleaseAndReclaim) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(123);
  ASSERT_TRUE(gen.defined());

  // Release the implementation.
  c10::GeneratorImpl* raw_impl = gen.unsafeReleaseGeneratorImpl();
  ASSERT_NE(raw_impl, nullptr);
  ASSERT_FALSE(gen.defined());

  // Reclaim the raw pointer into a new intrusive_ptr.
  // This should not cause double-free or crashes.
  c10::intrusive_ptr<c10::GeneratorImpl> reclaimed(
      c10::intrusive_ptr<c10::GeneratorImpl>::reclaim(raw_impl));
  ASSERT_TRUE(reclaimed.defined());
  ASSERT_EQ(reclaimed->current_seed(), 123u);

  // reclaimed will be properly destroyed when it goes out of scope.
}

// Verify that the generator is undefined after release and can be properly
// reclaimed.
TEST(CUDAGeneratorTest, UnsafeReleaseAndReclaimRoundTrip) {
  at::Generator gen = at::cuda::detail::createCUDAGenerator(0);
  gen.set_current_seed(789);
  ASSERT_TRUE(gen.defined());

  // Release ownership.
  c10::GeneratorImpl* raw_impl = gen.unsafeReleaseGeneratorImpl();
  ASSERT_NE(raw_impl, nullptr);
  ASSERT_FALSE(gen.defined());

  // Verify we can access the impl via raw pointer.
  ASSERT_EQ(raw_impl->current_seed(), 789u);

  // Reclaim into a new intrusive_ptr.
  c10::intrusive_ptr<c10::GeneratorImpl> reclaimed(
      c10::intrusive_ptr<c10::GeneratorImpl>::reclaim(raw_impl));
  ASSERT_TRUE(reclaimed.defined());

  // Create a new Generator from the reclaimed impl.
  at::Generator new_gen(reclaimed);
  ASSERT_TRUE(new_gen.defined());
  ASSERT_EQ(new_gen.current_seed(), 789u);

  // Modifying new_gen should not affect the old (already undefined) gen.
  new_gen.set_current_seed(999);
  ASSERT_EQ(new_gen.current_seed(), 999u);
}

// ============================================================================
// Tests for check_generator device_type validation
// ============================================================================

// check_generator should throw when the generator's device type does not match
// the requested implementation type (CPU generator passed where CUDA expected).
TEST(CUDAGeneratorTest, CheckGeneratorThrowsOnDeviceTypeMismatch) {
  // Create a CPU generator (device_type = kCPU).
  auto cpu_gen =
      c10::make_intrusive<c10::GeneratorImpl>(c10::Device(c10::kCPU));
  at::Generator cpu_wrapper(cpu_gen);
  std::optional<at::Generator> opt = cpu_wrapper;

  // Requesting CUDAGeneratorImpl from a CPU generator should throw.
  EXPECT_THROW(at::check_generator<at::CUDAGeneratorImpl>(opt),
               ::common::PD_Exception);
}

// check_generator with matching device type should succeed.
TEST(CUDAGeneratorTest, CheckGeneratorSucceedsWithMatchingDeviceType) {
  at::Generator cuda_gen = at::cuda::detail::createCUDAGenerator(0);
  cuda_gen.set_current_seed(555);
  std::optional<at::Generator> opt = cuda_gen;

  at::CUDAGeneratorImpl* impl = at::check_generator<at::CUDAGeneratorImpl>(opt);
  ASSERT_NE(impl, nullptr);
  ASSERT_EQ(impl->current_seed(), 555u);
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
