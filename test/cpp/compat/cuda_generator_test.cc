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

// graphsafe_set_state / graphsafe_get_state round-trip.
// NOTE: createCUDAGenerator on the same device shares the same underlying
// phi::DefaultCUDAGenerator, so set_current_seed on one affects the other.
// We use clone() to create truly independent generators for this test.
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
