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

#include <c10/core/GeneratorImpl.h>

#include <cstdint>
#include <memory>

#include "gtest/gtest.h"
#include "paddle/phi/core/generator.h"

// ============================================================================
// Tests for c10::GeneratorImpl (base class)
// ============================================================================

// ---------- Construction ----------------------------------------------------

TEST(GeneratorImplTest, ConstructWithNullptrCreatesDefaultGen) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  ASSERT_NE(impl.paddle_generator(), nullptr);
}

TEST(GeneratorImplTest, ConstructWithExistingGen) {
  auto gen = std::make_shared<phi::Generator>(42u);
  c10::GeneratorImpl impl{c10::Device(c10::kCPU), gen};
  ASSERT_EQ(impl.paddle_generator(), gen);
  ASSERT_EQ(impl.current_seed(), 42u);
}

// ---------- Seed / offset API (base-class versions) -------------------------

TEST(GeneratorImplTest, SetAndGetCurrentSeed) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  impl.set_current_seed(12345);
  ASSERT_EQ(impl.current_seed(), 12345u);
}

TEST(GeneratorImplTest, SeedGeneratesNewSeed) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  impl.set_current_seed(1);
  uint64_t new_seed = impl.seed();
  // seed() should return a new random seed (very unlikely to be 1 again).
  // We just verify it returns *something* and updates current_seed.
  ASSERT_EQ(impl.current_seed(), new_seed);
}

TEST(GeneratorImplTest, GetOffsetInitiallyZero) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  ASSERT_EQ(impl.get_offset(), 0u);
}

TEST(GeneratorImplTest, SetOffsetForward) {
  auto gen = std::make_shared<phi::Generator>(100u);
  c10::GeneratorImpl impl{c10::Device(c10::kCUDA, 0), gen};

  impl.set_offset(10);
  ASSERT_EQ(impl.get_offset(), 10u);
}

TEST(GeneratorImplTest, SetOffsetBackward) {
  auto gen = std::make_shared<phi::Generator>(100u);
  c10::GeneratorImpl impl{c10::Device(c10::kCUDA, 0), gen};

  impl.set_offset(20);
  ASSERT_EQ(impl.get_offset(), 20u);

  impl.set_offset(5);
  ASSERT_EQ(impl.get_offset(), 5u);
}

TEST(GeneratorImplTest, SetOffsetSameValue) {
  auto gen = std::make_shared<phi::Generator>(100u);
  c10::GeneratorImpl impl{c10::Device(c10::kCUDA, 0), gen};

  impl.set_offset(10);
  impl.set_offset(10);
  ASSERT_EQ(impl.get_offset(), 10u);
}

// ---------- Device / DispatchKeySet -----------------------------------------

TEST(GeneratorImplTest, DeviceReturnsCorrectDevice) {
  c10::Device cpu_dev{c10::kCPU};
  c10::GeneratorImpl impl{cpu_dev};
  ASSERT_EQ(impl.device(), cpu_dev);
}

TEST(GeneratorImplTest, KeySetCPU) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  c10::DispatchKeySet ks = impl.key_set();
  ASSERT_TRUE(ks.has(c10::DispatchKey::CPU));
}

TEST(GeneratorImplTest, KeySetCUDA) {
  c10::GeneratorImpl impl{c10::Device(c10::kCUDA, 0)};
  c10::DispatchKeySet ks = impl.key_set();
  ASSERT_TRUE(ks.has(c10::DispatchKey::CUDA));
}

TEST(GeneratorImplTest, KeySetOtherDevice) {
  // Use kCUSTOM which is neither CPU nor CUDA to exercise the fallback
  // branch that returns an empty DispatchKeySet.
  c10::GeneratorImpl impl{c10::Device(c10::kCUSTOM, 0)};
  c10::DispatchKeySet ks = impl.key_set();
  ASSERT_FALSE(ks.has(c10::DispatchKey::CPU));
  ASSERT_FALSE(ks.has(c10::DispatchKey::CUDA));
}

// ---------- Clone -----------------------------------------------------------

TEST(GeneratorImplTest, ClonePreservesState) {
  auto gen = std::make_shared<phi::Generator>(42u);
  c10::GeneratorImpl impl{c10::Device(c10::kCPU), gen};
  impl.set_current_seed(777);

  auto cloned = impl.clone();
  ASSERT_NE(cloned.get(), nullptr);
  ASSERT_EQ(cloned->current_seed(), 777u);
  ASSERT_EQ(cloned->device(), c10::Device(c10::kCPU));

  cloned->set_current_seed(888);
  ASSERT_EQ(impl.current_seed(), 777u);
  ASSERT_EQ(cloned->current_seed(), 888u);
}

// ---------- PyObject binding ------------------------------------------------

TEST(GeneratorImplTest, PyObjDefaultNull) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};
  ASSERT_EQ(impl.pyobj(), nullptr);
}

TEST(GeneratorImplTest, SetAndGetPyObj) {
  c10::GeneratorImpl impl{c10::Device(c10::kCPU)};

  // Use a dummy pointer (we never dereference it).
  int dummy = 0;
  auto* fake_pyobj = reinterpret_cast<PyObject*>(&dummy);

  impl.set_pyobj(fake_pyobj);
  ASSERT_EQ(impl.pyobj(), fake_pyobj);
}

// ---------- intrusive_ptr refcount semantics --------------------------------

TEST(GeneratorImplTest, MakeIntrusiveInitialRefcountIsOne) {
  auto ptr = c10::make_intrusive<c10::GeneratorImpl>(c10::Device(c10::kCPU));
  ASSERT_EQ(ptr.use_count(), 1u);
}

TEST(GeneratorImplTest, CopyIntrusivePtrIncrementsRefcount) {
  auto ptr = c10::make_intrusive<c10::GeneratorImpl>(c10::Device(c10::kCPU));
  ASSERT_EQ(ptr.use_count(), 1u);
  {
    auto copy = ptr;
    ASSERT_EQ(ptr.use_count(), 2u);
    ASSERT_EQ(copy.use_count(), 2u);
  }
  ASSERT_EQ(ptr.use_count(), 1u);
}

TEST(GeneratorImplTest, MoveIntrusivePtrKeepsRefcount) {
  auto ptr = c10::make_intrusive<c10::GeneratorImpl>(c10::Device(c10::kCPU));
  c10::GeneratorImpl* raw = ptr.get();
  auto moved = std::move(ptr);
  ASSERT_FALSE(ptr.defined());
  ASSERT_EQ(moved.use_count(), 1u);
  ASSERT_EQ(moved.get(), raw);
}

// ---------- Internal accessor -----------------------------------------------

TEST(GeneratorImplTest, PaddleGeneratorAccessor) {
  auto gen = std::make_shared<phi::Generator>(99u);
  c10::GeneratorImpl impl{c10::Device(c10::kCPU), gen};
  ASSERT_EQ(impl.paddle_generator(), gen);
  ASSERT_EQ(impl.paddle_generator()->GetCurrentSeed(), 99u);
}
