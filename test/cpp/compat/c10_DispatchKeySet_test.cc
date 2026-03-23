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

#include <c10/core/DispatchKeySet.h>

#include <cstdint>

#include "gtest/gtest.h"

// Use EXPECT_TRUE with == instead of EXPECT_EQ for enum types
// because gtest's EXPECT_EQ needs operator<< which is declared
// but not yet implemented for DispatchKey/BackendComponent.
#define EXPECT_DK_EQ(a, b) EXPECT_TRUE((a) == (b))
#define EXPECT_BC_EQ(a, b) EXPECT_TRUE((a) == (b))

// ==========================================================================
// Tests for c10::DispatchKeySet inline/constexpr methods
// ==========================================================================

// ---------- Constructors ----------------------------------------------------

TEST(DispatchKeySetTest, DefaultConstructorEmpty) {
  c10::DispatchKeySet ks;
  EXPECT_TRUE(ks.empty());
  EXPECT_EQ(ks.raw_repr(), 0u);
}

TEST(DispatchKeySetTest, ConstructFromDispatchKeyUndefined) {
  c10::DispatchKeySet ks(c10::DispatchKey::Undefined);
  EXPECT_TRUE(ks.empty());
}

TEST(DispatchKeySetTest, ConstructFromFunctionalityKey) {
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  EXPECT_FALSE(ks.empty());
  EXPECT_TRUE(ks.has(c10::DispatchKey::Dense));
}

TEST(DispatchKeySetTest, ConstructFromRuntimeBackendKey) {
  // CPU is a per-backend key (Dense + CPUBit).
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  EXPECT_FALSE(ks.empty());
  EXPECT_TRUE(ks.has(c10::DispatchKey::CPU));
  EXPECT_TRUE(ks.has_backend(c10::BackendComponent::CPUBit));
}

TEST(DispatchKeySetTest, ConstructFromBackendComponent) {
  c10::DispatchKeySet ks(c10::BackendComponent::CUDABit);
  EXPECT_TRUE(ks.has_backend(c10::BackendComponent::CUDABit));
}

TEST(DispatchKeySetTest, ConstructFromBackendComponentInvalid) {
  c10::DispatchKeySet ks(c10::BackendComponent::InvalidBit);
  EXPECT_TRUE(ks.empty());
}

TEST(DispatchKeySetTest, ConstructFromInitializerList) {
  c10::DispatchKeySet ks({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  EXPECT_TRUE(ks.has(c10::DispatchKey::Dense));
  EXPECT_TRUE(ks.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, ConstructFromBackendInitializerList) {
  c10::DispatchKeySet ks(
      {c10::BackendComponent::CPUBit, c10::BackendComponent::CUDABit});
  EXPECT_TRUE(ks.has_backend(c10::BackendComponent::CPUBit));
  EXPECT_TRUE(ks.has_backend(c10::BackendComponent::CUDABit));
}

TEST(DispatchKeySetTest, ConstructFull) {
  c10::DispatchKeySet ks(c10::DispatchKeySet::FULL);
  EXPECT_FALSE(ks.empty());
}

TEST(DispatchKeySetTest, ConstructFullAfter) {
  c10::DispatchKeySet ks(c10::DispatchKeySet::FULL_AFTER,
                         c10::DispatchKey::AutogradOther);
  EXPECT_FALSE(ks.empty());
}

TEST(DispatchKeySetTest, ConstructRaw) {
  c10::DispatchKeySet ks(c10::DispatchKeySet::RAW, 0x42);
  EXPECT_EQ(ks.raw_repr(), 0x42u);
}

// ---------- Key beyond runtime range ----------------------------------------

TEST(DispatchKeySetTest, ConstructFromKeyBeyondRuntime) {
  // Keys beyond EndOfRuntimeBackendKeys should produce an empty set.
  c10::DispatchKeySet ks(c10::DispatchKey::Autograd);
  EXPECT_TRUE(ks.empty());
}

// ---------- has / has_all / has_any / has_backend ----------------------------

TEST(DispatchKeySetTest, HasAll) {
  c10::DispatchKeySet ks({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet sub(c10::DispatchKey::Dense);
  EXPECT_TRUE(ks.has_all(sub));
}

TEST(DispatchKeySetTest, HasAnyFunctionality) {
  c10::DispatchKeySet ks({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet query(c10::DispatchKey::Sparse);
  EXPECT_TRUE(ks.has_any(query));
}

TEST(DispatchKeySetTest, IsSupersetOf) {
  c10::DispatchKeySet full({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet sub(c10::DispatchKey::Dense);
  EXPECT_TRUE(full.isSupersetOf(sub));
  EXPECT_FALSE(sub.isSupersetOf(full));
}

// ---------- Operators -------------------------------------------------------

TEST(DispatchKeySetTest, OperatorOr) {
  c10::DispatchKeySet a(c10::DispatchKey::Dense);
  c10::DispatchKeySet b(c10::DispatchKey::Sparse);
  auto combined = a | b;
  EXPECT_TRUE(combined.has(c10::DispatchKey::Dense));
  EXPECT_TRUE(combined.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, OperatorAnd) {
  c10::DispatchKeySet a({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet b(c10::DispatchKey::Dense);
  auto result = a & b;
  EXPECT_TRUE(result.has(c10::DispatchKey::Dense));
  EXPECT_FALSE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, OperatorXor) {
  c10::DispatchKeySet a({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet b(c10::DispatchKey::Dense);
  auto result = a ^ b;
  EXPECT_FALSE(result.has(c10::DispatchKey::Dense));
  EXPECT_TRUE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, OperatorMinus) {
  c10::DispatchKeySet a({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  c10::DispatchKeySet b(c10::DispatchKey::Dense);
  auto result = a - b;
  EXPECT_FALSE(result.has(c10::DispatchKey::Dense));
  // Sparse should remain.
  EXPECT_TRUE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, OperatorEqNeq) {
  c10::DispatchKeySet a(c10::DispatchKey::Dense);
  c10::DispatchKeySet b(c10::DispatchKey::Dense);
  c10::DispatchKeySet c(c10::DispatchKey::Sparse);
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_FALSE(a == c);
}

// ---------- add / remove / remove_backend -----------------------------------

TEST(DispatchKeySetTest, Add) {
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  auto result = ks.add(c10::DispatchKey::Sparse);
  EXPECT_TRUE(result.has(c10::DispatchKey::Dense));
  EXPECT_TRUE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, AddDispatchKeySet) {
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  auto result = ks.add(c10::DispatchKeySet(c10::DispatchKey::Sparse));
  EXPECT_TRUE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, Remove) {
  c10::DispatchKeySet ks({c10::DispatchKey::Dense, c10::DispatchKey::Sparse});
  auto result = ks.remove(c10::DispatchKey::Dense);
  EXPECT_FALSE(result.has(c10::DispatchKey::Dense));
  EXPECT_TRUE(result.has(c10::DispatchKey::Sparse));
}

TEST(DispatchKeySetTest, RemoveBackend) {
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  auto result = ks.remove_backend(c10::BackendComponent::CPUBit);
  EXPECT_FALSE(result.has_backend(c10::BackendComponent::CPUBit));
}

// ---------- empty / raw_repr / from_raw_repr --------------------------------

TEST(DispatchKeySetTest, FromRawRepr) {
  auto ks = c10::DispatchKeySet::from_raw_repr(0xFF);
  EXPECT_EQ(ks.raw_repr(), 0xFFu);
}

// ---------- highestFunctionalityKey / highestBackendKey ----------------------

TEST(DispatchKeySetTest, HighestFunctionalityKey) {
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  EXPECT_DK_EQ(ks.highestFunctionalityKey(), c10::DispatchKey::Dense);
}

TEST(DispatchKeySetTest, HighestFunctionalityKeyEmpty) {
  c10::DispatchKeySet ks;
  EXPECT_DK_EQ(ks.highestFunctionalityKey(), c10::DispatchKey::Undefined);
}

TEST(DispatchKeySetTest, HighestBackendKey) {
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  EXPECT_BC_EQ(ks.highestBackendKey(), c10::BackendComponent::CPUBit);
}

TEST(DispatchKeySetTest, HighestBackendKeyNoBackend) {
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  EXPECT_BC_EQ(ks.highestBackendKey(), c10::BackendComponent::InvalidBit);
}

TEST(DispatchKeySetTest, HighestPriorityTypeId) {
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  EXPECT_DK_EQ(ks.highestPriorityTypeId(), c10::DispatchKey::CPU);
}

TEST(DispatchKeySetTest, HighestPriorityTypeIdNonPerBackend) {
  c10::DispatchKeySet ks(c10::DispatchKey::BackendSelect);
  // BackendSelect is not per-backend, maps directly.
  EXPECT_DK_EQ(ks.highestPriorityTypeId(), c10::DispatchKey::BackendSelect);
}

// ---------- indexOfHighestBit ------------------------------------------------

TEST(DispatchKeySetTest, IndexOfHighestBitEmpty) {
  c10::DispatchKeySet ks;
  EXPECT_EQ(ks.indexOfHighestBit(), 0u);
}

TEST(DispatchKeySetTest, IndexOfHighestBitNonEmpty) {
  c10::DispatchKeySet ks(c10::DispatchKeySet::RAW, 0x8);
  EXPECT_EQ(ks.indexOfHighestBit(), 4u);  // bit 3 is set (0-indexed)
}

// ---------- getBackendIndex -------------------------------------------------

TEST(DispatchKeySetTest, GetBackendIndex) {
  c10::DispatchKeySet ks(c10::DispatchKey::CUDA);
  // CUDA should have a non-zero backend index (CPUBit=0, CUDABit=1).
  EXPECT_GT(ks.getBackendIndex(), 0u);
}

// ---------- getAutogradRelatedKeySetFromBackend ------------------------------

TEST(DispatchKeySetTest, AutogradRelatedKeySetCPU) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::CPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutogradCPU));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetCUDA) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::CUDABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutogradCUDA));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetXPU) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::XPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetXLA) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::XLABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetLazy) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::LazyBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetMeta) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::MetaBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetMPS) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::MPSBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetHPU) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::HPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetIPU) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::IPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetMTIA) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::MTIABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetMAIA) {
  auto ks =
      c10::getAutogradRelatedKeySetFromBackend(c10::BackendComponent::MAIABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetPrivateUse1) {
  auto ks = c10::getAutogradRelatedKeySetFromBackend(
      c10::BackendComponent::PrivateUse1Bit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetPrivateUse2) {
  auto ks = c10::getAutogradRelatedKeySetFromBackend(
      c10::BackendComponent::PrivateUse2Bit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetPrivateUse3) {
  auto ks = c10::getAutogradRelatedKeySetFromBackend(
      c10::BackendComponent::PrivateUse3Bit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
}

TEST(DispatchKeySetTest, AutogradRelatedKeySetDefault) {
  // InvalidBit falls through to default: inplace_or_view_ks |
  // autograd_other_ks.
  auto ks = c10::getAutogradRelatedKeySetFromBackend(
      c10::BackendComponent::InvalidBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutogradOther));
}

// ---------- getAutocastRelatedKeySetFromBackend ------------------------------

TEST(DispatchKeySetTest, AutocastRelatedKeySetCPU) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::CPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastCPU));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetCUDA) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::CUDABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastCUDA));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetXPU) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::XPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastXPU));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetIPU) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::IPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastIPU));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetHPU) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::HPUBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastHPU));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetXLA) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::XLABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastXLA));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetMPS) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::MPSBit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastMPS));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetMTIA) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::MTIABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastMTIA));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetMAIA) {
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::MAIABit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastMAIA));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetPrivateUse1) {
  auto ks = c10::getAutocastRelatedKeySetFromBackend(
      c10::BackendComponent::PrivateUse1Bit);
  EXPECT_TRUE(ks.has(c10::DispatchKey::AutocastPrivateUse1));
}

TEST(DispatchKeySetTest, AutocastRelatedKeySetDefault) {
  // Backends without a dedicated autocast key return empty.
  auto ks =
      c10::getAutocastRelatedKeySetFromBackend(c10::BackendComponent::LazyBit);
  EXPECT_TRUE(ks.empty());
}

// ---------- highestPriorityBackendTypeId ------------------------------------

TEST(DispatchKeySetTest, HighestPriorityBackendTypeId) {
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  auto key = c10::highestPriorityBackendTypeId(ks);
  EXPECT_DK_EQ(key, c10::DispatchKey::CPU);
}

// ---------- legacyExtractDispatchKey ----------------------------------------

TEST(DispatchKeySetTest, LegacyExtractDispatchKey) {
  c10::DispatchKeySet ks(c10::DispatchKey::CPU);
  auto key = c10::legacyExtractDispatchKey(ks);
  EXPECT_DK_EQ(key, c10::DispatchKey::CPU);
}
