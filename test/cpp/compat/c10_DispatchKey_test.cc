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

#include <c10/core/DispatchKey.h>

#include <cstdint>
#include <functional>

#include "gtest/gtest.h"

// Use EXPECT_TRUE with == instead of EXPECT_EQ for enum types
// because gtest's EXPECT_EQ needs operator<< which is declared
// but not yet implemented for DispatchKey/BackendComponent.
#define EXPECT_DK_EQ(a, b) EXPECT_TRUE((a) == (b))
#define EXPECT_BC_EQ(a, b) EXPECT_TRUE((a) == (b))

// Helper: wrap in a non-constexpr call to force runtime evaluation.
template <typename F, typename... Args>
auto runtime_call(F f, Args... args) -> decltype(f(args...)) {
  return f(args...);
}

// ==========================================================================
// Tests for constexpr/inline functions in c10::DispatchKey.h
// ==========================================================================

// ---------- isAliasDispatchKey
// ------------------------------------------------

TEST(DispatchKeyTest, IsAliasDispatchKey) {
  // Alias keys: Autograd .. CompositeExplicitAutogradNonFunctional
  c10::DispatchKey alias = c10::DispatchKey::Autograd;
  EXPECT_TRUE(c10::isAliasDispatchKey(alias));

  alias = c10::DispatchKey::CompositeImplicitAutograd;
  EXPECT_TRUE(c10::isAliasDispatchKey(alias));

  alias = c10::DispatchKey::CompositeExplicitAutograd;
  EXPECT_TRUE(c10::isAliasDispatchKey(alias));

  alias = c10::DispatchKey::CompositeExplicitAutogradNonFunctional;
  EXPECT_TRUE(c10::isAliasDispatchKey(alias));

  // Non-alias keys
  c10::DispatchKey non_alias = c10::DispatchKey::CPU;
  EXPECT_FALSE(c10::isAliasDispatchKey(non_alias));

  non_alias = c10::DispatchKey::Dense;
  EXPECT_FALSE(c10::isAliasDispatchKey(non_alias));

  non_alias = c10::DispatchKey::Undefined;
  EXPECT_FALSE(c10::isAliasDispatchKey(non_alias));
}

// ---------- isPerBackendFunctionalityKey -------------------------------------

TEST(DispatchKeyTest, IsPerBackendFunctionalityKey) {
  // Per-backend keys: Dense, Quantized, Sparse, SparseCsr,
  //                   AutogradFunctionality, NestedTensor
  c10::DispatchKey k = c10::DispatchKey::Dense;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::Quantized;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::Sparse;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::SparseCsr;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::AutogradFunctionality;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::NestedTensor;
  EXPECT_TRUE(c10::isPerBackendFunctionalityKey(k));

  // Non per-backend keys
  k = c10::DispatchKey::BackendSelect;
  EXPECT_FALSE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::Python;
  EXPECT_FALSE(c10::isPerBackendFunctionalityKey(k));

  k = c10::DispatchKey::Undefined;
  EXPECT_FALSE(c10::isPerBackendFunctionalityKey(k));
}

// ---------- toBackendComponent(DispatchKey)
// -----------------------------------

TEST(DispatchKeyTest, ToBackendComponentDense) {
  c10::DispatchKey k = c10::DispatchKey::CPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentQuantized) {
  c10::DispatchKey k = c10::DispatchKey::QuantizedCPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentSparse) {
  c10::DispatchKey k = c10::DispatchKey::SparseCPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentSparseCsr) {
  c10::DispatchKey k = c10::DispatchKey::SparseCsrCPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentNestedTensor) {
  c10::DispatchKey k = c10::DispatchKey::NestedTensorCPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentAutograd) {
  c10::DispatchKey k = c10::DispatchKey::AutogradCPU;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::CPUBit);
}

TEST(DispatchKeyTest, ToBackendComponentInvalid) {
  // A functionality-only key (no backend range) → InvalidBit
  c10::DispatchKey k = c10::DispatchKey::BackendSelect;
  c10::BackendComponent bc = c10::toBackendComponent(k);
  EXPECT_BC_EQ(bc, c10::BackendComponent::InvalidBit);
}

// ---------- toFunctionalityKey -----------------------------------------------

TEST(DispatchKeyTest, ToFunctionalityKeyPure) {
  // A pure functionality key maps to itself.
  c10::DispatchKey k = c10::DispatchKey::Dense;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Dense);

  k = c10::DispatchKey::BackendSelect;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::BackendSelect);
}

TEST(DispatchKeyTest, ToFunctionalityKeyDenseBackend) {
  c10::DispatchKey k = c10::DispatchKey::CPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Dense);

  k = c10::DispatchKey::CUDA;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Dense);
}

TEST(DispatchKeyTest, ToFunctionalityKeyQuantizedBackend) {
  c10::DispatchKey k = c10::DispatchKey::QuantizedCPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Quantized);
}

TEST(DispatchKeyTest, ToFunctionalityKeySparseBackend) {
  c10::DispatchKey k = c10::DispatchKey::SparseCPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Sparse);
}

TEST(DispatchKeyTest, ToFunctionalityKeySparseCsrBackend) {
  c10::DispatchKey k = c10::DispatchKey::SparseCsrCPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::SparseCsr);
}

TEST(DispatchKeyTest, ToFunctionalityKeyNestedTensorBackend) {
  c10::DispatchKey k = c10::DispatchKey::NestedTensorCPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::NestedTensor);
}

TEST(DispatchKeyTest, ToFunctionalityKeyAutogradBackend) {
  c10::DispatchKey k = c10::DispatchKey::AutogradCPU;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k),
               c10::DispatchKey::AutogradFunctionality);
}

TEST(DispatchKeyTest, ToFunctionalityKeyBeyondRuntime) {
  // Keys beyond EndOfRuntimeBackendKeys map to Undefined.
  c10::DispatchKey k = c10::DispatchKey::Autograd;
  EXPECT_DK_EQ(c10::toFunctionalityKey(k), c10::DispatchKey::Undefined);
}

// ---------- toRuntimePerBackendFunctionalityKey ------------------------------

TEST(DispatchKeyTest, ToRuntimeKeyDense) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::Dense, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::CPU);
}

TEST(DispatchKeyTest, ToRuntimeKeySparse) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::Sparse, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::SparseCPU);
}

TEST(DispatchKeyTest, ToRuntimeKeySparseCsr) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::SparseCsr, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::SparseCsrCPU);
}

TEST(DispatchKeyTest, ToRuntimeKeyQuantized) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::Quantized, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::QuantizedCPU);
}

TEST(DispatchKeyTest, ToRuntimeKeyNestedTensor) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::NestedTensor, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::NestedTensorCPU);
}

TEST(DispatchKeyTest, ToRuntimeKeyAutograd) {
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::AutogradFunctionality, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::AutogradCPU);
}

TEST(DispatchKeyTest, ToRuntimeKeyNonPerBackend) {
  // A key that is not per-backend should map to Undefined.
  c10::DispatchKey k = c10::toRuntimePerBackendFunctionalityKey(
      c10::DispatchKey::BackendSelect, c10::BackendComponent::CPUBit);
  EXPECT_DK_EQ(k, c10::DispatchKey::Undefined);
}

// ---------- std::hash<DispatchKey> -------------------------------------------

TEST(DispatchKeyTest, Hash) {
  std::hash<c10::DispatchKey> hasher;
  c10::DispatchKey k = c10::DispatchKey::CPU;
  EXPECT_EQ(hasher(k), static_cast<size_t>(k));
}
