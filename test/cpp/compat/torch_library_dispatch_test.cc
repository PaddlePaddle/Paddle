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

// Tests for the compat-layer dispatch key priority selection logic introduced
// in OperationInvoker::get_op_with_args (torch_compat.h).
//
// The lookup order is: CPU → BackendSelect → CatchAll.
// If none of those keys exist and exactly one implementation is registered it
// is used directly (deterministic). If multiple unrecognised keys exist the
// invoker raises an Ambiguous error rather than picking arbitrarily from an
// unordered_map (which has no stable iteration order).
// These tests exercise scenarios where the registrant uses BackendSelect
// (e.g. TORCH_LIBRARY_IMPL(..., BackendSelect, m)) so that the Python-facing
// invoker can reach it even when no CPU implementation exists.

#include <torch/library.h>

#include <vector>

#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// Operator implementations used by the tests below
// ---------------------------------------------------------------------------

namespace {

int backend_select_probe(int x) { return x + 10; }

int backend_select_and_cpu_cpu_fn(int x) { return x + 1; }
int backend_select_and_cpu_bs_fn(int x) { return x + 2; }

}  // namespace

int unique_non_preferred_fn(int x) { return x + 7; }
int ambiguous_cuda_fn(int x) { return x + 100; }
int ambiguous_xpu_fn(int x) { return x + 200; }

TORCH_LIBRARY(compat_dispatch_test_lib, m) {
  m.def("backend_select_only(int x) -> int");
  m.def("backend_select_and_cpu(int x) -> int");
  m.def("unique_non_preferred(int x) -> int");
  m.def("ambiguous_multi_key(int x) -> int");
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, BackendSelect, m) {
  m.impl("backend_select_only", &backend_select_probe);
  m.impl("backend_select_and_cpu", &backend_select_and_cpu_bs_fn);
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, CPU, m) {
  m.impl("backend_select_and_cpu", &backend_select_and_cpu_cpu_fn);
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, CUDA, m) {
  m.impl("unique_non_preferred", &unique_non_preferred_fn);
  m.impl("ambiguous_multi_key", &ambiguous_cuda_fn);
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, XPU, m) {
  m.impl("ambiguous_multi_key", &ambiguous_xpu_fn);
}

// ---------------------------------------------------------------------------
// Helper: simulate the priority-fallback lookup used by get_op_with_args
// ---------------------------------------------------------------------------

static decltype(torch::OperatorRegistry::instance()
                    .find_operator("")
                    ->implementations.end())
pick_impl(torch::OperatorRegistration* op) {
  using DK = torch::DispatchKey;
  const std::vector<DK> preferred_keys = {
      DK::CPU, DK::BackendSelect, DK::CatchAll};
  auto chosen = op->implementations.end();
  for (const auto& key : preferred_keys) {
    chosen = op->implementations.find(key);
    if (chosen != op->implementations.end()) break;
  }
  // Mirror the production rule: allow exactly-one-impl, reject ambiguous.
  if (chosen == op->implementations.end() && op->implementations.size() == 1) {
    chosen = op->implementations.begin();
  }
  return chosen;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// An operator registered only under BackendSelect must be queryable under
// that key and must NOT appear under CPU.
TEST(CompatTorchDispatchTest, BackendSelectOnlyRegistration) {
  const auto qname = "compat_dispatch_test_lib::backend_select_only";
  auto* op = torch::OperatorRegistry::instance().find_operator(qname);
  ASSERT_NE(op, nullptr);

  EXPECT_EQ(op->implementations.find(torch::DispatchKey::CPU),
            op->implementations.end());

  auto bs_it = op->implementations.find(torch::DispatchKey::BackendSelect);
  ASSERT_NE(bs_it, op->implementations.end());

  torch::FunctionArgs args;
  args.add_arg(torch::IValue(int64_t(32)));
  auto result = bs_it->second.call_with_args(args);
  ASSERT_TRUE(result.get_value().is_int());
  EXPECT_EQ(result.get_value().to_int(), 42);  // 32 + 10
}

// When CPU and BackendSelect are both registered, the priority lookup must
// pick CPU (higher priority in get_op_with_args).
TEST(CompatTorchDispatchTest, CpuPreferredOverBackendSelect) {
  const auto qname = "compat_dispatch_test_lib::backend_select_and_cpu";
  auto* op = torch::OperatorRegistry::instance().find_operator(qname);
  ASSERT_NE(op, nullptr);

  ASSERT_NE(op->implementations.find(torch::DispatchKey::CPU),
            op->implementations.end());
  ASSERT_NE(op->implementations.find(torch::DispatchKey::BackendSelect),
            op->implementations.end());

  auto chosen = pick_impl(op);
  ASSERT_NE(chosen, op->implementations.end());
  EXPECT_EQ(chosen->first, torch::DispatchKey::CPU);

  torch::FunctionArgs args;
  args.add_arg(torch::IValue(int64_t(41)));
  auto result = chosen->second.call_with_args(args);
  ASSERT_TRUE(result.get_value().is_int());
  EXPECT_EQ(result.get_value().to_int(), 42);  // CPU impl: x + 1
}

// When CPU is absent, the priority lookup must fall through to BackendSelect.
TEST(CompatTorchDispatchTest, BackendSelectPickedWhenCpuAbsent) {
  const auto qname = "compat_dispatch_test_lib::backend_select_only";
  auto* op = torch::OperatorRegistry::instance().find_operator(qname);
  ASSERT_NE(op, nullptr);

  auto chosen = pick_impl(op);
  ASSERT_NE(chosen, op->implementations.end());
  EXPECT_EQ(chosen->first, torch::DispatchKey::BackendSelect);

  torch::FunctionArgs args;
  args.add_arg(torch::IValue(int64_t(32)));
  auto result = chosen->second.call_with_args(args);
  ASSERT_TRUE(result.get_value().is_int());
  EXPECT_EQ(result.get_value().to_int(), 42);  // BackendSelect impl: x + 10
}

// An operator registered only under one non-preferred key (e.g. CUDA) must
// still be reachable when it's the sole implementation (deterministic).
TEST(CompatTorchDispatchTest, UniqueNonPreferredKeyIsCallable) {
  const auto qname = "compat_dispatch_test_lib::unique_non_preferred";
  auto* op = torch::OperatorRegistry::instance().find_operator(qname);
  ASSERT_NE(op, nullptr);
  ASSERT_EQ(op->implementations.size(), 1UL);

  auto chosen = pick_impl(op);
  ASSERT_NE(chosen, op->implementations.end());

  torch::FunctionArgs args;
  args.add_arg(torch::IValue(int64_t(35)));
  auto result = chosen->second.call_with_args(args);
  ASSERT_TRUE(result.get_value().is_int());
  EXPECT_EQ(result.get_value().to_int(), 42);  // unique impl: x + 7
}

// An operator with multiple non-preferred keys (CUDA + XPU) must produce
// end() from pick_impl (the production code would raise an Ambiguous error).
TEST(CompatTorchDispatchTest, AmbiguousMultiKeyProducesEnd) {
  const auto qname = "compat_dispatch_test_lib::ambiguous_multi_key";
  auto* op = torch::OperatorRegistry::instance().find_operator(qname);
  ASSERT_NE(op, nullptr);
  // Registered under CUDA and XPU – neither is in the preferred list.
  ASSERT_GE(op->implementations.size(), 2UL);
  EXPECT_EQ(op->implementations.find(torch::DispatchKey::CPU),
            op->implementations.end());
  EXPECT_EQ(op->implementations.find(torch::DispatchKey::BackendSelect),
            op->implementations.end());

  auto chosen = pick_impl(op);
  // Must not resolve to any implementation – ambiguous.
  EXPECT_EQ(chosen, op->implementations.end());
}
