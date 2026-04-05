// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Tests for the compat-layer dispatch key priority selection logic introduced
// in OperationInvoker::get_op_with_args (torch_compat.h).
//
// The lookup order is: CPU → BackendSelect → CatchAll → first-registered.
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

TORCH_LIBRARY(compat_dispatch_test_lib, m) {
  m.def("backend_select_only(int x) -> int");
  m.def("backend_select_and_cpu(int x) -> int");
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, BackendSelect, m) {
  m.impl("backend_select_only", &backend_select_probe);
  m.impl("backend_select_and_cpu", &backend_select_and_cpu_bs_fn);
}

TORCH_LIBRARY_IMPL(compat_dispatch_test_lib, CPU, m) {
  m.impl("backend_select_and_cpu", &backend_select_and_cpu_cpu_fn);
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
  if (chosen == op->implementations.end() && !op->implementations.empty()) {
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
