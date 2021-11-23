// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

// Eager Dygraph

#include "gtest/gtest.h"

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tests/data_structure_tests/grad_node_test.h"
#include "paddle/fluid/eager/utils.h"

TEST(EagerUtils, ComputeRequireGrad) {
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad2 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad3 = std::make_shared<egr::AutogradMeta>();
  CHECK_EQ(auto_grad0->NumericStopGradient(), -1);
  VLOG(6) << "Single Test ComputeRequireGrad";
  auto_grad0->SetStopGradient(true);
  CHECK(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get()) == false);
  CHECK(egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get()) == false);
  auto_grad0->SetStopGradient(false);
  CHECK(egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get()) == false);
  CHECK(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get()) == true);

  VLOG(6) << "Multi Test ComputeRequireGrad";
  auto_grad0->SetStopGradient(false);
  auto_grad1->SetStopGradient(true);
  CHECK(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get(),
                                            auto_grad1.get()) == true);
  CHECK(egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get(),
                                            auto_grad1.get()) == false);
  auto_grad0->SetStopGradient(true);
  CHECK(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get(),
                                            auto_grad1.get()) == false);
  CHECK(egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get(),
                                            auto_grad1.get()) == false);
}

TEST(EagerUtils, PassStopGradient) {
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad2 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad3 = std::make_shared<egr::AutogradMeta>();
  CHECK_EQ(auto_grad0->NumericStopGradient(), -1);
  VLOG(6) << "Test PassStopGradient";
  egr::EagerUtils::PassStopGradient(false, auto_grad0.get());
  CHECK(auto_grad0->StopGradient() == false);
  egr::EagerUtils::PassStopGradient(true, auto_grad0.get(), auto_grad1.get(),
                                    auto_grad2.get(), auto_grad3.get());
  CHECK(auto_grad0->StopGradient() == true);
  CHECK(auto_grad1->StopGradient() == true);
  CHECK(auto_grad2->StopGradient() == true);
  CHECK(auto_grad3->StopGradient() == true);
}
