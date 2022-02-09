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

#include <sstream>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tests/data_structure_tests/grad_node_test.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"

namespace egr {

TEST(EagerUtils, AutoGradMeta) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt0 = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  dt0->mutable_data<float>(paddle::platform::CPUPlace())[0] = 10.0;
  paddle::experimental::Tensor et0 = paddle::experimental::Tensor(dt0);

  std::shared_ptr<pten::DenseTensor> dt1 = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  dt1->mutable_data<float>(paddle::platform::CPUPlace())[0] = 20.0;
  paddle::experimental::Tensor et1 = paddle::experimental::Tensor(dt1);

  // unsafe_autograd_meta()
  // autograd_meta()
  AutogradMeta* autograd_meta0 = EagerUtils::autograd_meta(&et0);
  AutogradMeta* autograd_meta1 = EagerUtils::autograd_meta(&et1);

  AutogradMeta* unsafe_autograd_meta_after =
      EagerUtils::unsafe_autograd_meta(et0);
  CHECK_NOTNULL(unsafe_autograd_meta_after);

  // NOTE: Since autograd_meta will be copied make sure it's not null
  std::vector<paddle::experimental::Tensor> ets = {et0, et1};
  auto test_node = std::make_shared<eager_test::GradTestNode>();

  std::vector<AutogradMeta*> autograd_metas = EagerUtils::autograd_meta(&ets);
  std::vector<AutogradMeta*> unsafe_autograd_metas =
      EagerUtils::unsafe_autograd_meta(ets);
  CHECK_NOTNULL(unsafe_autograd_metas[0]);
  CHECK_NOTNULL(unsafe_autograd_metas[1]);

  // Set Autograd Meta
  autograd_meta0->SetSingleOutRankWithSlot(0, 1);

  autograd_meta0->SetGradNode(test_node);

  // OutRankInfo()
  std::pair<size_t, size_t> out_rank_info0 = EagerUtils::OutRankInfo(et0);
  CHECK_EQ(static_cast<int>(out_rank_info0.first), 0);
  CHECK_EQ(static_cast<int>(out_rank_info0.second), 1);

  // grad_node()
  std::shared_ptr<GradNodeBase> grad_node0 = EagerUtils::grad_node(et0);
  CHECK_NOTNULL(grad_node0.get());

  EagerUtils::SetHistory(autograd_meta1, test_node);
  EagerUtils::SetHistory({autograd_meta1}, test_node);
  std::shared_ptr<GradNodeBase> grad_node1 = EagerUtils::grad_node(et1);
  CHECK_NOTNULL(grad_node1.get());

  // SetOutRankWithSlot()
  EagerUtils::SetOutRankWithSlot(autograd_meta1, 0);
  std::pair<size_t, size_t> out_rank_info1 = EagerUtils::OutRankInfo(et1);
  CHECK_EQ(static_cast<int>(out_rank_info1.first), 0);
  CHECK_EQ(static_cast<int>(out_rank_info1.second), 0);

  EagerUtils::SetOutRankWithSlot(&autograd_metas, 0);
  std::pair<size_t, size_t> out_rank_info2 = EagerUtils::OutRankInfo(et0);
  CHECK_EQ(static_cast<int>(out_rank_info2.first), 0);
  CHECK_EQ(static_cast<int>(out_rank_info2.second), 0);

  std::pair<size_t, size_t> out_rank_info3 = EagerUtils::OutRankInfo(et1);
  CHECK_EQ(static_cast<int>(out_rank_info3.first), 0);
  CHECK_EQ(static_cast<int>(out_rank_info3.second), 1);
}

template <typename T>
paddle::experimental::Tensor CreateTestCPUTensor(
    T val, const paddle::framework::DDim& ddim) {
  pten::DenseTensorMeta meta =
      pten::DenseTensorMeta(pten::DataType::FLOAT32, ddim);
  paddle::experimental::Tensor tensor;
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<T>(paddle::platform::CPUPlace());
  for (int64_t i = 0; i < dt->numel(); i++) {
    dt_ptr[i] = val;
  }
  tensor.set_impl(dt);
  return tensor;
}

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

TEST(EagerUtils, TrySyncToVar) {
  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
  auto tensor = CreateTestCPUTensor(5.0f, ddim);
  std::vector<std::shared_ptr<egr::EagerTensor>> var_bases = {
      egr::EagerUtils::TrySyncToVar(tensor)};

  paddle::framework::Variable* var = var_bases[0]->MutableVar();
  const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

  const float* ptr = framework_tensor.data<float>();
  VLOG(6) << "Check Value for SyncToVarsSingle";
  CHECK_EQ(framework_tensor.numel(), tensor.numel());

  for (int i = 0; i < framework_tensor.numel(); i++) {
    CHECK_EQ(ptr[i], 5.0f);
  }
}

TEST(EagerUtils, TrySyncToVars) {
  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
  std::vector<paddle::experimental::Tensor> tensors = {
      CreateTestCPUTensor(1.0f, ddim), CreateTestCPUTensor(2.0f, ddim)};

  std::vector<std::shared_ptr<egr::EagerTensor>> var_bases =
      egr::EagerUtils::TrySyncToVars(tensors);

  {
    paddle::framework::Variable* var = var_bases[0]->MutableVar();
    const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

    const float* ptr = framework_tensor.data<float>();
    CHECK_EQ(framework_tensor.numel(), tensors[0].numel());

    for (int i = 0; i < framework_tensor.numel(); i++) {
      CHECK_EQ(ptr[i], 1.0);
    }
  }

  {
    paddle::framework::Variable* var = var_bases[1]->MutableVar();
    const auto& framework_tensor = var->Get<paddle::framework::LoDTensor>();

    const float* ptr = framework_tensor.data<float>();
    VLOG(6) << "Check Value for SyncToVarsMultiple";
    CHECK_EQ(framework_tensor.numel(), tensors[0].numel());

    for (int i = 0; i < framework_tensor.numel(); i++) {
      CHECK_EQ(ptr[i], 2.0);
    }
  }
}

TEST(EagerUtils, CreateVars) {
  VLOG(6) << "Check CreateVars";
  std::vector<std::shared_ptr<egr::EagerTensor>> outs =
      egr::EagerUtils::CreateVars(2);
  CHECK_EQ(outs.size(), size_t(2));
  CHECK(outs[0]->Var().IsInitialized() == false);
}

}  // namespace egr
