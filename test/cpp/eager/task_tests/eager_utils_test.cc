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
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/eager/data_structure_tests/grad_node_test.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace egr {

TEST(EagerUtils, AutoGradMeta) {
  // Construct Eager Tensor
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 1}));
  std::shared_ptr<phi::DenseTensor> dt0 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  dt0->mutable_data<float>(phi::CPUPlace())[0] = 10.0;
  paddle::Tensor et0 = paddle::Tensor(dt0);

  std::shared_ptr<phi::DenseTensor> dt1 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  dt1->mutable_data<float>(phi::CPUPlace())[0] = 20.0;
  paddle::Tensor et1 = paddle::Tensor(dt1);

  // unsafe_autograd_meta()
  // autograd_meta()
  AutogradMeta* autograd_meta0 = EagerUtils::autograd_meta(&et0);
  AutogradMeta* autograd_meta1 = EagerUtils::autograd_meta(&et1);

  AutogradMeta* unsafe_autograd_meta_after =
      EagerUtils::unsafe_autograd_meta(et0);
  PADDLE_ENFORCE_NOT_NULL(
      unsafe_autograd_meta_after,
      common::errors::PreconditionNotMet(
          "Unsafe autograd meta after should not be null."));

  // NOTE: Since autograd_meta will be copied make sure it's not null
  std::vector<paddle::Tensor> ets = {et0, et1};
  auto test_node = std::make_shared<eager_test::GradTestNode>();

  std::vector<AutogradMeta*> autograd_metas = EagerUtils::autograd_meta(&ets);
  std::vector<AutogradMeta*> unsafe_autograd_metas =
      EagerUtils::unsafe_autograd_meta(ets);
  PADDLE_ENFORCE_NOT_NULL(unsafe_autograd_metas[0],
                          common::errors::PreconditionNotMet(
                              "Unsafe autograd metas should not be null."));
  PADDLE_ENFORCE_NOT_NULL(unsafe_autograd_metas[1],
                          common::errors::PreconditionNotMet(
                              "Unsafe autograd metas should not be null."));

  // Set Autograd Meta
  autograd_meta0->SetSingleOutRankWithSlot(0, 1);

  autograd_meta0->SetGradNode(test_node);

  // OutRankInfo()
  std::pair<size_t, size_t> out_rank_info0 = EagerUtils::OutRankInfo(et0);
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info0.first),
      0UL,
      common::errors::InvalidArgument("The first element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info0.first)));
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info0.second),
      1UL,
      common::errors::InvalidArgument("The second element of out rank info "
                                      "mismatch. Expected 1 but received %d.",
                                      static_cast<int>(out_rank_info0.second)));

  // grad_node()
  std::shared_ptr<GradNodeBase> grad_node0 = EagerUtils::grad_node(et0);
  PADDLE_ENFORCE_NOT_NULL(
      grad_node0.get(),
      common::errors::PreconditionNotMet("Grad of node should not be null."));

  EagerUtils::SetHistory(autograd_meta1, test_node);
  EagerUtils::SetHistory(autograd_meta1, test_node);
  std::shared_ptr<GradNodeBase> grad_node1 = EagerUtils::grad_node(et1);
  PADDLE_ENFORCE_NOT_NULL(
      grad_node1.get(),
      common::errors::PreconditionNotMet("Grad of node should not be null."));

  // SetOutRankWithSlot()
  EagerUtils::SetOutRankWithSlot(autograd_meta1, 0);
  std::pair<size_t, size_t> out_rank_info1 = EagerUtils::OutRankInfo(et1);
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info1.first),
      0UL,
      common::errors::InvalidArgument("The first element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info1.first)));
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info1.second),
      0UL,
      common::errors::InvalidArgument("The second element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info1.second)));

  EagerUtils::SetOutRankWithSlot(&autograd_metas, 0);
  std::pair<size_t, size_t> out_rank_info2 = EagerUtils::OutRankInfo(et0);
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info2.first),
      0UL,
      common::errors::InvalidArgument("The first element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info2.first)));
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info2.second),
      0UL,
      common::errors::InvalidArgument("The second element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info2.second)));

  std::pair<size_t, size_t> out_rank_info3 = EagerUtils::OutRankInfo(et1);
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info3.first),
      0UL,
      common::errors::InvalidArgument("The first element of out rank info "
                                      "mismatch. Expected 0 but received %d.",
                                      static_cast<int>(out_rank_info3.first)));
  PADDLE_ENFORCE_EQ(
      static_cast<int>(out_rank_info3.second),
      1UL,
      common::errors::InvalidArgument("The second element of out rank info "
                                      "mismatch. Expected 1 but received %d.",
                                      static_cast<int>(out_rank_info3.second)));
}

template <typename T>
paddle::Tensor CreateTestCPUTensor(T val, const phi::DDim& ddim) {
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, ddim);
  paddle::Tensor tensor;
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<T>(phi::CPUPlace());
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
  PADDLE_ENFORCE_EQ(
      auto_grad0->NumericStopGradient(),
      -1,
      common::errors::InvalidArgument("The NumericStopGradient of auto grad "
                                      "mismatch. Expected -1 but received %d.",
                                      auto_grad0->NumericStopGradient()));
  VLOG(6) << "Single Test ComputeRequireGrad";
  auto_grad0->SetStopGradient(true);
  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get()),
                    false,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(true, auto_grad0) to be "
                        "false, but it is true."));
  PADDLE_ENFORCE_EQ(
      egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get()),
      false,
      ::common::errors::InvalidArgument(
          "Expected ComputeRequireGrad(false, auto_grad0) to be false, but it "
          "is true."));
  auto_grad0->SetStopGradient(false);
  PADDLE_ENFORCE_EQ(
      egr::EagerUtils::ComputeRequireGrad(false, auto_grad0.get()),
      false,
      ::common::errors::InvalidArgument(
          "Expected ComputeRequireGrad(false, auto_grad0) to be false, but it "
          "is true."));

  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(true, auto_grad0.get()),
                    true,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(true, auto_grad0) to be "
                        "true, but it is false."));

  VLOG(6) << "Multi Test ComputeRequireGrad";
  auto_grad0->SetStopGradient(false);
  auto_grad1->SetStopGradient(true);
  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(
                        true, auto_grad0.get(), auto_grad1.get()),
                    true,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(true, auto_grad0, "
                        "auto_grad1) to be true, but it is false."));

  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(
                        false, auto_grad0.get(), auto_grad1.get()),
                    false,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(false, auto_grad0, "
                        "auto_grad1) to be false, but it is true."));
  auto_grad0->SetStopGradient(true);
  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(
                        true, auto_grad0.get(), auto_grad1.get()),
                    false,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(true, auto_grad0, "
                        "auto_grad1) to be false, but it is true."));

  PADDLE_ENFORCE_EQ(egr::EagerUtils::ComputeRequireGrad(
                        false, auto_grad0.get(), auto_grad1.get()),
                    false,
                    ::common::errors::InvalidArgument(
                        "Expected ComputeRequireGrad(false, auto_grad0, "
                        "auto_grad1) to be false, but it is true."));
}

TEST(EagerUtils, PassStopGradient) {
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad2 = std::make_shared<egr::AutogradMeta>();
  auto auto_grad3 = std::make_shared<egr::AutogradMeta>();
  PADDLE_ENFORCE_EQ(
      auto_grad0->NumericStopGradient(),
      -1,
      common::errors::InvalidArgument("The NumericStopGradient of auto grad "
                                      "mismatch. Expected -1 but received %d.",
                                      auto_grad0->NumericStopGradient()));
  VLOG(6) << "Test PassStopGradient";
  egr::EagerUtils::PassStopGradient(false, auto_grad0.get());
  PADDLE_ENFORCE_EQ(
      auto_grad0->StopGradient(),
      false,
      ::common::errors::InvalidArgument(
          "Expected auto_grad0->StopGradient() to be false, but received %d.",
          auto_grad0->StopGradient()));
  egr::EagerUtils::PassStopGradient(true,
                                    auto_grad0.get(),
                                    auto_grad1.get(),
                                    auto_grad2.get(),
                                    auto_grad3.get());
  PADDLE_ENFORCE_EQ(
      auto_grad0->StopGradient(),
      true,
      ::common::errors::InvalidArgument(
          "Expected auto_grad0->StopGradient() to be true, but received %d.",
          auto_grad0->StopGradient()));
  PADDLE_ENFORCE_EQ(
      auto_grad1->StopGradient(),
      true,
      ::common::errors::InvalidArgument(
          "Expected auto_grad1->StopGradient() to be true, but received %d.",
          auto_grad1->StopGradient()));
  PADDLE_ENFORCE_EQ(
      auto_grad2->StopGradient(),
      true,
      ::common::errors::InvalidArgument(
          "Expected auto_grad2->StopGradient() to be true, but received %d.",
          auto_grad2->StopGradient()));
  PADDLE_ENFORCE_EQ(
      auto_grad3->StopGradient(),
      true,
      ::common::errors::InvalidArgument(
          "Expected auto_grad3->StopGradient() to be true, but received %d.",
          auto_grad3->StopGradient()));
}

TEST(EagerUtils, TrySyncToVar) {
  phi::DDim ddim = common::make_ddim({2, 4, 4, 4});
  auto tensor = CreateTestCPUTensor(5.0f, ddim);
  std::vector<std::shared_ptr<egr::EagerVariable>> var_bases = {
      egr::EagerUtils::TrySyncToVar(tensor)};

  paddle::framework::Variable* var = var_bases[0]->MutableVar();
  const auto& framework_tensor = var->Get<phi::DenseTensor>();

  const float* ptr = framework_tensor.data<float>();
  VLOG(6) << "Check Value for SyncToVarsSingle";
  PADDLE_ENFORCE_EQ(framework_tensor.numel(),
                    tensor.numel(),
                    common::errors::InvalidArgument(
                        "The numel of framework tensor and numel of "
                        "tensor should be the same, but received %d and %d.",
                        framework_tensor.numel(),
                        tensor.numel()));

  for (int i = 0; i < framework_tensor.numel(); i++) {
    PADDLE_ENFORCE_EQ(ptr[i],
                      5.0f,
                      common::errors::InvalidArgument(
                          "The numel of framework tensor mismatch. "
                          "Expected 5.0 but received %f.",
                          ptr[i]));
  }
}

TEST(EagerUtils, TrySyncToVars) {
  phi::DDim ddim = common::make_ddim({2, 4, 4, 4});
  std::vector<paddle::Tensor> tensors = {CreateTestCPUTensor(1.0f, ddim),
                                         CreateTestCPUTensor(2.0f, ddim)};

  std::vector<std::shared_ptr<egr::EagerVariable>> var_bases =
      egr::EagerUtils::TrySyncToVars(tensors);

  {
    paddle::framework::Variable* var = var_bases[0]->MutableVar();
    const auto& framework_tensor = var->Get<phi::DenseTensor>();

    const float* ptr = framework_tensor.data<float>();
    PADDLE_ENFORCE_EQ(
        framework_tensor.numel(),
        tensors[0].numel(),
        common::errors::InvalidArgument(
            "The numel of framework tensor and numel "
            "of tensor should be the same, but received %d and %d.",
            framework_tensor.numel(),
            tensors[0].numel()));

    for (int i = 0; i < framework_tensor.numel(); i++) {
      PADDLE_ENFORCE_EQ(ptr[i],
                        1.0,
                        common::errors::InvalidArgument(
                            "The numel of framework tensor mismatch. Expected "
                            "1.0 but received %f.",
                            ptr[i]));
    }
  }

  {
    paddle::framework::Variable* var = var_bases[1]->MutableVar();
    const auto& framework_tensor = var->Get<phi::DenseTensor>();

    const float* ptr = framework_tensor.data<float>();
    VLOG(6) << "Check Value for SyncToVarsMultiple";
    PADDLE_ENFORCE_EQ(
        framework_tensor.numel(),
        tensors[0].numel(),
        common::errors::InvalidArgument(
            "The numel of framework tensor and numel "
            "of tensor should be the same, but received %d and %d.",
            framework_tensor.numel(),
            tensors[0].numel()));

    for (int i = 0; i < framework_tensor.numel(); i++) {
      PADDLE_ENFORCE_EQ(ptr[i],
                        2.0,
                        common::errors::InvalidArgument(
                            "The numel of framework tensor mismatch. Expected "
                            "2.0 but received %f.",
                            ptr[i]));
    }
  }
}

TEST(EagerUtils, CreateVars) {
  VLOG(6) << "Check CreateVars";
  std::vector<std::shared_ptr<egr::EagerVariable>> outs =
      egr::EagerUtils::CreateVars(2);
  PADDLE_ENFORCE_EQ(
      outs.size(),
      2UL,
      common::errors::InvalidArgument(
          "Size of outs mismatch. Expected 2 but received %d.", outs.size()));
  PADDLE_ENFORCE_EQ(
      outs[0]->Var().IsInitialized(),
      false,
      ::common::errors::AlreadyExists("Expected the first variable to be "
                                      "uninitialized, but already exists."));
}

TEST(EagerUtils, GetGradAccumulationNode) {
  VLOG(6) << "Check GetGradAccumulationNode";
  paddle::Tensor t0("test_tensor");
  ASSERT_EQ(egr::EagerUtils::GetGradAccumulationNode(t0), nullptr);
  auto autograd_ptr0 = egr::EagerUtils::autograd_meta(&t0);
  autograd_ptr0->SetStopGradient(true);
  ASSERT_EQ(egr::EagerUtils::GetGradAccumulationNode(t0), nullptr);
  autograd_ptr0->SetStopGradient(false);
  auto res = std::dynamic_pointer_cast<egr::GradNodeAccumulation>(
      egr::EagerUtils::GetGradAccumulationNode(t0));
  ASSERT_TRUE(res != nullptr);
  auto res2 = egr::EagerUtils::GetGradAccumulationNode(t0);
  ASSERT_EQ(res2.get(), res.get());
  autograd_ptr0->SetStopGradient(true);
  auto res3 = egr::EagerUtils::GetGradAccumulationNode(t0);
  ASSERT_EQ(res3, nullptr);
  autograd_ptr0->SetStopGradient(false);
  autograd_ptr0->SetGradNode(
      std::make_shared<eager_test::GradTestNode>(1, 2.0, 3));
  ASSERT_ANY_THROW(egr::EagerUtils::GetGradAccumulationNode(t0));
}

TEST(EagerUtils, FillZeroForEmptyOptionalGradInput) {
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      grads = {std::vector<paddle::Tensor>(1)};
  paddle::small_vector<std::vector<GradSlotMeta>, egr::kSlotSmallVectorSize>
      slot_metas = {std::vector<GradSlotMeta>(1)};

  phi::DenseTensorMeta tensor_meta;
  tensor_meta.dtype = phi::DataType::FLOAT32;
  tensor_meta.dims = {2, 4};
  slot_metas[0][0].SetTensorMeta(tensor_meta);
  slot_metas[0][0].SetPlace(phi::CPUPlace());

  EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[0], slot_metas[0]);
  eager_test::CompareTensorWithValue<float>(grads[0][0], 0.0);
}

}  // namespace egr
