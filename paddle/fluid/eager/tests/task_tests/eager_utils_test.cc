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
#include "paddle/fluid/eager/utils.h"
#include "paddle/pten/api/lib/utils/allocator.h"

#include "paddle/pten/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(EagerUtils, AutoGradMeta) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt0 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  dt0->mutable_data<float>()[0] = 10.0;
  EagerTensor et0 = EagerTensor(dt0);

  std::shared_ptr<pten::DenseTensor> dt1 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  dt1->mutable_data<float>()[0] = 20.0;
  EagerTensor et1 = EagerTensor(dt1);

  std::vector<EagerTensor> ets = {et0, et1};
  auto test_node = std::make_shared<eager_test::GradTestNode>();

  // unsafe_autograd_meta()
  // autograd_meta()
  // multi_autograd_meta()
  AutogradMeta* autograd_meta0 = EagerUtils::autograd_meta(&et0);
  AutogradMeta* autograd_meta1 = EagerUtils::autograd_meta(&et1);

  AutogradMeta* unsafe_autograd_meta_after =
      EagerUtils::unsafe_autograd_meta(et0);
  CHECK_NOTNULL(unsafe_autograd_meta_after);

  std::vector<AutogradMeta*> autograd_metas =
      EagerUtils::multi_autograd_meta(&ets);
  std::vector<AutogradMeta*> unsafe_autograd_metas =
      EagerUtils::unsafe_autograd_meta(&ets);
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
