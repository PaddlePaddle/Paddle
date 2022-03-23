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
#pragma once
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/api/lib/utils/allocator.h"
namespace egr {
class TensorWrapper;
}

namespace eager_test {
class GradTestNode : public egr::GradNodeBase {
 public:
  ~GradTestNode() override = default;
  GradTestNode(float val, int in_num, int out_num)
      : GradNodeBase(in_num, out_num), val_(val) {}
  GradTestNode() : GradNodeBase() { val_ = 1.0; }
  std::string name() override { return "GradTestNode"; }
  std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      std::vector<std::vector<paddle::experimental::Tensor>>& grads,
      bool create_graph = false) override {
    val_ = std::dynamic_pointer_cast<phi::DenseTensor>(grads[0][0].impl())
               ->data<float>()[0];
    phi::DenseTensorMeta meta =
        phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 1}));
    std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        meta);
    auto* dt_ptr = dt->mutable_data<float>(paddle::platform::CPUPlace());
    dt_ptr[0] = 6.0f;
    paddle::experimental::Tensor et1(dt);
    std::vector<std::vector<paddle::experimental::Tensor>> res = {{et1}};
    return res;
  }
  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }
  bool IsTensorWrappersCleared() override {
    VLOG(6) << "Do nothing here now";
    return false;
  }
  float val_;
};
}  // namespace eager_test
