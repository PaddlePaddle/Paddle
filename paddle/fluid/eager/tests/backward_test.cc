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

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/nodes/scale_node.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"

#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/dense_tensor.h"

TEST(Backward, SingleNodeEmptyGrad) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
  auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
  auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
  
  auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
  std::vector<std::shared_ptr<pt::Tensor>> target_tensors = { std::move(tensor) }; 
  
  // Create Scale Node
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);

  // Connect Tensor and Node via AutoGradMeta
  auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
  auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
  auto_grad_meta->SetOutRank(0);

  target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});
}
