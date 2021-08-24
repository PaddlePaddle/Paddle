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

#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/nodes/scale_node.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"

#include "paddle/fluid/eager/api/api.h"

#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/dense_tensor.h"

pt::Tensor hook_function(const pt::Tensor& t) { 
    auto t_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t.impl());
    
    auto ret_meta = std::make_unique<pt::TensorMeta>(t_dense->dims(), t_dense->backend(), t_dense->type(), t_dense->layout());
    auto ret_dense = std::make_shared<pt::DenseTensor>(std::move(ret_meta));
    
    float* t_ptr = t_dense->mutable_data<float>();
    float* ret_ptr = ret_dense->mutable_data<float>();
    for(int i = 0; i < ret_dense->numel(); i++) {
        ret_ptr[i] = t_ptr[i] + 3.0;
    }
    
    auto ret_impl = std::dynamic_pointer_cast<pt::TensorInterface>(ret_dense);
    pt::Tensor ret = pt::Tensor();
    ret.SetImpl(ret_impl);
    
    return ret;
};

/*
AccumulationNode
  |
ScaleNode
  |
 inp0
*/
TEST(CrossBatchAccumulation, SingleScaleNode) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<std::shared_ptr<pt::Tensor>> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  pt::Tensor& target_tensor = *target_tensors[0].get();
  
  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<egr::GradNodeScale>();
  scale_node_ptr->SetAttributes(5.0/*scale*/);
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(scale_node_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      egr::RetainGradForTensor(target_tensor); // result: 1.0
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(acc_node_ptr);
      scale_node_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for leaf tensor1
  pt::Tensor leaf_tensor = pt::Tensor();
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
      auto_grad_meta->SetOutRank(0);
      leaf_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RetainGradForTensor(leaf_tensor);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print target tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(target_tensor);
      auto target_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = target_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 1.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 1.0, ptr[i]));
      }
  }

  // Print leaf tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(leaf_tensor);
      auto leaf_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = leaf_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 5.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 5.0, ptr[i]));
      }
  }
  
  // Cross-Batch Accumulation
  egr::RunBackward(target_tensors, {});
  
  // target tensor grad should remain the same
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(target_tensor);
      auto target_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = target_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 1.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 1.0, ptr[i]));
      }
  }

  // Leaf tensor should keep accumulated grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(leaf_tensor);
      auto leaf_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = leaf_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 10.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 10.0, ptr[i]));
      }
  }
}

