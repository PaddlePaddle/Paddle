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
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Scale Node
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);

  // Connect Tensor and Node via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});
}

TEST(Backward, SingleNodeCustomGrad) {
  // Create Target Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }

  // Create Grad Tensor
  std::vector<pt::Tensor> grad_tensors;
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 10.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Scale Node
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);

  // Connect Tensor and Node via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, grad_tensors);
}

/*
Node1
  |
Node0
  |
 inp0
*/
TEST(Backward, LinearNodes) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Node0
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);
  
  // Create Node1
  auto node1_ptr = std::make_shared<egr::GradNodeScale>();
  node1_ptr->SetAttributes(10.0/*scale*/);

  // Connect Input Tensor and Node0 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});
}

/*
Node1   Node3
|       |
Node0   Node2
|       |
inp0    inp1
*/
TEST(Backward, BranchedNodes) {
  // Create Target Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Grad Tensor
  std::vector<pt::Tensor> grad_tensors;
  
  // inp0
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 5.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 10.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Node0
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);
  
  // Create Node1
  auto node1_ptr = std::make_shared<egr::GradNodeScale>();
  node1_ptr->SetAttributes(10.0/*scale*/);
  
  // Create Node2
  auto node2_ptr = std::make_shared<egr::GradNodeScale>();
  node2_ptr->SetAttributes(20.0/*scale*/);
  
  // Create Node3
  auto node3_ptr = std::make_shared<egr::GradNodeScale>();
  node3_ptr->SetAttributes(40.0/*scale*/);

  // Connect inp0 and Node0 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }
  
  // Connect inp1 and Node2 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node2_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[1].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Connect Node2 -> Node3 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node3_ptr);
      node2_ptr->AddEdges({ &meta });
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, grad_tensors);
}

/*
    Node2
    |   |
Node0   Node1
  |      |
 inp0   inp1
*/
TEST(Backward, WithAccumulation) {
  // Create Target Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Grad Tensor
  std::vector<pt::Tensor> grad_tensors;
  
  // inp0
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 5.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 10.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Node0
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);
  
  // Create Node1
  auto node1_ptr = std::make_shared<egr::GradNodeScale>();
  node1_ptr->SetAttributes(10.0/*scale*/);
  
  // Create Node2
  auto node2_ptr = std::make_shared<egr::GradNodeScale>();
  node2_ptr->SetAttributes(20.0/*scale*/);

  // Connect Inp0 and Node0 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }
  
  // Connect Inp1 and Node1 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[1].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node2 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node2_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Connect Node1 -> Node2 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node2_ptr);
      node1_ptr->AddEdges({ &meta });
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, grad_tensors);
}
