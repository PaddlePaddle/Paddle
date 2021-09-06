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
#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"

#include "paddle/fluid/eager/api/api.h"

#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/dense_tensor.h"

#include "paddle/fluid/eager/tests/test_utils.h"

using namespace egr;

TEST(Backward, SingleNodeEmptyGrad) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());
  
  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  
  // Create Target Tensor
  pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor)); 
  
  pt::Tensor leaf_tensor;
  {
      // Create Scale Node
      auto node0_ptr = std::make_shared<GradNodeScale>();
      node0_ptr->SetAttributes(5.0/*scale*/);

      // Connect Tensor and Node via AutoGradMeta
      AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(target_tensors[0]);
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      
      // Connect Tensor and AccumulationNode via AutoGradMeta
      auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();
      
      AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(leaf_tensor);
      auto_grad_meta1->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
      auto_grad_meta1->SetOutRank(0);
      
      egr::RetainGradForTensor(leaf_tensor);
      
      // Connect Node0 -> AccumulationNode via Edge
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(acc_node_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Run Backward
  RunBackward(target_tensors, {});

  // Check Output Value
  PADDLE_ENFORCE(CompareGradTensorWithValue<float>(leaf_tensor, 5.0) == true, 
    paddle::platform::errors::Fatal("Numerical Error, Expected %f", 5.0));
}


TEST(Backward, SingleNodeCustomGrad) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());
  
  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  
  // Create Target Tensor
  pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor)); 

  std::vector<pt::Tensor> grad_tensors;
  // Create Grad Tensor
  pt::Tensor grad_tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        10.0 /*value*/, false /*is_leaf*/);
  grad_tensors.emplace_back(std::move(grad_tensor)); 
  
  pt::Tensor leaf_tensor;
  {
      // Create Scale Node
      auto node0_ptr = std::make_shared<GradNodeScale>();
      node0_ptr->SetAttributes(5.0/*scale*/);

      // Connect Tensor and Node via AutoGradMeta
      AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(target_tensors[0]);
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      
      // Connect Tensor and AccumulationNode via AutoGradMeta
      auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();
      
      AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(leaf_tensor);
      auto_grad_meta1->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
      auto_grad_meta1->SetOutRank(0);
      
      egr::RetainGradForTensor(leaf_tensor);
      
      // Connect Node0 -> AccumulationNode via Edge
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(acc_node_ptr);
      node0_ptr->AddEdges({ &meta });
  }

  // Run Backward
  RunBackward(target_tensors, grad_tensors);
  
  // Check Output Value
  PADDLE_ENFORCE(CompareGradTensorWithValue<float>(leaf_tensor, 50.0) == true, 
    paddle::platform::errors::Fatal("Numerical Error, Expected %f", 50.0));
}


/*
Node1
  |
Node0
  |
 inp0
*/
TEST(Backward, LinearNodes) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());
  
  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  
  // Create Target Tensor
  pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor)); 
  
  pt::Tensor leaf_tensor;
  {
      // Create Node0
      auto node0_ptr = std::make_shared<GradNodeScale>();
      node0_ptr->SetAttributes(5.0/*scale*/);
      
      // Create Node1
      auto node1_ptr = std::make_shared<GradNodeScale>();
      node1_ptr->SetAttributes(10.0/*scale*/);

      // Connect Input Tensor and Node0 via AutoGradMeta
      AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(target_tensors[0]);
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);

      // Connect Node0 -> Node1 via Edge
      auto meta0 = egr::AutogradMeta();
      meta0.SetOutRank(0);
      meta0.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta0 });
      
      // Connect Tensor and AccumulationNode via AutoGradMeta
      auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();
      
      AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(leaf_tensor);
      auto_grad_meta1->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
      auto_grad_meta1->SetOutRank(0);
      
      egr::RetainGradForTensor(leaf_tensor);
      
      // Connect Node1 -> AccumulationNode via Edge
      auto meta1 = egr::AutogradMeta();
      meta1.SetOutRank(0);
      meta1.SetGradNode(acc_node_ptr);
      node1_ptr->AddEdges({ &meta1 });
  }

  // Use Empty Grad Tensor
  RunBackward(target_tensors, {});
  
  // Check Output Value
  PADDLE_ENFORCE(CompareGradTensorWithValue<float>(leaf_tensor, 50.0) == true, 
    paddle::platform::errors::Fatal("Numerical Error, Expected %f", 50.0));
}


/*
    Node2
    |   |
Node0   Node1
  |      |
 inp0   inp1
*/
TEST(Backward, WithAccumulation) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());
  
  // Prepare Inputs
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  
  // Create Target Tensor
  std::vector<pt::Tensor> target_tensors;
  pt::Tensor tensor0 = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        1.0 /*value*/, false /*is_leaf*/);
  pt::Tensor tensor1 = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor0)); 
  target_tensors.emplace_back(std::move(tensor1)); 
  
  // Create Grad Tensor
  std::vector<pt::Tensor> grad_tensors;
  pt::Tensor grad_tensor0 = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        5.0 /*value*/, false /*is_leaf*/);
  pt::Tensor grad_tensor1 = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                        pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                        10.0 /*value*/, false /*is_leaf*/);
  grad_tensors.emplace_back(std::move(grad_tensor0)); 
  grad_tensors.emplace_back(std::move(grad_tensor1)); 
  
  pt::Tensor leaf_tensor;
  {
      // Create Node0
      auto node0_ptr = std::make_shared<GradNodeScale>();
      node0_ptr->SetAttributes(5.0/*scale*/);
      
      // Create Node1
      auto node1_ptr = std::make_shared<GradNodeScale>();
      node1_ptr->SetAttributes(10.0/*scale*/);
      
      // Create Node2
      auto node2_ptr = std::make_shared<GradNodeScale>();
      node2_ptr->SetAttributes(20.0/*scale*/);

      // Connect Inp0 and Node0 via AutoGradMeta
      AutogradMeta* auto_grad_meta0 = EagerUtils::autograd_meta(target_tensors[0]);
      auto_grad_meta0->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
      auto_grad_meta0->SetOutRank(0);
  
      // Connect Inp1 and Node1 via AutoGradMeta
      AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(target_tensors[1]);
      auto_grad_meta1->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(node1_ptr));
      auto_grad_meta1->SetOutRank(0);

      // Connect Node0 -> Node2 via Edge
      auto meta0 = egr::AutogradMeta();
      meta0.SetOutRank(0);
      meta0.SetGradNode(node2_ptr);
      node0_ptr->AddEdges({ &meta0 });
  
      // Connect Node1 -> Node2 via Edge
      auto meta1 = egr::AutogradMeta();
      meta1.SetOutRank(0);
      meta1.SetGradNode(node2_ptr);
      node1_ptr->AddEdges({ &meta1 });
      
      // Connect Tensor and AccumulationNode via AutoGradMeta
      auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();
      
      AutogradMeta* auto_grad_meta2 = EagerUtils::autograd_meta(leaf_tensor);
      auto_grad_meta2->SetGradNode(std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
      auto_grad_meta2->SetOutRank(0);
      
      egr::RetainGradForTensor(leaf_tensor);
      
      // Connect Node2 -> AccumulationNode via Edge
      auto meta2 = egr::AutogradMeta();
      meta2.SetOutRank(0);
      meta2.SetGradNode(acc_node_ptr);
      node2_ptr->AddEdges({ &meta2 });
  }

  // Use Empty Grad Tensor
  RunBackward(target_tensors, grad_tensors);
  
  // Check Output Value
  PADDLE_ENFORCE(CompareGradTensorWithValue<float>(leaf_tensor, 2500.0) == true, 
    paddle::platform::errors::Fatal("Numerical Error, Expected %f", 2500.0));

}
