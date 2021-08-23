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

#include "paddle/fluid/eager/api/api.h"

#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/dense_tensor.h"

/*
Node1
  | retain_grad
Node0
  |
 inp0
*/
TEST(RetainGrad, NonLeafTensor) {
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
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for tensor1 
  pt::Tensor fake_tensor = pt::Tensor();
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RetainGradForTensor(fake_tensor);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print retained grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(fake_tensor);
      auto fake_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = fake_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          VLOG(2) << ptr[i];
      }
      // result: 5.0
  }
}

/*
Node1
  | retain_grad
  | gradient_hook
Node0
  |
 inp0
*/
TEST(RetainGrad, HookBeforeRetainGrad) {
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
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for tensor1 
  pt::Tensor fake_tensor = pt::Tensor();
  {
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };

      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RegisterGradientHookForTensor(fake_tensor, hook1);
      egr::RetainGradForTensor(fake_tensor);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print retained grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(fake_tensor);
      auto fake_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = fake_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          VLOG(2) << ptr[i];
      }
      // result: 8.0
  }
}

/*
Node1
  | gradient_hook
  | retain_grad
Node0
  |
 inp0
*/
TEST(RetainGrad, HookAfterRetainGrad) {
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
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for tensor1 
  pt::Tensor fake_tensor = pt::Tensor();
  {
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };

      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RetainGradForTensor(fake_tensor);
      egr::RegisterGradientHookForTensor(fake_tensor, hook1);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print retained grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(fake_tensor);
      auto fake_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = fake_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          VLOG(2) << ptr[i];
      }
      // result: 8.0
  }
}

TEST(GradientHook, SingleNode) {
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
  
  // Create Scale Node
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);

  // Connect Tensor and Node via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Register GradientHook
  std::function<pt::Tensor(const pt::Tensor&)> hook = [](const pt::Tensor& t) { 
    pt::Tensor ret = pt::Tensor();

    // Copy t::impl()
    ret = t;
    auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
    float* ret_ptr = ret_dense->mutable_data<float>();

    for(int i = 0; i < ret_dense->numel(); i++) {
        ret_ptr[i] += 2.0;
    }

    return ret;
  };
  egr::RegisterGradientHookForTensor(*target_tensors[0].get(), hook);

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // result should be: (1.0 + 2.0) * 5.0 = 15.0
}

/*
Node1
  |
Node0
  |
 inp0
*/
TEST(GradientHook, LinearNodes) {
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
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Register Hooks
  {
      // Node0 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook0 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 1.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(*target_tensors[0].get(), hook0);
      
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };
      // Fake an AutogradMeta
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      pt::Tensor fake_tensor = pt::Tensor();
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      egr::RegisterGradientHookForTensor(fake_tensor, hook1);
  }
  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // result: ((1.0+1.0)*5.0 + 3.0)*10.0 = 130.0
}


/*
    Node2
    |   |
Node0   Node1
  |      |
 inp0   inp1
*/
TEST(GradientHook, WithAccumulation) {
  // Create Target Tensor
  std::vector<std::shared_ptr<pt::Tensor>> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  // inp0
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Grad Tensor
  std::vector<std::shared_ptr<pt::Tensor>> grad_tensors;
  
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
      auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
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
      auto tensor = std::make_shared<pt::Tensor>(tensor_impl);
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
      target_tensors[0]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }
  
  // Connect Inp1 and Node1 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[1]->SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
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

  // Register Hooks
  {
      // Node0 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook0 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 1.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(*target_tensors[0].get(), hook0);
      
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 2.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(*target_tensors[1].get(), hook1);
      
      // Node2 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook2 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };

      // Fake an AutogradMeta
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node2_ptr));
      auto_grad_meta->SetOutRank(0);
      pt::Tensor fake_tensor = pt::Tensor();
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      egr::RegisterGradientHookForTensor(fake_tensor, hook2);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, grad_tensors);

  // result: ((5+1)*5 + (10+2)*10 + 3) * 20 = 3060
}
