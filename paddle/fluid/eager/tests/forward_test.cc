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

TEST(Forward, SingleNode) {
  // Create Input Tensor
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
  auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
  {
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
        ptr[i] = 5.0;
      }
  }
  auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
  auto tensor = pt::Tensor(tensor_impl);
  egr::EagerUtils::autograd_meta(tensor)->SetNumericStopGradient(false);

  // Run Forward
  float scale = 2.0;
  float bias = 3.0;
  std::vector<pt::Tensor> outs = egr::scale(tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out = outs[0];
  
  // Examine Forward Output
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 13.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }

  // Examine GradNode
  {
      // 1. GradNode
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(out);
      egr::GradNodeBase* grad_node = meta->GradNode();
      egr::GradNodeScale* scale_node = dynamic_cast<egr::GradNodeScale*>(grad_node);
      PADDLE_ENFORCE(scale_node != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));    
      PADDLE_ENFORCE(meta->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // 2. TensorWrapper: No TensorWrapper for ScaleNode
      // 3. NextEdges: No NextEdges for Single Node Case
  }
}

/*
 inp
  |
Node0
  |
Node1
  |
 out
*/
TEST(Forward, LinearNodes) {
  // Create Input Tensor
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
  auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
  {
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
        ptr[i] = 5.0;
      }
  }
  auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
  auto tensor = pt::Tensor(tensor_impl);
  egr::EagerUtils::autograd_meta(tensor)->SetNumericStopGradient(false);
  
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  std::vector<pt::Tensor> outs0 = egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out0 = outs0[0];
  
  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  std::vector<pt::Tensor> outs1 = egr::scale(out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out1 = outs1[0];
  
  // Examine Forward Output 0
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out0.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 13.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }
  
  // Examine Forward Output 1
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out1.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 75.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }

  // Examine GradNode
  {
      // 1. GradNode
      // Node 0
      egr::AutogradMeta* meta0 = egr::EagerUtils::autograd_meta(out0);
      egr::GradNodeBase* grad_node0 = meta0->GradNode();
      egr::GradNodeScale* scale_node0 = dynamic_cast<egr::GradNodeScale*>(grad_node0);
      PADDLE_ENFORCE(scale_node0 != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));
      PADDLE_ENFORCE(meta0->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // Node 1
      egr::AutogradMeta* meta1 = egr::EagerUtils::autograd_meta(out1);
      egr::GradNodeBase* grad_node1 = meta1->GradNode();
      egr::GradNodeScale* scale_node1 = dynamic_cast<egr::GradNodeScale*>(grad_node1);
      PADDLE_ENFORCE(scale_node1 != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));
      PADDLE_ENFORCE(meta1->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // 2. TensorWrapper: No TensorWrapper for ScaleNode
      // 3. NextEdges: Node 1 -> Node 0
      const std::vector<egr::Edge>& node1_edges = grad_node1->GetEdges();
      PADDLE_ENFORCE(node1_edges.size() == 1, 
        paddle::platform::errors::Fatal("Node 1 should have exactly 1 edge"));

      const egr::Edge& node1_edge = node1_edges[0];
      PADDLE_ENFORCE(node1_edge.GetInputRank() == 0, 
        paddle::platform::errors::Fatal("Node1's edge should have input rank of 0"));
      
      PADDLE_ENFORCE(node1_edge.GetGradNode() == grad_node0, 
        paddle::platform::errors::Fatal("Node1's edge should point to Node 0"));

  }

}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(Forward, BranchedNodes) {
  // Create Input Tensor
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
  auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
  {
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
        ptr[i] = 5.0;
      }
  }
  auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
  auto tensor = pt::Tensor(tensor_impl);
  egr::EagerUtils::autograd_meta(tensor)->SetNumericStopGradient(false);
  
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  std::vector<pt::Tensor> outs0 = egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out0 = outs0[0];
  
  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  std::vector<pt::Tensor> outs1 = egr::scale(out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out1 = outs1[0];
  
  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  std::vector<pt::Tensor> outs2 = egr::scale(out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out2 = outs2[0];
  
  // Examine Forward Output 0
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out0.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 13.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }
  
  // Examine Forward Output 1
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out1.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 75.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }
  
  // Examine Forward Output 2
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out2.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 150.0, 
            paddle::platform::errors::Fatal("Numerical Error"));
      }
  }

  // Examine GradNode
  {
      // 1. GradNode
      // Node 0
      egr::AutogradMeta* meta0 = egr::EagerUtils::autograd_meta(out0);
      egr::GradNodeBase* grad_node0 = meta0->GradNode();
      egr::GradNodeScale* scale_node0 = dynamic_cast<egr::GradNodeScale*>(grad_node0);
      PADDLE_ENFORCE(scale_node0 != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));
      PADDLE_ENFORCE(meta0->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // Node 1
      egr::AutogradMeta* meta1 = egr::EagerUtils::autograd_meta(out1);
      egr::GradNodeBase* grad_node1 = meta1->GradNode();
      egr::GradNodeScale* scale_node1 = dynamic_cast<egr::GradNodeScale*>(grad_node1);
      PADDLE_ENFORCE(scale_node1 != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));
      PADDLE_ENFORCE(meta1->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // Node 2
      egr::AutogradMeta* meta2 = egr::EagerUtils::autograd_meta(out2);
      egr::GradNodeBase* grad_node2 = meta2->GradNode();
      egr::GradNodeScale* scale_node2 = dynamic_cast<egr::GradNodeScale*>(grad_node2);
      PADDLE_ENFORCE(scale_node2 != nullptr, 
        paddle::platform::errors::Fatal("AutogradMeta should have held grad_node with type GradNodeScale*"));
      PADDLE_ENFORCE(meta2->OutRank() == 0, 
        paddle::platform::errors::Fatal("OutRank in AutogradMeta should have been 0"));
      
      // 2. TensorWrapper: No TensorWrapper for ScaleNode
      // 3. NextEdges 
      // Node 1 -> Node 0
      const std::vector<egr::Edge>& node1_edges = grad_node1->GetEdges();
      PADDLE_ENFORCE(node1_edges.size() == 1, 
        paddle::platform::errors::Fatal("Node 1 should have exactly 1 edge"));
      const egr::Edge& node1_edge = node1_edges[0];
      PADDLE_ENFORCE(node1_edge.GetInputRank() == 0, 
        paddle::platform::errors::Fatal("Node1's edge should have input rank of 0"));
      PADDLE_ENFORCE(node1_edge.GetGradNode() == grad_node0, 
        paddle::platform::errors::Fatal("Node1's edge should point to Node 0"));
      
      // Node 2 -> Node 0
      const std::vector<egr::Edge>& node2_edges = grad_node2->GetEdges();
      PADDLE_ENFORCE(node2_edges.size() == 1, 
        paddle::platform::errors::Fatal("Node 2 should have exactly 1 edge"));
      const egr::Edge& node2_edge = node2_edges[0];
      PADDLE_ENFORCE(node2_edge.GetInputRank() == 0, 
        paddle::platform::errors::Fatal("Node2's edge should have input rank of 0"));
      PADDLE_ENFORCE(node2_edge.GetGradNode() == grad_node0, 
        paddle::platform::errors::Fatal("Node2's edge should point to Node 0"));

  }

}
