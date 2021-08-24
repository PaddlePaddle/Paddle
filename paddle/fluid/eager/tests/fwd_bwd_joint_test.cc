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
        ret_ptr[i] = t_ptr[i] + 5.0;
    }
    
    auto ret_impl = std::dynamic_pointer_cast<pt::TensorInterface>(ret_dense);
    pt::Tensor ret = pt::Tensor();
    ret.SetImpl(ret_impl);
    
    return ret;
};

TEST(FwdBwdJoint, SingleNode) {
  // 1. Prepare Input
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
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();    
  
  egr::AutogradMeta* autograd_meta = egr::EagerUtils::autograd_meta(tensor);
  autograd_meta->SetNumericStopGradient(false);
  autograd_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
  autograd_meta->SetOutRank(0);
  egr::RetainGradForTensor(tensor);

  // 3. Run Forward
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
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 13.0, ptr[i]));
      }
  }

  // 4. Run Backward
  egr::RunBackward(outs, {});
  
  // Examine Backward Grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 2.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 2.0, ptr[i]));
      }
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
TEST(FwdBwdJoint, LinearNodes) {
  // 1. Prepare Input
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
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();    
  
  egr::AutogradMeta* autograd_meta = egr::EagerUtils::autograd_meta(tensor);
  autograd_meta->SetNumericStopGradient(false);
  autograd_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
  autograd_meta->SetOutRank(0);
  egr::RetainGradForTensor(tensor);

  // 3. Run Forward
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
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 13.0, ptr[i]));
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

  // 4. Run Backward
  egr::RunBackward(outs1, {});
  
  // Examine Backward Grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 10.0,
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 10.0, ptr[i]));
      }
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
TEST(FwdBwdJoint, BranchedNodes) {
  // 1. Prepare Input
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
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();    
  
  egr::AutogradMeta* autograd_meta = egr::EagerUtils::autograd_meta(tensor);
  autograd_meta->SetNumericStopGradient(false);
  autograd_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
  autograd_meta->SetOutRank(0);
  egr::RetainGradForTensor(tensor);

  // 3. Run Forward
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
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 13.0, ptr[i]));
      }
  }
  
  // Examine Forward Output 1
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out1.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 75.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 75.0, ptr[i]));
      }
  }
  
  // Examine Forward Output 2
  {
      auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out2.impl());
      float* ptr = dense_out->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 150.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 150.0, ptr[i]));
      }
  }

  // 4. Run Backward
  std::vector<pt::Tensor> outs = {out1, out2};
  egr::RunBackward(outs, {});
  
  // Examine Backward Grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 30.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 30.0, ptr[i]));
      }
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
TEST(FwdBwdJoint, GradientHook) {
  // 1. Prepare Input
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
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();    
  
  egr::AutogradMeta* autograd_meta = egr::EagerUtils::autograd_meta(tensor);
  autograd_meta->SetNumericStopGradient(false);
  autograd_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
  autograd_meta->SetOutRank(0);
  egr::RetainGradForTensor(tensor);
  
  std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  std::vector<pt::Tensor> outs0 = egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out0 = outs0[0];
  egr::RetainGradForTensor(out0); // hook: +5
  egr::RegisterGradientHookForTensor(out0, hook); // hook: +5
  
  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  std::vector<pt::Tensor> outs1 = egr::scale(out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out1 = outs1[0];
  egr::RetainGradForTensor(out1); // hook: +5
  egr::RegisterGradientHookForTensor(out1, hook); // hook: +5
  
  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  std::vector<pt::Tensor> outs2 = egr::scale(out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);
  pt::Tensor& out2 = outs2[0];
  egr::RetainGradForTensor(out2); // hook: +5
  egr::RegisterGradientHookForTensor(out2, hook); // hook: +5
  
  // 4. Run Backward
  std::vector<pt::Tensor> outs = {out1, out2};
  egr::RunBackward(outs, {});
  
  // Examine Backward Grad
  // leaf grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 190.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 190.0, ptr[i]));
      }
  }

  // out0 grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(out0);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 90.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 90.0, ptr[i]));
      }
  }
  
  // out1 grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(out1);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 1.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 1.0, ptr[i]));
      }
  }

  // out2 grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(out2);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 1.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 1.0, ptr[i]));
      }
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
TEST(FwdBwdJoint, CrossBatchAccumulation) {
  // 1. Prepare Input
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
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();    
  
  egr::AutogradMeta* autograd_meta = egr::EagerUtils::autograd_meta(tensor);
  autograd_meta->SetNumericStopGradient(false);
  autograd_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
  autograd_meta->SetOutRank(0);
  egr::RetainGradForTensor(tensor);

  // 3. Run Forward
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

  // 4. Run Backward
  std::vector<pt::Tensor> outs = {out1, out2};
  egr::RunBackward(outs, {});
  
  // Examine Backward Grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 30.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 30.0, ptr[i]));
      }
  }

  // Cross Batch Accumulation
  egr::RunBackward(outs, {});
  
  // Examine Backward Grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(tensor);
      auto grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 60.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 60.0, ptr[i]));
      }
  }
  
}
