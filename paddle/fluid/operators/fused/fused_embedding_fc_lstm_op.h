/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

<<<<<<< HEAD
=======
using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
class FusedEmbeddingFCLSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
<<<<<<< HEAD
  phi::KernelKey GetExpectedKernelType(
=======
  framework::OpKernelType GetExpectedKernelType(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      const framework::ExecutionContext& ctx) const override;
};

class FusedEmbeddingFCLSTMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

}  // namespace operators
}  // namespace paddle
