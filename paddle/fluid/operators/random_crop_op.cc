// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/random_crop_op.h"
#include <vector>

namespace paddle {
namespace operators {
class RandomCropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "");
    AddOutput("Y", "");
    AddInput("Seed", "");
    AddOutput("SeedOut", "").AsDispensable();
    AddAttr<std::vector<int>>("shape", "");
  }
};

class RandomCropOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    auto shape = context->Attrs().Get<std::vector<int>>("shape");
    auto x_dim = context->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), static_cast<int64_t>(shape.size()));
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == -1) {
        shape[i] = static_cast<int>(x_dim[i]);
      } else {
        PADDLE_ENFORCE_GE(x_dim[i], shape[i]);
      }
    }
    context->SetOutputDim("Y", framework::make_ddim(shape));
    context->SetOutputDim("SeedOut", framework::make_ddim({1}));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace f = paddle::framework;
REGISTER_OPERATOR(random_crop, f::OperatorWithKernel, ops::RandomCropOpMaker,
                  ops::RandomCropOpInferShape);
template <typename T>
using Kernel = ops::RandomCropKernel<paddle::platform::CPUDeviceContext, T>;

REGISTER_OP_CPU_KERNEL(random_crop, Kernel<float>, Kernel<int>, Kernel<double>,
                       Kernel<uint8_t>, Kernel<int16_t>);
