/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/fully_connected_op.h"
#include <iostream>
namespace paddle {
namespace framework {

void FCOp::Run(const ScopePtr& scope,
               const platform::DeviceContext& dev_ctx) const override {
  std::cout << "FC" << std::endl;
}

void FCOp::InferShape(const ScopePtr& scope) const override {}

void FCGradientOp::Run(const ScopePtr& scope,
                       const platform::DeviceContext& dev_ctx) const override {
  std::cout << "FCGrad" << std::endl;
}

void FCGradientOp::InferShape(const ScopePtr& scope) const override {}

REGISTER_OP(my_fc, paddle::framework::FCOp,
            paddle::framework::FCOpProtoAndCheckerMaker);
REGISTER_OP(my_fc_grad, paddle::framework::FCGradientOp,
            paddle::framework::FCGradientOpProtoAndCheckerMaker);
}  // namespace framework
}  // namespace paddle
