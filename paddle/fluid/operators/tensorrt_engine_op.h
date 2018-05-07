/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace inference {
namespace analysis {

template <typename DeviceContext, typename T>
class TensorRTEngineKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {}
};

class TensorRTEngineOp : public framework::OperatorWithKernel {
 protected:
  void Build() {}
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
