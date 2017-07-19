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

#pragma once
#include "glog/logging.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::OpProtoAndCheckerMaker;
using framework::OpKernel;
using framework::KernelContext;

template <size_t InputN, size_t OutputN>
class FixSizeOp : public framework::OperatorWithKernel {
protected:
  template <typename Container>
  void EnforceAllNotNull(Container c, const char *var_name) const {
    size_t idx = 0;
    for (auto &ptr : c) {
      PADDLE_ENFORCE(
          ptr != nullptr, "%s %d of %s op must be set", var_name, idx, type_);
      ++idx;
    }
  }

  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == InputN,
                   "number of %s input must be %d",
                   type_,
                   InputN);
    PADDLE_ENFORCE(outputs.size() == OutputN,
                   "number of %s output must be %d",
                   type_,
                   OutputN);
    EnforceAllNotNull(inputs, "input");
    EnforceAllNotNull(outputs, "output");
  }
};

template <size_t N>
class ElemwiseOp : public FixSizeOp<N, 1> {
protected:
  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {
    FixSizeOp<N, 1>::InferShape(inputs, outputs);
    auto dim = inputs[0]->dims();
    for (size_t i = 1; i < inputs.size(); ++i) {
      PADDLE_ENFORCE(dim == inputs[i]->dims(),
                     "%s's input must be same shape",
                     this->type_);
    }
    outputs[0]->set_dims(inputs[0]->dims());
  }
};

template <typename Place>
class FakeKernel : public OpKernel {
public:
  virtual void Compute(const KernelContext &context) const {
    LOG(INFO) << "Run " << context.op_.type_ << "Kernel in "
              << typeid(Place).name();
  }
};

//! Add Common kernel here, such as elemwise kernel.

}  // namespace operators
}  // namespace paddle
