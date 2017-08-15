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

#include "paddle/operators/fill_op.h"

namespace paddle {
namespace operators {

template <typename T>
class FillOp : public framework::OperatorWithKernel {
 public:
  FillOp(const std::string &type, const VarNameMap &inputs,
         const VarNameMap &outputs, const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto &shape = GetAttr<std::vector<int>>("shape");
    auto dim = framework::make_ddim(shape);
    auto numel = framework::product(dim);
    PADDLE_ENFORCE_EQ(numel, GetAttr<std::vector<T>>("data").size(),
                      "Shape's numel should be as same as data element count");
    ctx.Output<framework::Tensor>("Out")->Resize(dim);
  }
};

template <typename T>
class FillOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FillOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "Output of Fill Op");
    AddComment("Fill a variable with shape and buffer each time.");
    AddAttr<int>("run_once", "Set it once or each time when run")
        .SetDefault(false)
        .InEnum({true, false});
    AddAttr<std::vector<int>>("shape", "The shape of fill parameter");
    AddAttr<std::vector<T>>("data", "The data will be filled");
  }
};

template <typename T>
class FillOpCPUKernel : public FillOpKernelBase<T> {
 public:
  void Copy(const platform::Place &place, const std::vector<T> &src,
            T *dst) const override {
    std::copy(src.begin(), src.end(), dst);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(fill, ops::FillOp<float>, ops::FillOpMaker<float>);
REGISTER_OP_CPU_KERNEL(fill, ops::FillOpCPUKernel<float>);
