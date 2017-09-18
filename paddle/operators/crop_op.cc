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

#include "paddle/operators/crop_op.h"
#include <boost/lexical_cast.hpp>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::LoDTensor;

class CropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto x_dim = ctx.Input<LoDTensor>("X")->dims();
    auto Y = ctx.Input<LoDTensor>("Y");
    if (Y == nullptr) {
      auto shape = Attr<std::vector<int>>("shape");
      PADDLE_ENFORCE_EQ(
          int64_t(shape.size()), x_dim.size(),
          "Shape size should be equal to dimention size of input tensor.");
      std::vector<int64_t> tensor_shape(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        tensor_shape[i] = (int64_t)shape[i];
      }
      ctx.Output<LoDTensor>("Out")->Resize(framework::make_ddim(tensor_shape));
    } else {
      ctx.Output<LoDTensor>("Out")->Resize(Y->dims());
    }
  }
};

class CropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CropOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddInput("Y",
             "The input used as reference for cropping"
             " with the same dimension as X. ");
    AddOutput("Out",
              "The output of crop op "
              "with the same dimension as X.");
    AddComment(R"DOC(
Crop Operator.
Crop input into output, as specified by offsets and shape.

There are two ways to set shape: 
1. referenc input: crop input X as shape as reference input.
                    The dimension of reference input should 
                    be as same as input X.
2. shape list: crop input X by shape described by a list<int>.
               The size of shape list should be as same as 
               dimension size of  input X.

The input should be a k-D tensor(k > 0 and k < 7). As an example:

Given:

X = [[0, 1, 2, 0, 0]
       [0, 3, 4, 0, 0]
       [0, 0, 0, 0, 0]]

and 

offsets = [0, 1]

and
 
shape = [2, 2]

then we get 

Out = [[1, 2],
   [3, 4]]

)DOC");
    AddAttr<std::vector<int>>("offsets",
                              "A list<int> describing offsets to be cropped."
                              "The size of offsets list should be as same as "
                              "dimension size of  input X.");
    AddAttr<std::vector<int>>("shape",
                              "A list<int> describing the shape of output."
                              "The size of shape list should be as same as "
                              "dimension size of  input X.")
        .SetDefault(std::vector<int>());
  }
};

class CropOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<LoDTensor>("X")->dims();
    auto *x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    if (x_grad != nullptr) {
      x_grad->Resize(x_dims);
    }
  }
};

int64_t transIndex(std::vector<int64_t> out_shape, std::vector<int64_t> x_shape,
                   std::vector<std::pair<int, int>> crop_rules, size_t index) {
  int64_t dim_size = out_shape.size();
  std::vector<int64_t> pos(dim_size);

  for (int64_t i = out_shape.size() - 1; i >= 0; --i) {
    pos[i] = (index % out_shape[i]) + crop_rules[i].first;
    index = index / out_shape[i];
  }

  size_t result = pos[0];
  for (size_t i = 1; i < x_shape.size(); ++i) {
    result = result * x_shape[i] + pos[i];
  }
  return result;
}

template <typename T>
class CropCPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *out = context.Output<LoDTensor>("Out");
    auto x_data = x->data<T>();
    T *out_data = out->mutable_data<T>(context.GetPlace());
    auto x_dims = x->dims();
    auto out_dims = out->dims();
    int64_t out_count = framework::product(out_dims);
    std::vector<int64_t> x_shape = framework::vectorize(x_dims);
    std::vector<int64_t> out_shape = framework::vectorize(out_dims);

    auto offsets = context.op().Attr<std::vector<int>>("offsets");
    PADDLE_ENFORCE_EQ(
        x_dims.size(), offsets.size(),
        "Offsets size should be equal to dimension size of input tensor.");

    std::vector<std::pair<int, int>> crop_rules(x_dims.size());
    for (size_t i = 0; i < crop_rules.size(); ++i) {
      crop_rules[i].first = offsets[i];
      crop_rules[i].second = x_dims[i] - out_dims[i] - offsets[i];
    }

    for (int64_t i = 0; i < out_count; ++i) {
      out_data[i] = x_data[transIndex(out_shape, x_shape, crop_rules, i)];
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(crop, ops::CropOp, ops::CropOpMaker, crop_grad, ops::CropOpGrad);
REGISTER_OP_CPU_KERNEL(crop, ops::CropCPUKernel<float>);
REGISTER_OP_CPU_KERNEL(crop_grad,
                       ops::CropGradKernel<paddle::platform::CPUPlace, float>);
