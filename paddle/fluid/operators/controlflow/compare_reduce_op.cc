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

#include "paddle/fluid/operators/controlflow/compare_reduce_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename Functor>
class CompareReduceOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    int axis = context.Attr<int>("axis");

    Tensor tmp;
    framework::DDim x_dims = x->dims();
    framework::DDim y_dims = y->dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> tmp_dims_array(max_dim);
    GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                           y_dims_array.data(), tmp_dims_array.data(), max_dim,
                           axis);
    tmp.mutable_data<bool>(framework::make_ddim(tmp_dims_array),
                           context.GetPlace());

    if (x->numel() == 1 && y->numel() == 1) {
      bool* z_data = tmp.mutable_data<bool>(context.GetPlace());
      z_data[0] = Functor()(x->data<T>()[0], y->data<T>()[0]);
    } else {
      ElementwiseComputeEx<Functor, platform::CPUDeviceContext, T, bool>(
          context, x, y, axis, Functor(), &tmp);
    }

    // Reduce by 'logical and' operator
    z->mutable_data<bool>(context.GetPlace());
    auto ipt = framework::EigenVector<bool>::Flatten(tmp);
    auto out = framework::EigenScalar<bool>::From(*z);
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    out.device(place) = ipt.all(reduce_dim);
  }
};

template <typename OpComment>
class CompareReduceOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf("the left hand operand of %s operator",
                                  comment.type));
    AddInput("Y", string::Sprintf("the right hand operand of %s operator",
                                  comment.type));
    AddAttr<int>(
        "axis",
        "The start dimension index for broadcasting Y onto X. [default -1]")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddOutput("Out", string::Sprintf(
                         "tensor with a bool element. If all "
                         "element %s, the Out tensor is [True], else [False]",
                         comment.equation));
    AddComment(string::Sprintf(R"DOC(
It operates element-wise on X and Y, and returns the Out. X, Y is a
N-dim tensor, which could be any type. If all element $%s$, the Out tensor 
is [True], else [False]
)DOC",
                               comment.equation));
  }
};

template <typename OpComment>
class CompareReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OpComment comment;
    PADDLE_ENFORCE_EQ(context->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "%s operator must have input X", comment.type));
    PADDLE_ENFORCE_EQ(context->HasInput("Y"), true,
                      platform::errors::InvalidArgument(
                          "%s operator must have input Y", comment.type));
    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    PADDLE_ENFORCE_GE(
        dim_x.size(), dim_y.size(),
        platform::errors::InvalidArgument(
            "The size of dim_y should not be greater than dim_x's."));

    context->SetOutputDim("Out", {1});
    context->ShareLoD("X", "Out");
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_REDUCE_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                             \
    static char type[];                                                    \
    static char equation[];                                                \
  };                                                                       \
  char _##op_type##Comment::type[]{#op_type};                              \
  char _##op_type##Comment::equation[]{_equation};                         \
  REGISTER_OPERATOR(                                                       \
      op_type, ::paddle::operators::CompareReduceOp<_##op_type##Comment>,  \
      ::paddle::operators::CompareReduceOpProtoMaker<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,    \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

#define REGISTER_COMPARE_REDUCE_CPU_KERNEL(op_type, functor)            \
  REGISTER_OP_CPU_KERNEL(                                               \
      op_type, ::paddle::operators::CompareReduceOpKernel<              \
                   ::paddle::platform::CPUDeviceContext, functor<int>>, \
      ::paddle::operators::CompareReduceOpKernel<                       \
          ::paddle::platform::CPUDeviceContext, functor<int64_t>>,      \
      ::paddle::operators::CompareReduceOpKernel<                       \
          ::paddle::platform::CPUDeviceContext, functor<float>>,        \
      ::paddle::operators::CompareReduceOpKernel<                       \
          ::paddle::platform::CPUDeviceContext, functor<double>>);
REGISTER_COMPARE_REDUCE_OP(equal_reduce, "X == Y");

REGISTER_COMPARE_REDUCE_CPU_KERNEL(equal_reduce,
                                   paddle::operators::EqualReduceFunctor);
