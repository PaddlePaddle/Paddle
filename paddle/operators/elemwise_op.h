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
#include <functional>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/string/printf.h"

namespace paddle {
namespace operators {

template <typename T>
class FunctorBase {
 public:
  using ELEM_TYPE = T;

  // A unary functor with attrs must implement that interface. Not use virtual
  // method for speed reason.
  //
  //  void operator()(const T* in, size_t numOfElem, T* out) const;
};

template <typename UnaryFunctor>
class UnaryOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const {
    using T = typename UnaryFunctor::ELEM_TYPE;
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in_buf = in->data<T>();
    auto* out_buf = out->mutable_data<T>(context.GetPlace());
    size_t numel = framework::product(in->dims());
    UnaryFunctor functor_;
    functor_(context, in_buf, numel, out_buf);
  }
};

#ifndef __NVCC__
class UnaryOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());
  }
};

class UnaryOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  UnaryOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", string::Sprintf("the input tensor of unary operator"));
    AddOutput("Out", string::Sprintf("the output tensor of unary operator"));
  }
};
#endif

#define REGISTER_UNARY_OP_AND_CPU_KERNEL(op_type, cpu_functor, maker, \
                                         grad_type, grad_op_type)     \
  REGISTER_OP(op_type, paddle::operators::UnaryOp, maker, grad_type,  \
              grad_op_type);                                          \
  REGISTER_OP_CPU_KERNEL(op_type, paddle::operators::UnaryOpKernel<cpu_functor>)

#define REGISTER_UNARY_OP_GPU_KERNEL(op_type, gpu_functor) \
  REGISTER_OP_GPU_KERNEL(op_type, gpu_functor)

#define DEFINE_UNARY_OP_MAKER(maker_cls, comment)            \
  class maker_cls : public paddle::operators::UnaryOpMaker { \
   public:                                                   \
    using PARENT_CLS = paddle::operators::UnaryOpMaker;      \
    maker_cls(paddle::framework::OpProto* proto,             \
              paddle::framework::OpAttrChecker* op_checker)  \
        : PARENT_CLS(proto, op_checker) {                    \
      AddComment(comment);                                   \
    }                                                        \
  }

#define DEFINE_EIGEN_UNARY_FUNCTOR(functor_cls, method)                      \
  template <typename Place, typename T>                                      \
  class functor_cls : public paddle::operators::FunctorBase<T> {             \
   public:                                                                   \
    void operator()(const paddle::framework::ExecutionContext& context,      \
                    const T* in, size_t numel, T* out) const {               \
      typename paddle::framework::EigenVector<T>::Type a(const_cast<T*>(in), \
                                                         numel);             \
      typename paddle::framework::EigenVector<T>::Type o(out, numel);        \
      auto& dev = context.GetEigenDevice<Place>();                           \
      o.device(dev) = method(a);                                             \
    }                                                                        \
  }

#define REGISTER_UNARY_OP_AND_EIGEN_CPU_KERNEL(                                \
    op_type, eigen_type, dtype, comment, grad_type, grad_op_type)              \
  using __##op_type##_eigen_cpu_functor__ =                                    \
      eigen_type<paddle::platform::CPUPlace, dtype>;                           \
  DEFINE_UNARY_OP_MAKER(__##op_type##_op_maker__, comment);                    \
  REGISTER_UNARY_OP_AND_CPU_KERNEL(op_type, __##op_type##_eigen_cpu_functor__, \
                                   __##op_type##_op_maker__, grad_type,        \
                                   grad_op_type)

#define REGISTER_UNARY_OP_EIGEN_GPU_KERNEL(op_type, eigen_type, dtype) \
  using __##op_type##_eigen_gpu_functor__ =                            \
      eigen_type<paddle::platform::GPUPlace, dtype>;                   \
  REGISTER_UNARY_OP_GPU_KERNEL(op_type, __##op_type##_eigen_gpu_functor__)

}  // namespace operators
}  // namespace paddle
