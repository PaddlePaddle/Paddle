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

#include "paddle/framework/operator.h"
#include "paddle/operators/functors/functor_base.h"
namespace paddle {
namespace operators {
namespace functors {
template <typename Place, typename Functor>
class FunctorKernel : public framework::OpKernel {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    if (Functor::IN_NUM == 1) {  // unary functor
      auto* in = context.Input<framework::Tensor>("X");
      auto* out = context.Output<framework::Tensor>("Out");
      out->mutable_data<typename Functor::ElemType>(context.GetPlace());
      Functor func(context.op_);
      func(context.GetEigenDevice<Place>(), *in, out);
    } else {
      PADDLE_THROW("Not implemented");
    }
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
